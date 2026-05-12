#!/usr/bin/env python3
"""Benchmark llama.cpp's llama-cli for Track A1 CPU inference experiments.

The runner writes the same basic files as the HTTP benchmark harness:

- requests.jsonl: one raw row per measured prompt/trial, including output text
- summary.json: per-prompt mean/stdev metrics and a representative output
- resources.csv: sampled RSS/VmHWM for each llama-cli child process
- system.txt: run configuration for downstream summarizers

TTFT is measured as the time from process start to the first bytes observed on
stdout. TPOT and prompt/decode throughput are parsed from llama.cpp's
llama_perf_context_print lines when available.
"""

import argparse
import csv
import json
import os
import platform
import re
import selectors
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens"
    r".*?([0-9.]+)\s*tokens per second",
    re.IGNORECASE,
)
EVAL_RE = re.compile(
    r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*(?:runs|tokens)"
    r".*?([0-9.]+)\s*tokens per second",
    re.IGNORECASE,
)


def load_prompts(path, categories, limit_per_category, mandatory_only):
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    if categories:
        prompts = [p for p in prompts if p.get("category") in categories]
    if mandatory_only:
        prompts = [p for p in prompts if p.get("mandatory") is True]
    if limit_per_category is not None:
        counts = {}
        selected = []
        for prompt in prompts:
            category = prompt.get("category", "unknown")
            counts.setdefault(category, 0)
            if counts[category] < limit_per_category:
                selected.append(prompt)
                counts[category] += 1
        prompts = selected
    return data.get("generation", {}), prompts


def parse_llamacpp_timings(stderr_text):
    timings = {
        "prompt_eval_s": None,
        "prompt_tokens": None,
        "prompt_tokens_per_s": None,
        "eval_s": None,
        "eval_tokens": None,
        "eval_tokens_per_s": None,
    }
    for line in stderr_text.splitlines():
        prompt_match = PROMPT_RE.search(line)
        if prompt_match:
            timings["prompt_eval_s"] = float(prompt_match.group(1)) / 1000.0
            timings["prompt_tokens"] = int(prompt_match.group(2))
            timings["prompt_tokens_per_s"] = float(prompt_match.group(3))
            continue
        eval_match = EVAL_RE.search(line)
        if eval_match:
            timings["eval_s"] = float(eval_match.group(1)) / 1000.0
            timings["eval_tokens"] = int(eval_match.group(2))
            timings["eval_tokens_per_s"] = float(eval_match.group(3))
    return timings


def status_values(pid):
    status_path = Path("/proc") / str(pid) / "status"
    values = {"rss_kb": None, "vmhwm_kb": None}
    try:
        with status_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    values["rss_kb"] = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    values["vmhwm_kb"] = int(line.split()[1])
    except OSError:
        pass
    return values


def sample_resources(process, stop_event, sample_interval_s, prompt_id, trial, samples):
    while not stop_event.is_set() and process.poll() is None:
        values = status_values(process.pid)
        samples.append({
            "timestamp_s": time.time(),
            "prompt_id": prompt_id,
            "trial": trial,
            "pid": process.pid,
            "rss_kb": values["rss_kb"],
            "vmhwm_kb": values["vmhwm_kb"],
        })
        time.sleep(sample_interval_s)
    values = status_values(process.pid)
    if values["rss_kb"] is not None or values["vmhwm_kb"] is not None:
        samples.append({
            "timestamp_s": time.time(),
            "prompt_id": prompt_id,
            "trial": trial,
            "pid": process.pid,
            "rss_kb": values["rss_kb"],
            "vmhwm_kb": values["vmhwm_kb"],
        })


def build_command(args, prompt, temperature, top_p, seed):
    command = [
        str(args.llama_cli),
        "-m",
        str(args.model),
        "-p",
        prompt["text"],
        "-n",
        str(args.max_tokens),
        "-t",
        str(args.threads),
        "-c",
        str(args.ctx_size),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
        "--no-display-prompt",
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    for extra_arg in args.extra_arg or []:
        command.append(extra_arg)
    return command


def read_process_output(process, timeout_s):
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")

    stdout_parts = []
    stderr_parts = []
    start = time.perf_counter()
    first_stdout_s = None

    while selector.get_map():
        if timeout_s and time.perf_counter() - start > timeout_s:
            process.kill()
            raise TimeoutError(f"llama-cli timed out after {timeout_s} seconds")
        for key, _ in selector.select(timeout=0.1):
            chunk = os.read(key.fileobj.fileno(), 4096)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            if key.data == "stdout":
                if first_stdout_s is None:
                    first_stdout_s = time.perf_counter() - start
                stdout_parts.append(chunk)
            else:
                stderr_parts.append(chunk)
        if process.poll() is not None and not selector.get_map():
            break

    return (
        b"".join(stdout_parts).decode("utf-8", errors="replace"),
        b"".join(stderr_parts).decode("utf-8", errors="replace"),
        first_stdout_s,
        time.perf_counter() - start,
    )


def run_one(args, prompt, trial, temperature, top_p, seed):
    command = build_command(args, prompt, temperature, top_p, seed)
    start_wall = datetime.now(timezone.utc).isoformat()
    samples = []
    stop_event = threading.Event()
    error = None
    stdout_text = ""
    stderr_text = ""
    first_stdout_s = None
    total_s = None
    return_code = None

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    monitor = threading.Thread(
        target=sample_resources,
        args=(process, stop_event, args.resource_interval, prompt["id"], trial, samples),
        daemon=True,
    )
    monitor.start()
    try:
        stdout_text, stderr_text, first_stdout_s, total_s = read_process_output(process, args.timeout)
        return_code = process.wait(timeout=5)
    except (TimeoutError, OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
        if process.poll() is None:
            process.kill()
        return_code = process.wait()
        total_s = total_s if total_s is not None else None
    finally:
        stop_event.set()
        monitor.join(timeout=2)

    timings = parse_llamacpp_timings(stderr_text)
    eval_tokens = timings["eval_tokens"]
    eval_s = timings["eval_s"]
    if error is None and return_code != 0:
        error = f"llama-cli exited with status {return_code}"

    tpot_s = None
    if eval_tokens and eval_s and eval_tokens > 0:
        tpot_s = eval_s / eval_tokens
    elif first_stdout_s is not None and total_s is not None and eval_tokens and eval_tokens > 1:
        tpot_s = max(total_s - first_stdout_s, 0.0) / (eval_tokens - 1)

    throughput = None
    if timings["eval_tokens_per_s"] is not None:
        throughput = timings["eval_tokens_per_s"]
    elif eval_tokens and total_s and total_s > 0:
        throughput = eval_tokens / total_s

    max_rss_kb = max((s["rss_kb"] for s in samples if s["rss_kb"] is not None), default=None)
    max_vmhwm_kb = max((s["vmhwm_kb"] for s in samples if s["vmhwm_kb"] is not None), default=None)
    row = {
        "prompt_id": prompt["id"],
        "category": prompt.get("category"),
        "mandatory": bool(prompt.get("mandatory", False)),
        "trial": trial,
        "started_at": start_wall,
        "status": return_code,
        "error": error,
        "model": str(args.model),
        "llama_cli": str(args.llama_cli),
        "threads": args.threads,
        "ctx_size": args.ctx_size,
        "max_tokens": args.max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "ttft_s": first_stdout_s,
        "tpot_s": tpot_s,
        "total_s": total_s,
        "prompt_eval_s": timings["prompt_eval_s"],
        "prompt_tokens": timings["prompt_tokens"],
        "prompt_throughput_tok_s": timings["prompt_tokens_per_s"],
        "eval_s": eval_s,
        "output_token_events": eval_tokens,
        "throughput_output_tok_s": throughput,
        "max_rss_kb": max_rss_kb,
        "max_vmhwm_kb": max_vmhwm_kb,
        "generated_text": stdout_text,
        "stderr_tail": stderr_text[-4000:],
        "command": command,
    }
    return row, samples


def mean(values):
    return statistics.mean(values) if values else None


def stdev(values):
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_summary(rows, path):
    groups = {}
    for row in rows:
        groups.setdefault((row["category"], row["prompt_id"]), []).append(row)

    summary = []
    for (category, prompt_id), items in sorted(groups.items()):
        clean = [r for r in items if r.get("error") is None]

        def values(key):
            return [float(r[key]) for r in clean if r.get(key) is not None]

        representative = next((r.get("generated_text", "") for r in clean if r.get("generated_text")), "")
        summary.append({
            "category": category,
            "prompt_id": prompt_id,
            "mandatory": any(r.get("mandatory") for r in items),
            "runs": len(items),
            "successful_runs": len(clean),
            "ttft_s_mean": mean(values("ttft_s")),
            "ttft_s_stdev": stdev(values("ttft_s")),
            "tpot_s_mean": mean(values("tpot_s")),
            "tpot_s_stdev": stdev(values("tpot_s")),
            "prompt_throughput_tok_s_mean": mean(values("prompt_throughput_tok_s")),
            "prompt_throughput_tok_s_stdev": stdev(values("prompt_throughput_tok_s")),
            "throughput_output_tok_s_mean": mean(values("throughput_output_tok_s")),
            "throughput_output_tok_s_stdev": stdev(values("throughput_output_tok_s")),
            "max_rss_kb": max(values("max_rss_kb"), default=None),
            "max_vmhwm_kb": max(values("max_vmhwm_kb"), default=None),
            "representative_generated_text": representative,
        })
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_system(args, path, prompt_count, temperature, top_p, seed):
    lines = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": "llamacpp-cli",
        "node": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "llama_cli": str(args.llama_cli),
        "model": str(args.model),
        "threads": str(args.threads),
        "ctx_size": str(args.ctx_size),
        "max_tokens": str(args.max_tokens),
        "trials": str(args.trials),
        "warmup_trials": str(args.warmup_trials),
        "mandatory_only": str(int(args.mandatory_only)),
        "categories": args.categories,
        "prompt_count": str(prompt_count),
        "temperature": str(temperature),
        "top_p": str(top_p),
        "seed": "" if seed is None else str(seed),
    }
    path.write_text("".join(f"{key}={value}\n" for key, value in lines.items()), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", type=Path, required=True, help="Path to llama.cpp's llama-cli binary.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a GGUF model.")
    parser.add_argument("--prompts", type=Path, default=Path("prompts/track_a_prompts.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--categories", default="short,medium,long")
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--mandatory-only", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resource-interval", type=float, default=0.25)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra single argument passed to llama-cli. Repeat as needed; use --extra-arg=--flag for flags.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.llama_cli.exists():
        print(f"missing llama-cli binary: {args.llama_cli}", file=sys.stderr)
        return 2
    if not args.model.exists():
        print(f"missing model file: {args.model}", file=sys.stderr)
        return 2

    categories = {c.strip() for c in args.categories.split(",") if c.strip()}
    generation_defaults, prompts = load_prompts(
        args.prompts,
        categories,
        args.limit_per_category,
        args.mandatory_only,
    )
    if not prompts:
        print("No prompts selected; check --categories, --limit-per-category, and --mandatory-only.", file=sys.stderr)
        return 2

    temperature = args.temperature if args.temperature is not None else float(generation_defaults.get("temperature", 0.0))
    top_p = args.top_p if args.top_p is not None else float(generation_defaults.get("top_p", 1.0))
    seed = args.seed if args.seed is not None else generation_defaults.get("seed")
    seed = int(seed) if seed is not None else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = args.out_dir / "requests.jsonl"
    resources_path = args.out_dir / "resources.csv"
    summary_path = args.out_dir / "summary.json"
    write_system(args, args.out_dir / "system.txt", len(prompts), temperature, top_p, seed)

    rows = []
    resource_fieldnames = ["timestamp_s", "prompt_id", "trial", "pid", "rss_kb", "vmhwm_kb"]
    with resources_path.open("w", newline="", encoding="utf-8") as resource_file:
        resource_writer = csv.DictWriter(resource_file, fieldnames=resource_fieldnames)
        resource_writer.writeheader()

        for trial in range(1, args.warmup_trials + 1):
            for prompt in prompts:
                row, samples = run_one(args, prompt, -trial, temperature, top_p, seed)
                for sample in samples:
                    resource_writer.writerow(sample)
                resource_file.flush()
                if row["error"]:
                    print(f"[WARN] warmup {prompt['id']} trial={trial}: {row['error']}", file=sys.stderr)

        with requests_path.open("w", encoding="utf-8") as output:
            for trial in range(1, args.trials + 1):
                for prompt in prompts:
                    row, samples = run_one(args, prompt, trial, temperature, top_p, seed)
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    for sample in samples:
                        resource_writer.writerow(sample)
                    resource_file.flush()
                    if row["error"]:
                        print(f"[WARN] {prompt['id']} trial={trial}: {row['error']}", file=sys.stderr)

    write_summary(rows, summary_path)
    print(f"wrote {len(rows)} request rows to {requests_path}")
    print(f"wrote summary to {summary_path}")
    print(f"wrote resource samples to {resources_path}")
    return 0 if all(r.get("error") is None for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
