#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible streaming chat endpoint with live JSON progress.

Writes final per-request rows to JSONL (same schema as benchmark_openai_stream.py),
and optionally updates a live JSON file while tokens are arriving so progress is visible.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_prompts(
    path,
    categories,
    limit_per_category,
    mandatory_only,
):
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    if categories:
        prompts = [p for p in prompts if p.get("category") in categories]
    if mandatory_only:
        prompts = [p for p in prompts if p.get("mandatory") is True]
    if limit_per_category is not None:
        mandatory_counts = {}
        for prompt in prompts:
            if prompt.get("mandatory") is True:
                category = prompt.get("category", "unknown")
                mandatory_counts[category] = mandatory_counts.get(category, 0) + 1
        counts = {}
        selected = []
        for prompt in prompts:
            category = prompt.get("category", "unknown")
            counts.setdefault(category, 0)
            if prompt.get("mandatory") is True:
                selected.append(prompt)
            else:
                nonmandatory_limit = max(0, limit_per_category - mandatory_counts.get(category, 0))
                if counts[category] < nonmandatory_limit:
                    selected.append(prompt)
                    counts[category] += 1
        prompts = selected
    return data.get("generation", {}), prompts


def parse_sse_line(line):
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def extract_delta(event):
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


class LiveProgressWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.entries = {}
        self.started = 0
        self.completed = 0
        self.failed = 0

    def _dump_locked(self):
        payload = {
            "updated_at_epoch_s": time.time(),
            "started": self.started,
            "completed": self.completed,
            "failed": self.failed,
            "in_progress": max(self.started - self.completed, 0),
            "requests": [self.entries[k] for k in sorted(self.entries.keys())],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def start(self, request_key, prompt, trial):
        with self.lock:
            self.started += 1
            self.entries[request_key] = {
                "request_key": request_key,
                "status": "running",
                "prompt_id": prompt.get("id"),
                "category": prompt.get("category"),
                "mandatory": bool(prompt.get("mandatory", False)),
                "trial": trial,
                "output_token_events": 0,
                "generated_text": "",
                "ttft_s": None,
                "error": None,
            }
            self._dump_locked()

    def token(self, request_key, generated_text, output_token_events, ttft_s):
        with self.lock:
            entry = self.entries.get(request_key)
            if not entry:
                return
            entry["generated_text"] = generated_text
            entry["output_token_events"] = output_token_events
            entry["ttft_s"] = ttft_s
            self._dump_locked()

    def finish(self, request_key, row):
        with self.lock:
            entry = self.entries.get(request_key)
            if not entry:
                return
            entry["status"] = "done" if row["error"] is None else "error"
            entry["error"] = row["error"]
            entry["status_code"] = row["status"]
            entry["ttft_s"] = row["ttft_s"]
            entry["tpot_s"] = row["tpot_s"]
            entry["total_s"] = row["total_s"]
            entry["throughput_output_tok_s"] = row["throughput_output_tok_s"]
            entry["output_token_events"] = row["output_token_events"]
            entry["generated_text"] = row["generated_text"]
            self.completed += 1
            if row["error"] is not None:
                self.failed += 1
            self._dump_locked()


def run_one(
    url,
    model,
    prompt,
    trial,
    max_tokens,
    temperature,
    top_p,
    seed,
    timeout,
    request_key,
    live_writer=None,
):
    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["text"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if seed is not None:
        request_body["seed"] = seed
    body = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token_time = None
    token_times = []
    pieces = []
    error = None
    status = None

    if live_writer is not None:
        live_writer.start(request_key, prompt, trial)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                event = parse_sse_line(raw_line)
                if not event:
                    continue
                if event.get("done"):
                    break
                delta = extract_delta(event)
                if delta:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    token_times.append(now)
                    pieces.append(delta)
                    if live_writer is not None:
                        ttft_s = None if first_token_time is None else first_token_time - start
                        live_writer.token(request_key, "".join(pieces), len(token_times), ttft_s)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)

    end = time.perf_counter()
    generated = "".join(pieces)
    output_token_events = len(token_times)
    ttft_s = None if first_token_time is None else first_token_time - start
    inter_token = [b - a for a, b in zip(token_times, token_times[1:])]
    tpot_s = statistics.mean(inter_token) if inter_token else None
    total_s = end - start
    throughput_tok_s = output_token_events / total_s if total_s > 0 else None

    row = {
        "prompt_id": prompt["id"],
        "category": prompt.get("category"),
        "mandatory": bool(prompt.get("mandatory", False)),
        "trial": trial,
        "status": status,
        "error": error,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "total_s": total_s,
        "output_token_events": output_token_events,
        "throughput_output_tok_s": throughput_tok_s,
        "generated_text": generated,
    }
    if live_writer is not None:
        live_writer.finish(request_key, row)
    return row


def write_summary(rows, path):
    groups = {}
    for row in rows:
        groups.setdefault((row["category"], row["prompt_id"]), []).append(row)

    summary = []
    for (category, prompt_id), items in sorted(groups.items()):
        clean = [r for r in items if r["error"] is None]

        def mean_of(key):
            values = [r[key] for r in clean if r.get(key) is not None]
            return statistics.mean(values) if values else None

        def stdev_of(key):
            values = [r[key] for r in clean if r.get(key) is not None]
            return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None

        summary.append(
            {
                "category": category,
                "prompt_id": prompt_id,
                "runs": len(items),
                "successful_runs": len(clean),
                "ttft_s_mean": mean_of("ttft_s"),
                "ttft_s_stdev": stdev_of("ttft_s"),
                "tpot_s_mean": mean_of("tpot_s"),
                "tpot_s_stdev": stdev_of("tpot_s"),
                "throughput_output_tok_s_mean": mean_of("throughput_output_tok_s"),
                "throughput_output_tok_s_stdev": stdev_of("throughput_output_tok_s"),
            }
        )
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--prompts", type=Path, default=Path("prompts/track_a_prompts.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--live-out", type=Path, help="Live JSON progress file updated during generation.")
    parser.add_argument("--categories", default="short,medium,long")
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--mandatory-only", action="store_true", help="Run only prompts marked mandatory in the prompt JSON.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-trials", type=int, default=0, help="Run this many full prompt-set trials before recording measurements.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int, default=600)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    categories = {c.strip() for c in args.categories.split(",") if c.strip()}
    generation_defaults, prompts = load_prompts(args.prompts, categories, args.limit_per_category, args.mandatory_only)
    if not prompts:
        print("No prompts selected; check --categories, --limit-per-category, and --mandatory-only.", file=sys.stderr)
        return 2
    temperature = args.temperature if args.temperature is not None else float(generation_defaults.get("temperature", 0.0))
    top_p = args.top_p if args.top_p is not None else float(generation_defaults.get("top_p", 1.0))
    seed = args.seed if args.seed is not None else generation_defaults.get("seed")
    seed = int(seed) if seed is not None else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    live_writer = LiveProgressWriter(args.live_out) if args.live_out else None

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        warmup_jobs = []
        for trial in range(1, args.warmup_trials + 1):
            for prompt in prompts:
                request_key = f"warmup-{trial}:{prompt['id']}"
                warmup_jobs.append(
                    executor.submit(
                        run_one,
                        args.url,
                        args.model,
                        prompt,
                        -trial,
                        args.max_tokens,
                        temperature,
                        top_p,
                        seed,
                        args.timeout,
                        request_key,
                        None,
                    )
                )
        for future in as_completed(warmup_jobs):
            row = future.result()
            if row["error"]:
                print(f"[WARN] warmup {row['prompt_id']} trial={abs(row['trial'])}: {row['error']}", file=sys.stderr)

        jobs = []
        for trial in range(1, args.trials + 1):
            for prompt in prompts:
                request_key = f"trial-{trial}:{prompt['id']}"
                jobs.append(
                    executor.submit(
                        run_one,
                        args.url,
                        args.model,
                        prompt,
                        trial,
                        args.max_tokens,
                        temperature,
                        top_p,
                        seed,
                        args.timeout,
                        request_key,
                        live_writer,
                    )
                )

        with args.out.open("w", encoding="utf-8") as output:
            for future in as_completed(jobs):
                row = future.result()
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                if row["error"]:
                    print(f"[WARN] {row['prompt_id']} trial={row['trial']}: {row['error']}", file=sys.stderr)

    summary_path = args.summary_out or args.out.with_suffix(".summary.json")
    write_summary(rows, summary_path)
    return 0 if all(row["error"] is None for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
