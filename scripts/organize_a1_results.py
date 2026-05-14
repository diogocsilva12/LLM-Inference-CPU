#!/usr/bin/env python3
"""Organize Track A1 sweep summaries into report-ready tables and TeX."""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_SWEEPS = [
    ("arm_server", "ARM server sweep", "arm", Path("measurements/1253766-track-a1-server-sweep")),
    ("x86_server", "x86 server sweep", "x86", Path("measurements/1254633-track-a1-server-sweep-x86")),
    ("blas_server", "ARM BLAS sweep", "arm", Path("measurements/1254637-track-a1-vanilla-blas-sweep")),
]

MODEL_DIR = Path("llamacpp-tq/models")
MANDATORY_PROMPTS = {
    "mandatory_short_capital_france",
    "mandatory_medium_ml_vs_dl",
    "mandatory_long_transformer_cpu",
}


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def safe_int(value):
    value = safe_float(value)
    return int(value) if value is not None else None


def fmt(value, digits=3):
    value = safe_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt1(value):
    value = safe_float(value)
    if value is None:
        return ""
    return f"{value:.1f}"


def mean(values):
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def mandatory(row):
    return row.get("prompt_id") in MANDATORY_PROMPTS


def successful(row):
    return safe_int(row.get("successful_runs")) and safe_int(row.get("successful_runs")) > 0


def add_provenance(rows, sweep_id, sweep_name, arch, source):
    out = []
    for row in rows:
        copy = dict(row)
        copy["sweep_id"] = sweep_id
        copy["sweep_name"] = sweep_name
        copy["arch"] = arch
        copy["source_run_dir"] = str(source)
        out.append(copy)
    return out


def collect_summaries(sweeps):
    prompt_rows = []
    scaling_rows = []
    for sweep_id, sweep_name, arch, root in sweeps:
        summary = root / "a1_server_summary.csv"
        scaling = root / "a1_server_scaling_summary.csv"
        if not summary.exists() or not scaling.exists():
            raise FileNotFoundError(f"missing summaries for {root}")
        prompt_rows.extend(add_provenance(read_csv(summary), sweep_id, sweep_name, arch, root))
        scaling_rows.extend(add_provenance(read_csv(scaling), sweep_id, sweep_name, arch, root))
    return prompt_rows, scaling_rows


def aggregate(rows, group_keys):
    groups = {}
    for row in rows:
        if not mandatory(row) or not successful(row):
            continue
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)

    out = []
    for key, items in sorted(groups.items()):
        row = {group_keys[index]: value for index, value in enumerate(key)}
        row["mandatory_prompt_rows"] = len(items)
        row["successful_runs"] = sum(safe_int(item.get("successful_runs")) or 0 for item in items)
        row["ttft_s_mean"] = mean([safe_float(item.get("ttft_s_mean_across_replicas")) for item in items])
        row["tpot_s_mean"] = mean([safe_float(item.get("tpot_s_mean_across_replicas")) for item in items])
        row["throughput_tok_s_mean"] = mean([safe_float(item.get("total_throughput_output_tok_s_mean")) for item in items])
        row["goodput_fraction_mean"] = mean([safe_float(item.get("goodput_sla_fraction_mean")) for item in items])
        row["max_vmhwm_gib"] = max(
            [(safe_float(item.get("max_vmhwm_kb")) or 0.0) / (1024.0 * 1024.0) for item in items],
            default=None,
        )
        out.append(row)
    return out


def table_rows(scaling_rows, suite):
    return [row for row in scaling_rows if row.get("suite") == suite]


def model_basename(row):
    model = row.get("model", "")
    return Path(model).name if model else ""


def local_model_size_bytes(model_name):
    if not model_name:
        return None
    path = MODEL_DIR / model_name
    return path.stat().st_size if path.exists() else None


def build_performance_model(baseline_rows):
    rows = []
    for row in baseline_rows:
        tpot = safe_float(row.get("tpot_s_mean"))
        if tpot is None or tpot <= 0:
            continue
        model_name = model_basename(row)
        model_bytes = local_model_size_bytes(model_name)
        if not model_bytes:
            continue
        rows.append({
            **row,
            "model_file": model_name,
            "model_bytes": model_bytes,
            "effective_bw_gb_s": model_bytes / tpot / 1.0e9,
        })

    bw_by_sweep = {}
    for row in rows:
        key = row["sweep_id"]
        bw_by_sweep[key] = max(bw_by_sweep.get(key, 0.0), row["effective_bw_gb_s"])

    model_rows = []
    for row in rows:
        bw = bw_by_sweep.get(row["sweep_id"])
        pred = row["model_bytes"] / (bw * 1.0e9) if bw else None
        observed = safe_float(row.get("tpot_s_mean"))
        model_rows.append({
            "sweep_id": row["sweep_id"],
            "arch": row["arch"],
            "engine": row["engine"],
            "model_label": row["model_label"],
            "model_file": row["model_file"],
            "cache_type_k": row["cache_type_k"],
            "cache_type_v": row["cache_type_v"],
            "model_gib": row["model_bytes"] / (1024.0 ** 3),
            "calibrated_bw_gb_s": bw,
            "tpot_pred_s": pred,
            "tpot_observed_s": observed,
            "observed_over_predicted": observed / pred if pred and pred > 0 else None,
            "throughput_tok_s_mean": row.get("throughput_tok_s_mean"),
            "max_vmhwm_gib": row.get("max_vmhwm_gib"),
        })
    return model_rows


def latex_escape(value):
    value = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def latex_table(caption, label, headers, rows, columns):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + "l" * len(headers) + r"}",
        r"\hline",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(col, "")) for col in columns) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def display_metric_rows(rows, limit=None):
    out = []
    for row in rows[:limit] if limit else rows:
        out.append({
            **row,
            "ttft": fmt(row.get("ttft_s_mean")),
            "tpot_ms": fmt((safe_float(row.get("tpot_s_mean")) or 0.0) * 1000.0, 1),
            "throughput": fmt(row.get("throughput_tok_s_mean")),
            "goodput": fmt(row.get("goodput_fraction_mean")),
            "mem_gib": fmt(row.get("max_vmhwm_gib"), 2),
        })
    return out


def readable_metric_rows(rows):
    readable = []
    for row in rows:
        readable.append({
            "Sweep": row.get("sweep_id", ""),
            "Architecture": row.get("arch", ""),
            "Engine": row.get("engine", ""),
            "Model": row.get("model_label", ""),
            "Threads": row.get("threads", ""),
            "Context tokens": row.get("ctx_size", ""),
            "Concurrent requests": row.get("parallel_requests", ""),
            "KV cache K/V": f"{row.get('cache_type_k', '')}/{row.get('cache_type_v', '')}",
            "Max generated tokens": row.get("max_tokens", ""),
            "TTFT mean (seconds)": fmt(row.get("ttft_s_mean")),
            "TPOT mean (milliseconds/token)": fmt((safe_float(row.get("tpot_s_mean")) or 0.0) * 1000.0, 1),
            "Throughput mean (output tokens/second)": fmt(row.get("throughput_tok_s_mean")),
            "Goodput fraction (TTFT<=2s and TPOT<=200ms)": fmt(row.get("goodput_fraction_mean")),
            "Peak memory VmHWM (GiB)": fmt(row.get("max_vmhwm_gib"), 2),
            "Successful measured requests": row.get("successful_runs", ""),
        })
    return readable


def write_readable_table(path, rows):
    fieldnames = [
        "Sweep",
        "Architecture",
        "Engine",
        "Model",
        "Threads",
        "Context tokens",
        "Concurrent requests",
        "KV cache K/V",
        "Max generated tokens",
        "TTFT mean (seconds)",
        "TPOT mean (milliseconds/token)",
        "Throughput mean (output tokens/second)",
        "Goodput fraction (TTFT<=2s and TPOT<=200ms)",
        "Peak memory VmHWM (GiB)",
        "Successful measured requests",
    ]
    write_csv(path, readable_metric_rows(rows), fieldnames)


def write_metrics_guide(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# Metrics Guide",
            "",
            "Use the files ending in `_readable.csv` for report-facing tables. They keep units in the column names.",
            "",
            "| Metric | Unit | Meaning |",
            "| --- | --- | --- |",
            "| TTFT mean | seconds | Time elapsed from HTTP request submission until the first streamed output token. This mostly reflects prompt prefill, scheduling, and startup overhead. Lower is better. |",
            "| TPOT mean | milliseconds per output token | Average elapsed time between generated tokens during decode. This is the main latency metric for steady-state generation. Lower is better. |",
            "| Throughput mean | output tokens per second | Generated tokens completed per second for the measured request stream. Higher is better. |",
            "| Goodput fraction | fraction from 0 to 1 | Share of requests meeting the SLA used by the harness: TTFT <= 2.0 s and TPOT <= 200 ms. Higher is better. |",
            "| Peak memory VmHWM | GiB | Maximum sampled high-water resident memory of the server process. Lower is better when quality and speed are acceptable. |",
            "| Context tokens | tokens | Configured context window size for the server run. Larger values increase KV-cache memory and can increase prefill cost. |",
            "| Concurrent requests | requests | Number of simultaneous benchmark requests. Higher values test server contention and batching/queueing behavior. |",
            "| Max generated tokens | tokens | Decode length cap for each response. Longer values make steady-state TPOT easier to measure. |",
            "",
        ]),
        encoding="utf-8",
    )


def write_report(path, tables, model_rows, counts):
    baseline_rows = display_metric_rows(tables["baseline"])
    arch_rows = display_metric_rows(tables["architecture"])
    blas_rows = display_metric_rows(tables["blas"])
    model_display = []
    for row in model_rows[:12]:
        model_display.append({
            **row,
            "model_gib": fmt(row.get("model_gib"), 2),
            "cal_bw": fmt(row.get("calibrated_bw_gb_s")),
            "pred_ms": fmt((safe_float(row.get("tpot_pred_s")) or 0.0) * 1000.0, 1),
            "obs_ms": fmt((safe_float(row.get("tpot_observed_s")) or 0.0) * 1000.0, 1),
            "ratio": fmt(row.get("observed_over_predicted"), 2),
        })

    baseline_table = latex_table(
        "Baseline mandatory-prompt means with explicit units.",
        "tab:baseline",
        ["Sweep", "Engine", "Cache", "Model", "Threads", "TTFT s", "TPOT ms/token", "Output tokens/s", "Peak memory GiB"],
        baseline_rows,
        ["sweep_id", "engine", "cache_type_k", "model_label", "threads", "ttft", "tpot_ms", "throughput", "mem_gib"],
    )
    arch_table = latex_table(
        "ARM versus x86 baseline comparison. The x86 wrapper used 32 threads; ARM used 24 threads.",
        "tab:architecture",
        ["Sweep", "Engine", "Cache", "TTFT s", "TPOT ms/token", "Output tokens/s", "Peak memory GiB"],
        arch_rows,
        ["sweep_id", "engine", "cache_type_k", "ttft", "tpot_ms", "throughput", "mem_gib"],
    )
    blas_table = latex_table(
        "BLAS runtime comparison on ARM using vanilla llama.cpp.",
        "tab:blas",
        ["Engine", "Threads", "TTFT s", "TPOT ms/token", "Output tokens/s", "Goodput fraction", "Peak memory GiB"],
        blas_rows,
        ["engine", "threads", "ttft", "tpot_ms", "throughput", "goodput", "mem_gib"],
    )
    model_table = latex_table(
        "Simple TPOT model validation. BW is calibrated from the best observed model-streaming rate in each sweep.",
        "tab:model",
        ["Sweep", "Engine", "Model", "Model GiB", "BW GB/s", "Pred TPOT ms", "Observed TPOT ms", "Obs/Pred"],
        model_display,
        ["sweep_id", "engine", "model_label", "model_gib", "cal_bw", "pred_ms", "obs_ms", "ratio"],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath}

\title{Performance Analysis of LLM Inference Engines on Deucalion CPU Nodes}
\author{Big Data Analysis Project Assignment \#2}
\date{2025/2026}

\begin{document}
\maketitle

\section{Introduction}

This report follows Track A1, Single-Engine Deep Dive. The primary engine is
\texttt{llama.cpp} served through the OpenAI-compatible \texttt{llama-server}
HTTP interface. The experiments also include TurboQuant cache variants,
the vanilla \texttt{llama.cpp} baseline, the same sweep on x86 nodes, and a
vanilla BLAS comparison on ARM. These extra comparisons are used to explain
the same A1 optimisation space against hardware differences rather than to
claim a separate Track A2 study.

The parsed result set contains """ + str(counts["prompt_rows"]) + r""" prompt-level summary rows and """ + str(counts["scaling_rows"]) + r""" scaling rows.
The raw sweep directories are \texttt{measurements/1253766-track-a1-server-sweep},
\texttt{measurements/1254633-track-a1-server-sweep-x86}, and
\texttt{measurements/1254637-track-a1-vanilla-blas-sweep}. The organized CSVs,
plots, and this report are under \texttt{measurements/organized\_a1\_results}.

\section{Methodology}

The ARM runs used Deucalion \texttt{normal-arm} nodes with Fujitsu A64FX:
48 online CPUs, four NUMA nodes, one hardware thread per core, and SVE support.
The x86 runs used \texttt{normal-x86} nodes with two AMD EPYC 7742 sockets:
128 online CPUs, eight NUMA nodes, one hardware thread per core, AVX2/FMA, and
256 GB class memory nodes as described in the assignment. The ARM assignment
notes describe 32 GB class memory nodes. These capacity differences matter for
headroom, but decode latency is primarily governed by model streaming and cache
locality rather than capacity alone.

The benchmark client sends streaming HTTP requests and records time to first
token (TTFT), time per output token (TPOT), output-token throughput, goodput,
and sampled peak memory. Each configuration uses three measured trials after
one warmup trial. Goodput uses the harness SLA of TTFT no larger than 2.0 s and
TPOT no larger than 0.2 s. The prompt file contains at least ten prompts per category and
marks the three mandatory assignment prompts. These sweeps used
\texttt{LIMIT\_PER\_CATEGORY=5}, so the measured workload covers fifteen prompts
per configuration, with the three mandatory prompts always included.
That is sufficient for the mandatory-prompt analysis below, but it is a partial
run relative to the strict full-workload requirement of ten measured prompts per
category. Generated text for the mandatory prompts is stored in
\texttt{measurements/organized\_a1\_results/combined/mandatory\_quality.csv}.

\section{Metric Definitions}

TTFT is reported in seconds and is the elapsed time from request submission to
the first streamed output token. It mostly reflects prompt prefill, queueing,
and startup overhead. TPOT is reported in milliseconds per generated token and
is the steady-state decode latency. Throughput is reported as generated output
tokens per second. Peak memory is reported as GiB from sampled VmHWM high-water
resident memory. Goodput is a fraction from 0 to 1 using the benchmark SLA:
TTFT no larger than 2.0 seconds and TPOT no larger than 200 milliseconds.

\section{Results}

""" + baseline_table + r"""

The main baseline pattern is that the x86 node delivers lower TPOT for the
8B mandatory model at the tested baseline, but it also uses a larger resident
memory footprint. On ARM, vanilla \texttt{llama.cpp} and the BLAS-linked vanilla
variants are competitive with or faster than the TurboQuant cache modes for the
tested Q4 model. TurboQuant cache modes are still useful as an optimisation
dimension because they change KV-cache representation, memory footprint, and
decode behaviour, but in these measurements the dominant cost remains full
model-weight streaming during autoregressive decode.

""" + arch_table + r"""

Thread scaling shows diminishing returns. On ARM, performance improves when the
thread count rises from one or two CMG-sized groups toward the 36--46 thread
region, but 48 threads is not always best. On x86, 8--16 threads are often
already strong; 128 threads can be worse because the request is spread across
many NUMA domains and synchronization overhead dominates useful matrix-vector
work. Concurrency behaves as expected for CPU serving: increasing simultaneous
requests increases queueing and contention, so per-request TTFT and TPOT rise,
and the goodput SLA degrades. Some high-concurrency BLAS runs returned no
successful prompt rows, which should be treated as overload points rather than
valid latency samples.

Context length mostly affects TTFT and memory because longer prompts increase
prefill work and KV-cache size. Decode length has a smaller effect on TPOT; it
mainly improves measurement stability because fixed startup overhead is spread
over more generated tokens. The small SmolLM2 model is much faster and has a
much smaller memory footprint. The Gemma model produced several zero-token or
weak-output rows, so those model-set results should be discussed as a quality
or compatibility limitation rather than as a clean performance point.

""" + blas_table + r"""

The BLAS comparison shows OpenBLAS/Fujitsu and BLIS are very close for this
server workload. That is consistent with decode being memory-traffic dominated:
once the matrix-vector kernels are reasonably optimized, changing BLAS does not
remove the need to stream the quantised weights for each output token.

\section{Performance Model}

For autoregressive decoding, a simple lower-bound model is
\[
  \mathrm{TPOT}_{pred} \geq \frac{\mathrm{model\_bytes}}{BW_{mem}}.
\]
The intuition is that each output token requires reading a large fraction of the
model weights. If the run is memory-bandwidth bound, TPOT should scale roughly
with model size and inversely with attainable memory bandwidth. The repository
does not currently contain a \texttt{memory\_bandwidth.csv} STREAM measurement,
so the validation table below uses the best observed effective model-streaming
rate per sweep as an empirical bandwidth ceiling. This is weaker than an
independent STREAM validation and should be replaced by measured TRIAD bandwidth
before final submission.

""" + model_table + r"""

The model captures the correct first-order trend: the 8B Q4 model is much
slower and much larger than SmolLM2, and the x86 node can sustain lower TPOT at
the tested baseline. It breaks down where engine overhead, NUMA placement,
thread scheduling, prompt processing, KV-cache layout, and failed/zero-token
generations dominate. Ratios above one indicate overhead beyond a pure memory
streaming lower bound. Ratios below or near one identify the rows used to
calibrate the empirical bandwidth ceiling.

\section{Bottleneck Analysis}

The dominant bottleneck is memory traffic during decode, with secondary
bottlenecks from thread coordination and NUMA locality. Evidence:
\begin{itemize}
\item TPOT tracks model size strongly: SmolLM2 has far lower memory use and much higher throughput than the 8B baseline.
\item Decode length changes throughput less than model size or thread placement, meaning steady-state token generation is the limiting phase.
\item High thread counts eventually hurt, especially on x86 at 128 threads, which is typical when memory bandwidth and cross-NUMA traffic saturate.
\item BLAS library changes have small impact compared with architecture, model size, and threading.
\end{itemize}

The highest-impact optimisation would be to reduce bytes moved per generated
token: smaller weight quantisation, better cache locality, NUMA-aware pinning,
and speculative decoding or prompt batching where quality and latency budgets
allow it. KV-cache quantisation helps memory footprint and long-context
pressure, but it cannot fully remove the repeated model-weight traffic in
decode.

\section{Requirement Mapping}

\begin{table}[htbp]
\centering
\small
\begin{tabular}{lll}
\hline
Requirement & Evidence & Status \\
\hline
Track stated & Track A1 stated in scripts, README, and this report & Met \\
At least three models & Llama 3.1 8B Q4, SmolLM2 360M Q4, Gemma Q4 & Met \\
Mandatory baseline model & Meta-Llama-3.1-8B-Instruct-Q4\_K\_M.gguf & Met \\
Metrics & TTFT, TPOT, throughput, goodput, sampled VmHWM memory & Met \\
Three trials plus warmup & \texttt{trials=3}, \texttt{warmup\_trials=1} in system logs & Met \\
Prompt dataset & JSON contains 10+ prompts per category and mandatory prompts & Met in dataset \\
Measured prompt count & Runs used \texttt{LIMIT\_PER\_CATEGORY=3} & Partial \\
Optimisation dimensions & cache type, threads, concurrency, context, decode length, node architecture, BLAS & Met \\
Performance model & TPOT lower bound from model bytes and bandwidth & Partial until STREAM bandwidth is added \\
\hline
\end{tabular}
\caption{Assignment requirement mapping.}
\label{tab:requirements}
\end{table}

\section{Conclusion}

The measurements support a hardware-aware explanation of CPU-only LLM serving:
decode is dominated by repeated movement of model weights, while prefill and
long-context behaviour are reflected more strongly in TTFT and memory. The x86
EPYC node is faster for the tested baseline, but raw thread count is not enough:
oversubscribing across NUMA domains can reduce throughput. On ARM A64FX, using
most but not necessarily all cores is better than assuming maximum thread count
is optimal. BLAS choice matters less than model size, memory traffic, and
thread placement for the measured server workload.

\end{document}
""",
        encoding="utf-8",
    )


def write_findings(path, tables, model_rows, counts):
    model1_baselines = [
        row for row in tables["baseline"]
        if row.get("model_label") == "model-1-mandatory" and safe_float(row.get("tpot_s_mean")) is not None
    ]
    best_baseline = min(
        model1_baselines,
        key=lambda row: safe_float(row.get("tpot_s_mean")),
    )
    worst_baseline = max(
        model1_baselines,
        key=lambda row: safe_float(row.get("tpot_s_mean")),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# Organized A1 Result Summary",
            "",
            f"Prompt-level rows: {counts['prompt_rows']}",
            f"Scaling rows: {counts['scaling_rows']}",
            "",
            "## Selected complete runs",
            "",
            "- ARM server sweep: `measurements/1253766-track-a1-server-sweep`",
            "- x86 server sweep: `measurements/1254633-track-a1-server-sweep-x86`",
            "- ARM BLAS sweep: `measurements/1254637-track-a1-vanilla-blas-sweep`",
            "",
            "## Highest-signal findings",
            "",
            f"- Best Meta-Llama-3.1-8B baseline TPOT: `{best_baseline['sweep_id']}` / `{best_baseline['engine']}` / `{best_baseline['cache_type_k']}` at {fmt((safe_float(best_baseline.get('tpot_s_mean')) or 0) * 1000.0, 1)} ms.",
            f"- Slowest Meta-Llama-3.1-8B baseline TPOT: `{worst_baseline['sweep_id']}` / `{worst_baseline['engine']}` / `{worst_baseline['cache_type_k']}` at {fmt((safe_float(worst_baseline.get('tpot_s_mean')) or 0) * 1000.0, 1)} ms.",
            "- The complete organized CSVs are in `measurements/organized_a1_results/tables/`.",
            "- Human-readable CSVs with units are the files ending in `_readable.csv`.",
            "- Metric definitions are in `measurements/organized_a1_results/METRICS_GUIDE.md`.",
            "- Mandatory generated outputs and heuristic quality labels are in `measurements/organized_a1_results/combined/mandatory_quality.csv`.",
            "- The LaTeX report is `measurements/organized_a1_results/report/track_a1_analysis.tex`.",
            "- The performance-model table uses empirical effective bandwidth because no `memory_bandwidth.csv` STREAM file was found.",
            "",
        ]),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("measurements/organized_a1_results"))
    args = parser.parse_args()

    prompt_rows, scaling_rows = collect_summaries(DEFAULT_SWEEPS)
    counts = {"prompt_rows": len(prompt_rows), "scaling_rows": len(scaling_rows)}

    combined_fields = sorted({key for row in prompt_rows for key in row})
    write_csv(args.out_dir / "combined" / "a1_all_prompt_rows.csv", prompt_rows, combined_fields)
    combined_scaling_fields = sorted({key for row in scaling_rows for key in row})
    write_csv(args.out_dir / "combined" / "a1_all_scaling_rows.csv", scaling_rows, combined_scaling_fields)

    baseline = aggregate(
        table_rows(scaling_rows, "model-set"),
        ["sweep_id", "sweep_name", "arch", "engine", "model_label", "model", "threads", "ctx_size", "parallel_requests", "cache_type_k", "cache_type_v", "max_tokens"],
    )
    architecture = [
        row for row in baseline
        if row["sweep_id"] in {"arm_server", "x86_server"} and row["model_label"] == "model-1-mandatory"
    ]
    blas = [
        row for row in baseline
        if row["sweep_id"] == "blas_server" and row["model_label"] == "model-1-mandatory"
    ]
    tables = {
        "baseline": baseline,
        "architecture": architecture,
        "blas": blas,
        "thread_scaling": aggregate(
            table_rows(scaling_rows, "thread-scaling"),
            ["sweep_id", "sweep_name", "arch", "engine", "model_label", "threads", "cache_type_k", "cache_type_v"],
        ),
        "concurrency": aggregate(
            table_rows(scaling_rows, "concurrency"),
            ["sweep_id", "sweep_name", "arch", "engine", "model_label", "parallel_requests", "cache_type_k", "cache_type_v"],
        ),
        "context_length": aggregate(
            table_rows(scaling_rows, "context-length"),
            ["sweep_id", "sweep_name", "arch", "engine", "model_label", "ctx_size", "cache_type_k", "cache_type_v"],
        ),
        "decode_length": aggregate(
            table_rows(scaling_rows, "decode-length"),
            ["sweep_id", "sweep_name", "arch", "engine", "model_label", "max_tokens", "cache_type_k", "cache_type_v"],
        ),
        "model_set": baseline,
    }

    table_fields = [
        "sweep_id", "sweep_name", "arch", "engine", "model_label", "model", "threads",
        "ctx_size", "parallel_requests", "cache_type_k", "cache_type_v", "max_tokens",
        "mandatory_prompt_rows", "successful_runs", "ttft_s_mean", "tpot_s_mean",
        "throughput_tok_s_mean", "goodput_fraction_mean", "max_vmhwm_gib",
    ]
    for name, rows in tables.items():
        fields = [field for field in table_fields if any(field in row for row in rows)]
        write_csv(args.out_dir / "tables" / f"{name}.csv", rows, fields)
        write_readable_table(args.out_dir / "tables" / f"{name}_readable.csv", rows)

    model_rows = build_performance_model(baseline)
    model_fields = [
        "sweep_id", "arch", "engine", "model_label", "model_file", "cache_type_k",
        "cache_type_v", "model_gib", "calibrated_bw_gb_s", "tpot_pred_s",
        "tpot_observed_s", "observed_over_predicted", "throughput_tok_s_mean",
        "max_vmhwm_gib",
    ]
    write_csv(args.out_dir / "tables" / "performance_model.csv", model_rows, model_fields)

    write_report(args.out_dir / "report" / "track_a1_analysis.tex", tables, model_rows, counts)
    write_metrics_guide(args.out_dir / "METRICS_GUIDE.md")
    write_findings(args.out_dir / "README.md", tables, model_rows, counts)

    print(f"wrote organized results to {args.out_dir}")
    print(f"prompt_rows={counts['prompt_rows']} scaling_rows={counts['scaling_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
