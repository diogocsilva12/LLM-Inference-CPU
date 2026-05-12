# Metrics Guide

Use the files ending in `_readable.csv` for report-facing tables. They keep units in the column names.

| Metric | Unit | Meaning |
| --- | --- | --- |
| TTFT mean | seconds | Time elapsed from HTTP request submission until the first streamed output token. This mostly reflects prompt prefill, scheduling, and startup overhead. Lower is better. |
| TPOT mean | milliseconds per output token | Average elapsed time between generated tokens during decode. This is the main latency metric for steady-state generation. Lower is better. |
| Throughput mean | output tokens per second | Generated tokens completed per second for the measured request stream. Higher is better. |
| Goodput fraction | fraction from 0 to 1 | Share of requests meeting the SLA used by the harness: TTFT <= 2.0 s and TPOT <= 200 ms. Higher is better. |
| Peak memory VmHWM | GiB | Maximum sampled high-water resident memory of the server process. Lower is better when quality and speed are acceptable. |
| Context tokens | tokens | Configured context window size for the server run. Larger values increase KV-cache memory and can increase prefill cost. |
| Concurrent requests | requests | Number of simultaneous benchmark requests. Higher values test server contention and batching/queueing behavior. |
| Max generated tokens | tokens | Decode length cap for each response. Longer values make steady-state TPOT easier to measure. |
