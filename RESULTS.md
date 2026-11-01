# Selected results and evidence scope

The [English submission report](reports/ROSA_single_GPU_submission_EN_20261101.pdf) is the public account of the methods, measured results, unsuccessful trials and limitations. Its main text is five pages, followed by a twelve-page appendix. The [reproduction materials](docs/REPRODUCTION_ASSETS.md) bind historical cases to source versions and inputs; they do not contain the full raw experiment logs.

| Experiment | Recorded observation | Limitation |
| --- | --- | --- |
| P1 shared inference | Four independent environments; 223 qualified action chunks in 30 s; Q = 7.4333/s; S1 p99 = 120.16 ms | A measured operating point, not maximum capacity |
| Isolated S1 | 3,000/3,000 normal responses at 8 Hz; replicate p99 = 94.72 / 97.47 / 96.97 ms | Not a complete accepted profile curve |
| v0.20.8 colocated timing | All observed component p99 targets met in three 300 s runs | Individual overruns remained; P3/P4 stopped on Safety decisions |
| Qwen3 Safety comparison | 4B and 8B each had 100% P3/P4 original-S1 stop rates in the 120 s measurement windows | Replacement alone did not resolve the measured stopping behavior |
| v0.20.15 recovery confirmation | P3 stops = 7.34%, 5.21%, 5.61%; P4 stops = 0%, 0%, 0.40% | Only confirmation 3 passed all fixed criteria |
| v0.20.16 final GPU confirmation | Run 1: P3 stops = 4.76%, P4 stops = 100%; run 2: P3 Monitor = 148/150 scheduled slots (98.67%) | No third run; recovery and periodic fulfillment remained unresolved |
| v0.20.17 termination fix | 45 related CPU tests passed in the recorded validation | No GPU validation of this fix |

These results come from different versions and conditions; no single final configuration achieved every objective. Qualified throughput counts applied chunks that meet the required timing and dependency checks. Original-S1 stop rate is rejected original proposals divided by original action decisions, not a task-failure rate. Independent retreat actions do not count as successful original S1 actions. The exploratory 90% stop-rate threshold is an engineering goal, not an established safety benchmark.

The [combined demonstration](media/ROSA_P1_P4_combined_demo_EN_20261101.mp4) begins with the separate P1 `p1-b1` connection test and then appends the fixed P2–P4 success/failure video. The P1 video source is not the same trial as the 223-action headline result above. It shows four simulated environments sharing one model. The later P2–P4 footage shows three environments in separate trials. No seven-environment simultaneous run or physical-robot deployment is claimed.

The ROSA paper's multi-GPU speedups, maximum robot counts, physical failure recovery and monetary savings were not reproduced. Single-GPU colocation, geometric checks and retreat are project extensions. CPU scheduling feasibility does not establish GPU deadline performance. Safety/Monitor semantic quality and stable end-to-end recovery remain unverified or incomplete.

For the public copy's current CPU and packaging checks, see [PUBLICATION_VALIDATION.json](PUBLICATION_VALIDATION.json). Those checks are not additional robot experiments. Historical notes in `docs/` may reference non-bundled evidence by its original relative path; see [publication scope](PUBLICATION.md).
