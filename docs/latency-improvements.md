# Latency improvements (2026-10-31)

These are explicitly versioned prototype extensions. Original raw trials,
deadline settings, Safety freshness checks, and normal R07 adoption gates remain
unchanged. Model-label correctness must be reviewed separately from latency.

## Probe evidence persistence

The fixed observation payloads are hashed before warmup. Each request receives
its own copy. The driver records dispatch and response times on the event loop,
then submits evidence to a bounded, single-writer thread (64 waiting jobs).
Action compression, output hashing and JSONL persistence occur off the dispatch
loop. The queue drains before evidence sealing. Queue overflow or a write failure
makes the trial fail; missing evidence is never silently accepted. A dispatch
that is already late after payload preparation is retained as an unsent arrival.
Per-request preparation and persistence timings are saved. This addresses a
possible source of the earlier 81 ms lag; its historical cause is not proven.

## VLM timing and Safety output variant

`latency_breakdown` distinguishes host request preparation, adapter admission
wait, time until the first nonempty engine output, subsequent output time, and
total engine stream duration. The first-output measurement includes engine
queueing, image processing and prefill. It is not an isolated CUDA measurement of
those stages. Prompt and generated token counts accompany the timings.

`safety_label_v1` constrains generation to a single `decision` key with
`safe`, `unsafe`, or `unknown`. Unknown still stops the robot. The existing
transport's required reason field explicitly says that no explanation was
generated. Truncated, extra-key and invalid outputs are rejected. The runtime
fingerprint records the schema, prompt and output budget. No semantic-equivalence
or paper-configuration claim follows from this variant.

## Single-GPU group scheduling

Prototype preparation accepts `queue_policy='edf_queue_expiry_v1'` or the
unchanged default `fifo`. It applies only to the single-GPU multi-model extension;
P1 and ordinary dedicated/R07 plans retain their original execution policy.

The new policy orders pending requests by worker admission time plus the role's
configured deadline budget, with FIFO ties. The clock is the physical actor's
monotonic clock. This is a worker-side budget, not the driver's full end-to-end
deadline; the original evaluator still checks that deadline including transport.
The active call is never preempted and at most one model call runs at a time.
Expired queued calls return an explicit `QueueDeadlineExceeded` error with no
model output, no batch ID, and no inference call. The session is released and the
request remains in every relevant denominator. Queue wait, policy, expiry and
model execution timing are saved in closed-loop request details. Safety cannot
reuse an expired queued result as a safe confirmation.

This policy can reduce stale work but cannot make an overloaded GPU feasible,
interrupt long running calls, or guarantee service to every role. Warmup sample
shortfalls remain an early-stop condition. Scheduling comparisons must keep model
configurations, task banks, durations and client counts fixed.
