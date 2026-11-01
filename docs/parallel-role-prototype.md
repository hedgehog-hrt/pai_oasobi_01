# Short VLM outputs and parallel roles

v0.20.6 adds opt-in variants for the single-GPU prototype. Existing role formats,
FIFO defaults, deadlines, periodic frequencies and R07 review gates stay in place.

`monitor_label_v1` emits only the status enum. The transport text explicitly says
no explanation was generated; it is not a model-written reason. `s2_short_v1`
limits the next-subgoal string to 64 characters. Both require constrained vLLM
decoding, reject incomplete or extra-key outputs and change the configuration
fingerprint. They do not certify semantic equivalence or task success.

`parallel_roles_v1` removes the group-wide invocation lock for a declared
prototype, replacing it with one FIFO lock per role. Each role still runs at most
one batch-one call at a time. Up to four distinct roles can be outstanding on the
same physical GPU. This permits CUDA scheduling overlap across the existing
engines; it does not partition the GPU, promise simultaneous kernel execution,
enable MPS, or guarantee latency isolation. Per-role memory reservations and the
S1 headroom check remain required. A caller cancellation does not prematurely
release a model still running: the existing shield/drain and session-generation
rules apply. Queue policy, dispatch/completion times and active invocation count
are saved with each response. Normal dedicated/R07 plans remain serialized.

Evaluation must distinguish request deadlines from semantic behavior. In
particular, an unknown Safety label continues to stop actions even if delivered
on time. Successful model calls cannot be counted as qualified robot actions.
Compare fixed inputs for the output variants, then identical shortened model
settings and populations for FIFO versus parallel roles. Original evidence is
retained; interrupted/warmup-shortfall trials are not performance samples.
