# Simulation-only hybrid Safety

The `libero_t01_shadow_v1` extension is specific to the pinned LIBERO Object
alphabet-soup-can task. It supplements the single-GPU P3/P4 prototype; it is not
the paper's VLM-only Safety method or a physical-robot safety certification.
The VLM-only results remain a separate baseline.

The real Safety VLM returns `visibility`, `person`, and `collision`. Missing or
invalid output, unknown visibility/person, a visible person, and an explicit
collision warning stop the action. A collision label of `unknown` is resolved
only by the independent MuJoCo guard; it does not by itself authorize motion.
The `safety_visual_guard_v1` format requires a geometry guard in the execution
plan. The prompt, schema, parser, precision, and checkpoint are fingerprinted.

The guard previews the actual S1 action in a separate simulator initialized
from the same seed/state and controller reset. Every 2 ms physics substep checks
contacts and geometry distances against a 2 mm clearance. Position/contact
queries run on an independent native `MjData` copy so they cannot alter the
predicting simulator's dynamics caches. The allowance is a
project convention. It permits the gripper contacting the target, target with
basket or floor, fixed base mounting, same rigid body, adjacent robot links,
gripper internals, and body pairs excluded by the pinned model. It checks robot
against world, nonadjacent robot self-contact, and target against distractors.
It does not measure excessive grasp force, semantic task intent, or real-world
hazards absent from the simulated geometry. The environment uses a floor work
surface, not a tabletop.

A clear preview produces a one-use lease bound to the action, pre-state,
configuration and predicted post-state. The live executor rejects absent,
changed or reused leases. State and step counts must match after application.
A rejected or unused preview is reset and accepted history is replayed before
another preview; no inverse dynamics rollback is assumed. The twin retains the same full model/controller reset and disables unused image observables during prediction. Live cameras
and the real VLM remain active.

The driver rechecks the latest timely VLM result immediately before motion,
including when a new VLM result arrived during preview. The physics preview has a 500 ms budget; twin recovery has a separate
2,000 ms budget (2,500 ms maximum combined). Recovery and physics durations are
reported separately and included in wall time. These local simulator budgets
are additional to the periodic VLM's unchanged 500 ms deadline. Late previews and failures stop motion. If the trial window
ends during preview, the unexecuted action is recorded as censored.

`lifecycle.jsonl` retains previews, visual authorization links, causes and
execution receipts. `reaudit_closed` requires both authorizations for every
applied guarded action and rejects altered action/state hashes, expired visual
results, missing proof and reused leases. `scripts/safety_stop_report.py` reports
stop rates over audited measured action decisions, alongside cause counts,
actual simulator steps and per-component latency. These rates are not task
success rates or validated semantic safety accuracy.
