# Issue #23 Stage0 — shared Sim-side VAE and Local-TTT cost/benefit

Status: **PROPOSED / DESIGN REVIEW REQUIRED**. Owner date: 2026-11-07 Asia/Shanghai.
This package is a non-executable specification. Implementation is NOT_STARTED;
CPU Worker, real Wan, GPU, RoboCasa, latency, throughput and SR gates are NOT_RUN.
No production flags, runnable launchers or measured results are introduced here.

## First acceptance table: P0 single Server × single Sim

All mandatory arms use exactly one Policy Server, one Sim, actual Policy B1,
Action Horizon16, executed raw15, state15 and 20Hz **data semantics**. This is not
a promise of 20Hz wall-clock execution. Match tasks, initial states/seeds, maximum
steps, sampler steps, precision and GPU topology. Run arms sequentially with
exclusive resources; balance/randomize arm order across matched episodes.

| Arm | Checkpoint / memory | Historical visual processing | Fast Weight | Purpose |
| --- | --- | --- | --- | --- |
| A | Frozen native No-TTT N / off | None; no history Worker | None; no Prefix | Native full-system baseline |
| B | Frozen Local-TTT L / required | Official server_vae: 16 independent Encode1 calls | Original T16 scan | Original Local-TTT cost baseline |
| C | **Same L and inference configuration as B** / required | Sim Worker Bvae1, wait0; Policy-owned boundary feature + 15 Encode1 | 16 causal T1 updates | Same-weight engineering comparison |
| D (optional) | Same L / explicit mechanism ablation | None | No update or Prefix | Separate mechanism attribution, only if legally implemented and reviewed |

A vs B/C compares complete systems with **different model weights**; SR differences
cannot be attributed solely to online adaptation. B vs C isolates the engineering
change. D is currently **UNSUPPORTED / NOT_RUN**, not native No-TTT: existing
mode validators reject a Local-TTT checkpoint with memory off. Do not delete those
checks, reset consumer steps every request, or merely suppress Prefix while still
updating memory. An explicit ablation mode would require its own review and proof
that history encoding, scan/update and Prefix injection are all absent.

P0 outputs: first-action and subsequent Replan P50/P95/P99/mean/CI; each timing
component; background compute vs foreground blocking; steps/episode and walltime;
peak memory/utilization; paired per-task SR, overall SR and delta in percentage
points; extra milliseconds per executed action. Quantile differences and quantiles
of paired differences must be labeled separately. No results are populated now.

Same-GPU and split Sim/Policy-GPU deployments are separate strata and reports.
In same-GPU C, the history Worker adds a VAE copy **in addition to the Policy
current-frame VAE**; one Worker per Sim GPU does not mean one total VAE model per
physical GPU. Include that memory and contention cost. Multi throughput cannot
substitute for this P0 experiment. An 18×2 smoke cannot establish SR improvement.

## Scope, authority and immutable baseline

Owner authority: [Root Issue23](https://github.com/wxwy/psm_wma/issues/23), including
its P0 addendum. Fresh official baseline observed at task start:

- Root V3: `a77f3d4bd8a691435d1935025cc1a9766da646a1`.
- Child / official Gitlink: `1ca308055fc41fb69efaeb9efaa23808f164a18e`.
- New independent branch in both repositories:
  `v3-sim-shared-vae-issue23-stage0-20261107`.

PR4 and Child5fe2b8de are not bases, dependencies or cherry-pick sources. Issue23
reports PR4's matched H100 36EP result as 82.53 vs 170.37 EP/h and Queue P50
23.02s vs 1.40s; these are **reported external evidence**, not measurements made
by this Stage0 work. Its admission-age batching defect must not be inherited.
This is an independent replacement design, not a PR4 performance closure.

Only docs are changed. Root Gitlink stays at the official baseline; the Child
design commit is a **docs review target**, not a runnable Root/Child Formal Pair.
Existing runs, training, model parameters, checkpoints, caches, compilation
boundaries, M/B Resume checks and frozen Manifest semantics remain unchanged.
Later integration requires a new reviewed implementation Pair and Fresh identity;
this design never permits source-switch Resume of an existing run.

## Actual source wiring and reuse points

Paths below are relative to this Child unless prefixed Root. Line numbers refer
to Child1ca30805 / Roota77f3d4b, not proposed implementations.

| Existing authority | Source / implication |
| --- | --- |
| Actual Sim path | `cosmos_framework/simulation/robocasa/eval_v2_worker.py:18,138` → `eval_v2_slot.py:63` → native episode/closed-loop path; reuse this path rather than create a simulation scheduler |
| Evidence causality | `cosmos_framework/simulation/robocasa/closed_loop_eval.py:386–426`: capture pre-action RGB, canonicalize executed action, record only after successful env.step |
| Local evidence / original scan | `cosmos_framework/inference/robocasa_local_memory_policy.py:157–204`, `cosmos_framework/inference/local_memory_online.py:189–204`; preserve model-owned FastState and causal consumer frontier |
| Mode validation | `cosmos_framework/inference/robocasa_transaction_mode.py:14–34`; `cosmos_framework/scripts/action_policy_server_robocasa.py:818–828`; mode off does not legalize a Local checkpoint |
| Preprocessing | `cosmos_framework/inference/robocasa_composite_visual.py:27–46`; `cosmos_framework/model/generator/vision_encoder.py:65–75`: existing resize/reflection pad and fp32 /127.5−1 |
| Wan T1 behavior | `cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py:835–840,1047–1056`: deterministic encoding, fresh cache per independent T1 call |
| Visual96 | `cosmos_framework/model/generator/mot/robocasa_exact_window_local.py:26–32`: pool full pre-crop latent; do not replace it with Policy's subsequently cropped latent |
| AOT limit | Wan tokenizer `:895–913,1403,1781–1802`: AOT guarded by B==1, key/export lack general dynamic B; B4/B8 use eager |
| Lifecycle | Root `scripts/robocasa_eval_v2/runtime.py:169,179,864–878,911`, `scripts/robocasa_eval_v2_processes.py:208`: Registry-owned process groups; per-episode media quiesce reaps Sim descendants |
| Placement/resources | Root `scripts/robocasa_eval_v2/runtime.py:295,548`, `resources.py:15,83,131`, `startup.py:21`, `config.py:204,303`; explicit GPU mapping, separate CUDA/EGL visibility and reservations |

Exact filenames and lines must be re-resolved before Stage1/3 source edits. The
standalone Worker reuses preprocessing/Encode1/Visual96 math. The current helper
has transitive training-dataset imports; Stage1 must isolate a pure inference
helper or lazy imports, without initializing training data or a Policy model.
Real VAE loading remains optional/lazy in CPU/synthetic tests.

## Two independent profiles

[single_latency recipe](recipes/single_latency.json) freezes the P0 arms above.
No multi scheduler, 16-Slot minimum or second Server is required. C Worker B1
dispatches immediately after validation; no intentional aggregation wait. Policy
B1 likewise has no peer wait. The current Policy wait setting has its own legacy
validation range; do not pretend the proposed Worker wait0 is an existing CLI
flag. Future profile plumbing must demonstrate zero actual aggregation wait.

[multi_sr_throughput recipe](recipes/multi_sr_throughput.json) proposes Sim/render
GPU4/5, 16 Sims and one shared Worker on each, and one Policy Server on GPU6/7:
32 Sim Slots, independent Policy Bmax8. Physical GPU UUIDs must be resolved and
frozen before execution, not inferred from CUDA ordinal names. Sweep seven legal
Worker configurations: (1,0), (4,0/2/5), (8,0/2/5), in Bmax/wait_ms units. Policy
and Worker B/wait are independent. Group only compatible HW/preprocess/weights/
dtype inputs; different Sessions and source timesteps may share a batch. There
is no 16-Sim, same-step, episode or Replan barrier for Worker dispatch.

The multi 18 tasks×2 smoke targets 36 valid, zero invalid/errors/unresolved and
proper owned-process cleanup. Compare against a newly matched baseline; the
Issue-reported 170.37 EP/h is a target reference only when hardware, input work,
steps/episode and configuration match. Report actual Bvae/Bpolicy histograms,
queue/prepare/generation and steps/sec, not only EP/h. Larger SR runs need a
separate Owner Gate. Neither profile is runnable at Stage0.

## Worker ownership, buffering and scheduling

The Worker is a **run-level Registry peer of Sims**, one per actual Sim/render
GPU. Do not spawn it as a Sim descendant: per-episode quiesce would kill peers'
encoder. Only Worker receives CUDA visibility for its assigned GPU; Sim retains
the existing EGL/CUDA isolation. OSMesa/no render GPU requires an explicit Worker
GPU mapping, never an inferred device. Root startup/resource accounting must
eventually add an explicit Worker role, include RSS/VRAM and live reservations,
and avoid charging 16 VAE copies. Worker exits/restarts are observed through the
owned Registry; unrelated processes are never targeted.

Start with local Unix-socket **immutable copied byte payloads**, not mutable NumPy
views or shared-memory aliases. After successful env.step, copy pre-action RGB
and executed raw15 into owned buffers before asynchronous submission. Worker
rechecks shape/range/digest and retains bytes until the relevant lifetime ends.
The Sim continues stepping unless bounded capacity or a causal frontier makes
backpressure necessary; indefinite dropping or unbounded buffering is forbidden.

Proposed capacity defaults to review: 16 unacknowledged source items per Session,
256 per 16-Sim Worker; composite uint8[256,512,3] is 393216 bytes, so a 128MiB
raw-input cap (8MiB in single) covers those items with explicit accounting.
Also cap active batches (initially one), metadata, output retention and total
RSS/VRAM. Dequeue is not release: queued, copied/inflight, GPU input and retained
result obligations all count until completion/ACK/cancel drain. Limits are future
configuration frozen in identity; hardware reservations remain unmeasured.

Worker READY time is set **once after validation**. B1 or a full compatible batch
dispatches immediately. Otherwise dispatch when oldest READY+window expires or
an earlier immutable deadline requires it. Select READY subsets fairly across
Sessions; non-ready entries cannot cause head-of-line blocking. Never wait for
all peers, a matching source_step, or a future frame. Bound queued jobs per lease
and use per-Session round-robin tie-breaking. Control End/cancel remains responsive
even when data capacity is full. Timeout emits a first error; it never loses a
source item silently or manufactures a successful Episode.

Policy batching needs a separate Stage3 correction: eligibility begins only
after complete verified history and private candidate preparation, not request
admission. Select READY requests; timestamp once, cap/window/deadline decide
dispatch. Preserve existing Native Batch. A single model executor may process
one bounded T1 update then re-evaluate due Policy batches, respecting existing
strict-media fairness; no unbounded update drain or lock held across Worker wait.

## Trusted feature protocol and causality

[Feature envelope schema](feature_envelope.schema.json) defines metadata, with a
fixed binary body: executed action is 15 little-endian finite float32 values
(60 bytes), Visual96 is 96 little-endian finite float32 values (384 bytes).
Reject extra fields, wrong lengths, NaN/Inf, changed retries, stale epochs,
unissued action chunks and steps outside the Session's legal completed frontier.
Sim forwards Worker receipts to one authoritative Policy evidence API. C sends
compact signed features/actions and references, not a second full-history PNG
parse on every Replan. The legacy server_vae payload/API remains available.

Owner generates a separate random 32-byte HMAC-SHA256 key per run/policy/Worker
epoch. Give it to Worker through a private inherited FD and install the verifier
through a narrow Owner-authenticated enroll/revoke endpoint. Do not reuse the
strict Session registration contract, expose global Owner token to Worker, or
give keys to Sim. No keys in argv, environment, actor config, /info, logs or
artifacts; public key_id is only a lookup identifier, not a verification secret.

MAC input is ASCII `PSM23-VIS96-v1\0`, u32be canonical-header byte length,
canonical-header UTF-8 bytes, raw15 bytes, then Visual96 bytes. Header encoding:
JSON sorted keys, compact separators, ensure_ascii=true, no floats, duplicate
keys or nonfinite values; strictly schema-validated before use. MAC is lowercase
hex and verified in constant time. Header binds full Session/Lease identity,
run/Manifest, Policy and Worker epochs, task/attempt/episode/generation, source
step, issued Policy chunk/request key, source RGB digest and shape, action digest,
Visual96 digest, preprocess and **actual loaded VAE weights/config/scale** identity,
original deadline/clock domain. Worker independently computes source digests.
The VAE is loaded separately from the Local DCP; DCP identity alone is insufficient
to authenticate that encoder. Owner attests actual loaded files, not just a
Worker-reported model name. Schema completion is not yet a security implementation.

Canonical source digest is SHA256 of ASCII `PSM23-RGB-v1\0`, three u32be dimensions
(H,W,C), then immutable C-order HWC uint8 **RGB** bytes (256×512×3 here). It is
computed before resize, normalization or padding, never over PNG bytes, CHW float
buffers or BGR. Policy's own boundary cache uses the same domain after decoding
the original current image. Any codec round trip must preserve these bytes.
raw15_sha256 and visual96_sha256 hash their exact fixed-length little-endian
binary bodies. Reject a source-contract mismatch even if shape happens to match.

Lease keys must exactly reuse `SessionLease.identity()` from
`cosmos_framework/inference/robocasa_session_transactions.py:146–160`:
server_epoch, run_lease, manifest_digest, slot_id, generation (1..2^63−1), task,
episode_index, attempt_id, session_id and episode_id. Header policy_epoch must
equal lease.server_epoch; capability stays out of the public signed header and
remains in the existing authenticated transport proof. This is not a renamed
or reduced lease contract.

HMAC verifies bytes produced by the trusted encoder; it does **not** prove an
env.step happened. The trusted Sim shim must bind pre-action RGB to an issued
action chunk, successful real step and monotonically completed frontier. If an
untrusted Sim can read Worker/Owner secrets through same-UID filesystem/proc/FD/
ptrace access, 0600 files and HMAC alone are insufficient. Stage3 Feature Action
Path stays disabled/BLOCKED until OS trust isolation or an explicitly approved
same-trust-domain threat model is attested. No silent fallback that relabels C
while actually running B. Explicit user-selected B is the fallback configuration.

Boundary frame: Policy uses its own previous current-frame **pre-crop** Visual96
cache, addressed by issued chunk and source digest. Sim supplies a reference and
executed action, never an externally trusted replacement boundary feature. The
remaining 15 frames come from Worker. Missing/mismatched boundary fails closed;
first Replan has no history. A terminal short chunk is separately handled and
traced, never padded with future actions or falsely labeled a full16 update.

Exactly-once is enforced at both Worker and Policy by lease/chunk/source_step and
exact digest: byte-identical retries return the retained result, changed bytes
fail; duplicate responses never update twice. Out-of-order encode completions are
retained and consumed only at the next causal frontier. Candidate FastState and
Prefix are Session-private and unpublished until existing Replan Snapshot and
atomic Commit/ACK. Abort/timeout drops candidate; original published state is
unchanged. No Session state or encoder cache crosses samples/Sessions.

Worker restart rotates epoch/key and invalidates old outstanding receipts. Mark
affected Attempts invalid, drain/quarantine and create new Sessions; never reset
the old frontier under the same lease. End stops that Session's submissions,
cancels queued work, drains active GPU work or quarantines uncertain ownership,
discards late results, releases retained bytes, and emits a per-Session Worker End
receipt before Policy End ACK. End must not wait for all other Sims or kill the
shared Worker at each Episode. Reuse is forbidden without proven drain. Run close
terminates all owned process groups and leaves external processes untouched.

A dead Worker cannot emit an End receipt. Fatal recovery therefore accepts only
an Owner-authenticated cleanup proof in its place: Registry proves the old
incarnation's PID/start-time and owned process groups reaped, revokes its epoch,
cancels affected Sessions, and Policy confirms their candidates/executor work
quiescent. This exception can close an old Session only; it cannot authenticate
old features, resurrect a candidate or reopen the old lease. If any proof is
missing, keep quarantine and stop the run; never reuse that Slot optimistically.
The recovery control contract must be independently reviewed at Stage3.

## Immutable deadlines and no hidden budget reset

The existing client creates prediction budget after payload work; migration must
not hide Worker/payload delays outside the new experiment's end-to-end budget.
Propose an explicit experiment wrapper that freezes one Replan absolute deadline
at the last-action boundary **before** payload/flush/wait, applies the same scope
to A/B/C, and forwards it unchanged through IPC/HTTP/retries/TTT/generation/ACK.
Legacy default behavior is not silently altered. Budget is min(original Episode
deadline, boundary+configured transaction timeout); every downstream stage may
tighten, never extend it. Existing transaction IDs, snapshot/ACK and fail-closed
timeout behavior remain authoritative. This wrapper/API needs Stage3 review.

Background frame jobs predate the next Replan: give each a single submission
deadline bounded by original Episode deadline. An eventual Replan can tighten it;
no receive/retry refresh and no claim that the future Replan was already created.
Use a validated shared-host monotonic clock domain or a reviewed translation
handshake; do not compare unrelated hosts' monotonic timestamps. End cleanup has
its own bounded lifecycle/drain budget and cannot create a new inference budget
for an expired transaction. Preserve the first error and report cleanup errors
separately rather than masking it with later EOF/End exceptions.

## Numerical Gate before feature integration

Preserve exact existing RGB canonicalization, resize/reflection padding, fp32
normalization, loaded BF16 Wan, deterministic Encode1 with fresh per-sample cache,
latent scaling and Visual96 pooling of full pre-crop latent. Stack on B into
[B,3,1,H,W], never encode T16 video. Reorder outputs back by bound Session/step.
Compare B1 vs B4/B8 using identical real frames/weights/HW/dtype/backend, including
mixed Session/time ordering and partial batches. Record latent **and** Visual96
max_abs/mean_abs and input/output hashes. Then compare Worker with the actual
Policy AOT/eager backend separately. B4/B8 AOT is not implemented by this task.

Default required difference is **zero** until a numerical tolerance is explicitly
registered and independently approved **before** examining validation outcomes.
Nonzero or undeclared tolerances mean BLOCKED; do not loosen training semantics,
borrow old unrelated tolerances (including Wan AOT atol0.2), or call synthetic
CPU equality proof of real Wan BF16 parity. Following that Gate, prove T16 versus
16×T1 FastState, Prefix and Action parity using same L, including no-noisy-action,
retry/abort, short terminal chunks and Session isolation. Hardware claims remain
NOT_RUN until their separate Gate is executed.

## Raw trace, attribution and cost/benefit statistics

[Trace schema](trace.schema.json) describes raw events with correlation identity,
monotonic ns, clock domain, CPU interval and optional correctly-streamed CUDA
elapsed ns. Record initial model/Worker load separately. CUDA events bracket
actual work; resolve on the correct stream/window completion, not a global sync
at every tiny operation that destroys overlap. Cross-process queue/IPC uses
host-clock spans and causal links, never subtraction of CUDA event clocks.

Required phases: observation/state/preprocess; env.step/EGL; history capture/PNG
where applicable; enqueue/copy; VAE READY/queue/encode; feature IPC/verification;
executor wait; TTT queue/update and loss telemetry; Prefix prepare/inject; current
frame VAE; Policy READY/queue/prepare/native generation; response/decode/ACK/
Commit/End; boundary history drain and full Replan. Log actual B distributions,
queue depth/capacity, source count (old16/new15+boundary1), memory/utilization.

Timing definitions:

- `T_load`: process start → required models/Worker ready, separate from inference.
- `T_first_action`: first valid observation ready → first legal Action Chunk
  executable. Also report cold process → first action; compile/warmup tagged.
- `T_replan`: last actually executed action completion → next legal Action Chunk
  executable, including payload, remaining-history wait, IPC, ACK and commit.
- `T_step`: observation ready → actual action begins, plus separately adjacent
  observation cycle and env.step duration. Distinguish chunk-middle vs boundary.
- Each16-action interval: total CPU/GPU compute, interval union overlapped with
  Sim, and measured foreground blocked intervals. Attribute the causal critical
  path; additive totals are not a latency decomposition when spans overlap.

Stratify first/no-history, steady full16-history and partial-terminal; task,
episode/seed, topology/backend; no initialization contamination of steady P50.
Report count, P50/P95/P99, mean and confidence intervals with estimator/version,
warmup policy and exclusions frozen before execution. P99 with few samples is
explicitly unstable, not concealed. Fixed-input repeated inference diagnoses
compute; real closed-loop matched-initial-state episodes measure total impact.
After actions diverge, later images are not claimed to be identical across arms.

SR protocol: freeze 18 underlying tasks, reset/initial-state fingerprints and
paired episode seeds/max steps/instructions; run A/B/C independently in balanced
order. Report per-task and equal-task-weight overall SR, valid/invalid/pending/
unresolved counts, and paired outcomes 00/01/10/11. For fixed paired N,
deltaSR(pp)=100×(01−10)/N. Use task-stratified paired bootstrap with episode as
the cluster and all arms retained together for deltaSR and latency deltas; record
seed, resample count and 95%CI. Per-arm Wilson CIs do not replace paired-delta CI.
All planned pairs must be accounted for; invalids block clean Gate and must not
be silently dropped or counted as ordinary behavioral failure.

18×2 is functionality smoke only; e.g.18×50 is a separate Owner-authorized study.
Never import earlier multi-Sim SR into the single cost/benefit table. Publish
deltaReplan quantiles/mean and paired-delta summaries, deltaFirstAction,
deltaCriticalPath, actual incremental latency per executed step (not just sumTTT/16),
deltaGPU memory/utilization, SR delta and paired CI. Report B−A/C−A system costs
and C−B same-weight effects. If latency delta CI touches zero, pp/ms is unstable:
show joint uncertainty/Pareto tradeoff rather than divide by near-zero latency.
No benefit, regression or insufficient precision is a valid conclusion. H100
timings do not establish deployment latency on different robot hardware.

## Stages, minimum future edits and acceptance artifacts

The machine-readable [acceptance matrix](acceptance_matrix.json) is authoritative
for proposed Gate coverage, not execution status. Both recipes are specifications,
not existing CLI configs. No project tests are rerun for a docs-only design.

| Stage | Proposed minimum change / artifacts | Permission and exit |
| --- | --- | --- |
| 0 now | This design, matrix, two recipe specs, trace/envelope schemas; Root docs handoff and design validation | GPT independent design review; request APPROVE_DESIGN_STAGE0 or REQUEST_CHANGES, not APPROVE_CODE |
| 1 after design approval | New Child inference Worker/protocol/client and focused synthetic tests; small Encode1 batch helper only if needed, no model maths edits | CPU/static + standalone Worker Draft PR; no production Policy feature integration, GPU or real Sim |
| 2 separately authorized | Real Wan weights/backend parity, B1/4/8 timing, Worker16-Sim/EGL contention and reservations | Numerical pass and actual Sim-GPU benefit required before Action Path integration |
| 3 separately reviewed | Child local-memory client/transaction adapter, feature verifier/API, batch eligibility; Root runtime/config/startup/resources/identity and explicit Worker resource contract | New Formal Pair; CPU before authorized single A/B/C and multi closed loops, SR separately permitted |

Stage1 proposed new paths: `cosmos_framework/inference/robocasa_vae_worker.py`,
`robocasa_vae_protocol.py`, `robocasa_vae_client.py`, and corresponding focused
tests, plus a small pure Visual96/preprocess helper if required to isolate current
transitive training imports. These files do not yet exist. Reuse existing Transaction, OnlineLocalMemory,
Encode1 and Registry; do not build another complete eval framework. Stage3 exact
existing paths will be frozen after source review; no broad package reorganization.

Stage1 synthetic matrix must cover actual process IPC and fake bounded encoder,
B1 immediate/no-peer, B4/8 full/tail/window, mixed steps/order, capacity states,
copy-before-async/alias mutation, exact/changed retry, all signed-field tampering,
wrong key/epoch/lease, unknown boundary, deadline exhaustion, out-of-order consume,
Session isolation, reset/End/restart/drain/quarantine, first-error retention and
owned-process/external-process isolation. Stage3 adds real transaction candidate
abort/commit/ACK and No-TTT zero-Worker/end-to-end deadline integration tests.
CPU mocks cannot close real model numerical, EGL contention or SR Gates.

Review decisions requested: (1) trust model/key provisioning and Stage3 isolation
precondition; (2) bit-exact default and pre-registered tolerance procedure;
(3) common experiment budget boundary; (4) exact boundary reuse and candidate
commit semantics; (5) separate single/multi recipes and paired SR methodology.
Before real runs, Owner must freeze checkpoint/asset/GPU IDs, numerical tolerance
if any, latency non-inferiority margin/SLA and minimum meaningful SR benefit.
Unset criteria/identities keep execution disabled. Optional D may be deferred.

No V3 merge, Gitlink promotion, PR4 edit, server file access, running-job operation,
training, real GPU or long simulation is part of this Stage0 submission. A design
approval only unlocks the specified next CPU implementation stage.
