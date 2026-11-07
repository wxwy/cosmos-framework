# Issue23 Stage1 — standalone shared VAE Worker / CPU review

Stage0 authority is exact Child design `8a170d56014af937b146b35e2b89753313614785`,
independently approved as `APPROVE_DESIGN_STAGE0` in Root review commit
`5c66ea37` and canonical ledger `77f130bb`. Owner additionally authorized Stage1.
This candidate is an independent standalone implementation, not PR4 and not a
production Policy Action Path integration. PR5 remains the unchanged design PR.

## Scope and authority

New modules under `cosmos_framework/inference/` separate immutable frame/receipt
validation, bounded shared Worker scheduling, local Unix IPC and a loaded-encoder
adapter. One Worker can serve multiple independent Sessions; Bvae1/4/8 is
independent of Policy B. Single-Sim Worker B1 requires wait0 and dispatches without
waiting for peers. Existing Policy, Local Memory, simulation, training, parameters,
checkpoints, native batching and M/B Resume logic are not modified.

The CPU harness supplies fake deterministic encoders and a trusted test control
plane. It can validate dispatch, copied bytes, signatures, legal Session/step
ranges, exact retries, result ownership, deadlines and lifecycle. It cannot
attest that a real Sim called env.step, or prove real Wan numerics/performance.
There is no code path from these receipts into Policy generation in this stage.

## Lifecycle and evidence ownership

The standalone Owner control plane enrolls bounded Sessions, authorizes issued
chunks and advances their completed source frontier. Sim credentials are scoped
to their own Session and cannot enroll or complete frontiers. The signing key
and Worker-local control credential are separate from the production global
Owner token; no such global token is used by this component. HMAC proves the
binding of encoder-produced bytes, not correctness of a simulation event.

Frame input is an owned copy of canonical HWC uint8 RGB and finite executed
raw15 float32 bytes; results are finite Visual96 float32 bytes and the approved
signed public identity. Validation precedes READY. Worker batching uses compatible
READY jobs, full-batch/window/deadline dispatch and per-Session fairness. Queued,
active and retained items remain bounded; dequeuing does not release retained
resource obligations. Exact retries never cause another encode or state update.
Session End must reject late submissions/results and drain only its own work;
uncertain active work keeps quarantine. A Worker restart has a new epoch/key and
cannot make old receipts fresh. No FastState is published by this component.

An injected synchronous CPU encoder owns all its work until it returns or raises.
A backend that cannot prove this must raise `EncoderDrainUnproven`; the Worker
must keep a sticky quarantine and decline positive drain/End receipts. The loaded
adapter treats CUDA exceptions this way. This is conservative failure handling,
not real-device drain evidence; Root process reaping and GPU recovery are later
Gates.

Unix IPC is length-prefixed bounded JSON plus immutable base64 payloads, with
absolute monotonic deadlines and separate authenticated control access. It does
not use pickle, remote network access or unbounded data queues. Socket cleanup
must target the owned inode only. CPU process tests retain their original JUnit
and temporary OS evidence. Production Registry/graphics/CUDA resource placement
is deferred to the separately reviewed integration stage.

## Numerical adapter limits

The adapter accepts an already-loaded encoder; no checkpoint discovery/download
or GPU launch is added. Its tests use CPU doubles to validate exact preprocessing,
normalization, the batch axis `[B,3,1,H,W]`, full pre-crop Visual96 pooling and
output ordering. It does not turn independent T1 into T16 video encoding or
share retained streaming caches. Existing Wan B1 AOT guard is untouched; batched
B4/B8 eager versus B1 AOT remain separately labeled backends.

Real loaded Wan weights/configuration identity needs Owner attestation. CPU
synthetic equality is not real Wan BF16 parity. Stage2 must compare real latent
and Visual96 with preregistered tolerances and actual H/W/backend, as well as EGL
contention, memory/utilization and performance, before integration is allowed.

## Review and execution boundaries

Request a fresh `APPROVE_CODE` / `REQUEST_CHANGES` for the new exact implementation
SHA and standalone Stage1 scope. Author CPU/static results will be linked from
the Root handoff; the approved Stage0 documentation checks are not reused as
Worker behavior evidence. Any failures, skipped tests, unresolved cleanup or
source changes during a run keep that author Gate open.

The following remain **NOT_RUN / NOT_IMPLEMENTED** in this stage:

- Policy-owned boundary Visual96 caching/reuse (the required future 15+1 path).
- Worker range adaptation for that 15+1 path: Stage1 encodes contiguous sources;
  Stage3 must independently authorize the 15-frame encoder subrange while keeping
  actual source steps (1..15,17..31,...) and Policy boundary0/16/... distinct. No
  invented source renumbering, implicit skipped evidence or extra boundary Encode1.
- Real post-env.step provenance, OS trust isolation and Policy feature ingestion.
- Policy B1 actual zero-peer-wait trace and READY-only Policy scheduler changes.
- Private FastState transaction integration, full single/multi Replan traces.
- Real Wan/GPU/EGL/RoboCasa, A/B/C latency, throughput, paired SR and ablation D.

No official V3/Gitlink promotion, PR4/PR5 merge, real GPU or simulation, training,
running-job operation, long study or 900EP evaluation is authorized here. Future
single cost/benefit P0 and multi throughput are independent Gates; this CPU
candidate provides the standalone foundation, not their performance conclusion.
