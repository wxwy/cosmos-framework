# Issue24 Stage3A — B1 Policy integration candidate

Implementation basis: Root design `afd0b21f1f994bdf2b21cfbaa4a260ced8eeaaf9`
and Child `0a2119782fb5078fee3359f9ef7383d788744458`. This candidate requires
a new independent Code Gate. Real Wan/Policy/GPU/EGL/RoboCasa execution and P0
latency/SR results are NOT_RUN. PR4/5/6 and official V3 are not modified.

## Existing authorities retained

The native closed-loop path copies pre-action RGB and canonical executed raw15;
only a successful env.step return invokes its scoped Actor completion channel.
Root verifies the admitted Actor/Lease and an already committed native response,
then Policy validates the canonical issued action and records the completion.
Sim possesses its own scoped credentials, never the global Policy Owner token,
Worker control capability or signing key. The explicit trust model is
`registry_same_trust_domain`: software custody of a Registry-managed Actor does
not constitute cryptographic proof of physics or OS isolation. Real execution
still requires the Owner's approved controlled-run trust assumptions.

Policy observes its original current-frame Encode1 before latent crop. The
observer is a default no-op; it never invokes the tokenizer. Its private
Visual96 candidate is staged into the existing Local Memory visual record and
published with FastState and cached Response in the same original snapshot.
Abort, invalid completion, epoch revocation and End cannot publish the candidate.
Source0/16/32 boundaries use that Policy cache; Worker Owner-only boundary_skip
atomically advances all four real source frontiers. Frames1..15/17..31 keep their
true indices and use independent T1 Worker encoding, without renumbering.

Signed features additionally require the Policy Owner's matching post-step
ledger, full Lease, Worker epoch/key, source/action/chunk/digests, loaded VAE
identity and original deadline. Existing OnlineMemoryRequest and model-owned
prepare/commit/abort implement C's causal16×T1. Legacy B still uses16 Policy
Encode1/T16; A has no history Worker or Local update. No second FastState store
or transaction framework is introduced.

## Explicit single profiles and lifecycle

Root configuration selects `single_latency_arm: A|B|C`, PhaseB, exactly one Policy
GPU/one Sim and Policy cap1. A is `local_memory_mode: off`, B/C are required;
A/B use `local_evidence_profile: server_vae`, C uses `sim_shared_vae` and
`batch_wait_ms: 0`. A/B retain their old valid wait setting; with cap1 the queue
dispatches immediately without a peer. B and C must use the same Local checkpoint,
sampler/precision/state15/raw15/H16 and20Hz data semantics. A uses its separately
frozen native No-TTT checkpoint; its SR difference is not solely a TTT effect.

C additionally requires the exact public `sim_shared_vae` configuration keys:
`vae_weights`, `vae_weights_sha256`, `vae_config`, `vae_config_sha256`,
`egl_vendor_json`, `egl_vendor_sha256`, `bmax:1`, `wait_ms:0`, `trust_model`,
`resources`. Paths/hashes and resource settings are frozen in the new opt-in
Manifest execution identity; legacy exact schemas and full M/B Resume comparison
remain strict. No credentials belong in this configuration or its Manifest.

The Worker is a run-level Registry peer with private FD bootstrap and bounded
IPC. Actual loaded tensors/scale/config are hashed independently in Policy and
Worker; supplied filenames or hashes alone are not attestations. C currently
requires frozen eager BF16 Wan; an installed incompatible AOT backend fails
closed instead of silently changing compilation. Actual loaded parameter device
UUID/PCI must match the selected device. Only Sim receives the NVIDIA-only EGL
vendor environment; existing physical/current-context checks remain enabled.

Per-Session End revokes the evidence owner, drains that Worker Session and ends
the native lease; other Sessions and the shared process remain independent.
Run close drains/reaps only Registry-owned groups. An uncertain device drain or
failed peer prevents positive quiescence and reuse. Restarts require fresh epochs,
keys and Sessions. B4/B8 remain available in standalone research but are rejected
at the new Policy feature admission seam until the separate Stage3B Gate.

## Budgets, READY and raw profiling

C freezes one Replan budget at the completed-action boundary, before payload,
feature flush or HTTP. Integer/float conversion can only tighten it. Retries,
candidate updates, generation and ACK retain that budget; cleanup has a separate
bounded budget and cannot resurrect inference. Feature preparation runs on the
same existing model executor without holding the actor admission lock. Only a
prepared candidate enters native READY; C Policy B1/wait0 has no peer delay.

P0 profiles optionally retain bounded raw host spans and correct-stream CUDA
events. CUDA events are resolved after existing transfers/quiescence without
per-operation global synchronization. Unavailable/pending GPU events, dropped
trace rows or overflow are explicitly incomplete, never synthetic GPU timing.
Sim raw traces are atomic sidecars; Core JSON stores only path/SHA/count. Immutable
Responses contain pre-publication compute traces; final commit/ACK spans are
retained as bounded `PSM24_POLICY_TRACE` records in the owned Policy process log.
Intervals that overlap must not be added as serial latency; EGL not separately
observable is explicitly included in env.step, not invented as a separate sample.

Root author handoff supplies the exact new Pair and focused CPU commands/JUnit,
source hashes, local process/FD/cleanup proof and open limitations. These CPU
substitutes establish contracts, not new real-model numerical or performance
results. Paired real A/B/C and larger SR still require separate Owner Gates.
