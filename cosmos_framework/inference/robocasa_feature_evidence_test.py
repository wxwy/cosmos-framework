"""Synthetic Owner/receipt contracts, not real env.step or loaded-model evidence."""

from __future__ import annotations

import hashlib
import struct
import unittest
from copy import deepcopy

from cosmos_framework.inference.robocasa_feature_evidence import (
    FEATURE_EVIDENCE_FORMAT,
    FEATURE_EVIDENCE_VERSION,
    FeatureEvidenceOwner,
    PolicyBoundary,
    validate_boundary_record,
    validate_post_step_receipt,
)
from cosmos_framework.inference.robocasa_vae_protocol import (
    RGB_BYTES,
    Frame,
    ProtocolError,
    canonical_json,
    sign_feature,
    source_digest,
)

TOKEN = "synthetic-owner-token-private"
CLOCK = "synthetic-boot:monotonic:0"
KEY = bytes(range(32))
IDENTITY = {
    "vae_weights_sha256": "a" * 64,
    "vae_config_sha256": "b" * 64,
    "preprocess_profile_sha256": "c" * 64,
    "visual96_contract_sha256": "d" * 64,
    "encoder_backend": "eager",
    "encoder_dtype": "bfloat16",
}
LEASE = {
    "server_epoch": "policy-1",
    "run_lease": "run-1",
    "manifest_digest": "e" * 64,
    "slot_id": "slot-0",
    "generation": 1,
    "task": "CloseFridge",
    "episode_index": 0,
    "attempt_id": "attempt-1",
    "session_id": "session-1",
    "episode_id": "episode-1",
}


class EvidenceHarness:
    def __init__(self, *, max_sessions=1, bind_lookup=True, history_k=16):
        self.clock = [100]
        self.lease = deepcopy(LEASE)
        self.identity = deepcopy(IDENTITY)
        self.owner = FeatureEvidenceOwner(
            TOKEN,
            clock_ns=lambda: self.clock[0],
            max_sessions=max_sessions,
            policy_identity=self.identity,
            history_k=history_k,
        )
        self.owner.enroll_worker(
            TOKEN,
            worker_id="worker",
            worker_epoch="worker-epoch",
            key_id="key",
            key=KEY,
            identity=self.identity,
            clock_domain=CLOCK,
        )
        self.owner.register_session(
            TOKEN,
            self.lease,
            worker_id="worker",
            actor_id="actor",
            actor_epoch="actor-epoch",
            episode_deadline_ns=10000,
        )
        self.boundaries = {}
        if bind_lookup:
            self.owner.bind_boundary_lookup(lambda session, step: self.boundaries.get((session, step)))

    @staticmethod
    def rgb(step):
        return bytes([step % 256]) * RGB_BYTES

    @staticmethod
    def action(step):
        return struct.pack("<15f", *([step / 64] * 15))

    @staticmethod
    def visual(step):
        return struct.pack("<96f", *([step / 16] * 96))

    def boundary(self, start=0, *, request_key=None, lease=None):
        return PolicyBoundary.create(
            lease=self.lease if lease is None else lease,
            request_key=request_key or f"request-{start}",
            chunk_id=request_key or f"request-{start}",
            source_step=start,
            source_rgb_sha256=source_digest(self.rgb(start)),
            identity=self.identity,
            deadline_ns=9000,
            clock_domain=CLOCK,
            visual96=self.visual(start),
        )

    def publish_boundary(self, start=0):
        boundary = self.boundary(start)
        self.boundaries[(self.lease["session_id"], start)] = boundary
        return boundary

    def commit_chunk(self, start=0):
        return self.owner.commit_action_chunk(
            TOKEN,
            self.lease,
            chunk_id=f"request-{start}",
            request_key=f"request-{start}",
            source_start=start,
            actions_f32le=b"".join(self.action(step) for step in range(start, start + 16)),
            deadline_ns=9000,
        )

    def complete(self, step, **changes):
        self.clock[0] += 10
        arguments = {
            "chunk_id": f"request-{step // 16 * 16}",
            "request_key": f"request-{step // 16 * 16}",
            "source_step": step,
            "source_rgb_sha256": source_digest(self.rgb(step)) if step % (16 // self.owner.history_k) == 0 else None,
            "raw15": self.action(step),
            "step_completed_ns": self.clock[0] - 1,
            "actor_id": "actor",
            "actor_epoch": "actor-epoch",
        }
        arguments.update(changes)
        return self.owner.complete_post_step(TOKEN, self.lease, **arguments)

    def feature(self, step, **changes):
        frame_arguments = {
            "lease": self.lease,
            "chunk_id": f"request-{step // 16 * 16}",
            "request_key": f"request-{step // 16 * 16}",
            "source_step": step,
            "rgb": self.rgb(step),
            "raw15": self.action(step),
            "identity": self.identity,
            "episode_deadline_ns": 10000,
            "job_deadline_ns": 9500,
            "clock_domain": CLOCK,
        }
        frame_arguments.update(changes)
        return sign_feature(
            Frame.create(**frame_arguments),
            self.visual(step),
            KEY,
            worker_id="worker",
            worker_epoch="worker-epoch",
            key_id="key",
        )

    def rows(self, start=0, count=16):
        return [
            {
                "source_step": step,
                "post_step_receipt": self.complete(step),
                "feature": None if step % 16 == 0 or step % (16 // self.owner.history_k) else self.feature(step),
            }
            for step in range(start, start + count)
        ]

    def payload(self, rows, *, consumer_step=16):
        payload = {
            "session_id": self.lease["session_id"],
            "episode_id": self.lease["episode_id"],
            "consumer_step": consumer_step,
            "reset": consumer_step == 0,
            "image_size": 256,
            "evidence_version": FEATURE_EVIDENCE_VERSION,
            "evidence_format": FEATURE_EVIDENCE_FORMAT,
            "evidence": rows,
        }
        if self.owner.history_k != 16:
            from cosmos_framework.inference.robocasa_feature_evidence import (
                SPARSE_EVIDENCE_FORMAT,
                SPARSE_EVIDENCE_VERSION,
            )

            payload.update(
                evidence_version=SPARSE_EVIDENCE_VERSION,
                evidence_format=SPARSE_EVIDENCE_FORMAT,
                local_ttt_history_k=self.owner.history_k,
            )
        return payload

    def admit(self, rows, *, consumer_step=16, previous_step=0, **changes):
        arguments = {"lease": self.lease, "deadline_ns": 8000, "clock_domain": CLOCK, "previous_step": previous_step}
        arguments.update(changes)
        return self.owner.admit(self.payload(rows, consumer_step=consumer_step), **arguments)


class FeatureEvidenceTest(unittest.TestCase):
    def ready(self):
        harness = EvidenceHarness()
        harness.publish_boundary()
        harness.commit_chunk()
        return harness

    def test_real_source_15_plus_one_requires_both_authorities(self):
        h = self.ready()
        rows = h.rows()
        batch = h.admit(rows)
        self.assertEqual(batch.source_steps, tuple(range(16)))
        self.assertEqual(batch.visual96, tuple(h.visual(step) for step in range(16)))
        self.assertEqual(batch.raw15, tuple(h.action(step) for step in range(16)))
        self.assertIsNone(rows[0]["feature"])
        self.assertEqual(len([row for row in rows if row["feature"] is not None]), 15)
        self.assertEqual(h.admit(deepcopy(rows)), batch)
        self.assertEqual(len(h.owner._sessions[h.lease["session_id"]].completed), 16)

    def test_sparse_continuation_preserves_real_steps_and_bounded_ledger(self):
        h = self.ready()
        old_rows = h.rows()
        h.admit(old_rows)
        h.publish_boundary(16)
        h.commit_chunk(16)
        new_rows = h.rows(16)
        batch = h.admit(new_rows, consumer_step=32, previous_step=16)
        self.assertEqual(batch.source_steps, tuple(range(16, 32)))
        self.assertEqual(new_rows[1]["feature"]["header"]["source_step"], 17)
        self.assertEqual(len(h.owner._sessions[h.lease["session_id"]].completed), 16)
        with self.assertRaises(ProtocolError):
            h.admit(old_rows)

    def test_unattested_or_mismatched_loaded_identity_and_b4_b8_are_rejected(self):
        owner = FeatureEvidenceOwner(TOKEN, clock_ns=lambda: 100)
        enroll = dict(worker_id="w", worker_epoch="e", key_id="k", key=KEY, identity=IDENTITY, clock_domain=CLOCK)
        with self.assertRaises(ProtocolError):
            owner.enroll_worker(TOKEN, **enroll)
        owner.attest_policy_identity(TOKEN, IDENTITY)
        for bmax in (4, 8, True):
            with self.subTest(bmax=bmax), self.assertRaises(ProtocolError):
                owner.enroll_worker(TOKEN, **enroll, bmax=bmax)
        with self.assertRaises(ProtocolError):
            owner.enroll_worker(TOKEN, **{**enroll, "identity": {**IDENTITY, "vae_config_sha256": "f" * 64}})
        with self.assertRaises(ProtocolError):
            owner.attest_policy_identity(TOKEN, {**IDENTITY, "vae_weights_sha256": "f" * 64})
        copied = owner.identity
        copied["vae_weights_sha256"] = "f" * 64
        self.assertEqual(owner.identity, IDENTITY)

    def test_owner_control_and_committed_chunk_are_required(self):
        h = EvidenceHarness()
        with self.assertRaises(PermissionError):
            h.owner.register_session(
                "untrusted-actor-token",
                h.lease,
                worker_id="worker",
                actor_id="actor",
                actor_epoch="actor-epoch",
                episode_deadline_ns=10000,
            )
        with self.assertRaises(ProtocolError):
            h.commit_chunk()
        h.publish_boundary()
        with self.assertRaises(ProtocolError):
            h.complete(0)
        h.commit_chunk()
        with self.assertRaises(ProtocolError):
            h.complete(1)
        with self.assertRaises(ProtocolError):
            h.complete(0, actor_epoch="other-actor")
        with self.assertRaises(ProtocolError):
            h.complete(0, raw15=h.action(1))
        with self.assertRaises(ProtocolError):
            h.complete(0, source_rgb_sha256=source_digest(h.rgb(1)))
        with self.assertRaises(ProtocolError):
            h.complete(0, step_completed_ns=0)
        self.assertEqual(h.owner._sessions[h.lease["session_id"]].frontier, 0)

    def test_poststep_exact_retry_cannot_change_timestamp_or_identity(self):
        h = self.ready()
        receipt = h.complete(0)
        arguments = {
            "chunk_id": "request-0",
            "request_key": "request-0",
            "source_step": 0,
            "source_rgb_sha256": receipt["source_rgb_sha256"],
            "raw15": h.action(0),
            "step_completed_ns": int(receipt["step_completed_ns"]),
            "actor_id": "actor",
            "actor_epoch": "actor-epoch",
        }
        self.assertEqual(h.owner.complete_post_step(TOKEN, h.lease, **arguments), receipt)
        self.assertEqual(h.owner._sessions[h.lease["session_id"]].frontier, 1)
        with self.assertRaises(ProtocolError):
            h.owner.complete_post_step(
                TOKEN, h.lease, **{**arguments, "step_completed_ns": arguments["step_completed_ns"] + 1}
            )
        with self.assertRaises(PermissionError):
            h.owner.complete_post_step("untrusted-actor-token", h.lease, **arguments)

    def test_public_poststep_shape_is_not_an_authoritative_completion(self):
        h = self.ready()
        rows = h.rows()
        changed = deepcopy(rows)
        changed[2]["post_step_receipt"]["actor_id"] = "forged"
        validate_post_step_receipt(changed[2]["post_step_receipt"])
        with self.assertRaises(ProtocolError):
            h.admit(changed)
        h.owner._sessions[h.lease["session_id"]].completed.pop(2)
        with self.assertRaises(ProtocolError):
            h.admit(rows)

    def test_missing_reordered_future_and_renumbered_history_fail_closed(self):
        h = self.ready()
        rows = h.rows()
        for changed, consumer in ((rows[:-1], 16), (list(reversed(rows)), 16), (rows, 17), (rows, 10**15)):
            with self.subTest(consumer=consumer), self.assertRaises(ProtocolError):
                h.admit(changed, consumer_step=consumer)
        changed = deepcopy(rows)
        changed[1]["source_step"] = 0
        with self.assertRaises(ProtocolError):
            h.admit(changed)

    def test_feature_mac_rgb_action_chunk_and_absolute_budget_are_bound(self):
        h = self.ready()
        rows = h.rows()
        for changes in (
            {"rgb": h.rgb(2)},
            {"raw15": h.action(2)},
            {"request_key": "other-request"},
            {"chunk_id": "other-chunk"},
            {"source_step": 2},
            {"episode_deadline_ns": 9999},
        ):
            changed = deepcopy(rows)
            changed[1]["feature"] = h.feature(1, **changes)
            with self.subTest(changes=list(changes)), self.assertRaises(ProtocolError):
                h.admit(changed)
        for header_changes in ({"worker_epoch": "new-epoch"}, {"source_rgb_sha256": "f" * 64}):
            changed = deepcopy(rows)
            changed[1]["feature"]["header"].update(header_changes)
            with self.subTest(changes=header_changes), self.assertRaises(ProtocolError):
                h.admit(changed)
        changed = deepcopy(rows)
        changed[0]["feature"] = h.feature(0)
        with self.assertRaises(ProtocolError):
            h.admit(changed)

    def test_original_deadlines_clock_and_expired_job_are_not_refreshed(self):
        h = self.ready()
        rows = h.rows()
        for changes in ({"deadline_ns": 10001}, {"clock_domain": "other-host"}, {"deadline_ns": h.clock[0]}):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                h.admit(rows, **changes)
        h.clock[0] = 9600
        with self.assertRaises(ProtocolError):
            h.admit(rows, deadline_ns=9900)
        self.assertEqual(rows[1]["feature"]["header"]["job_deadline_ns"], "9500")

    def test_worker_rotation_revokes_old_session_and_publication(self):
        h = self.ready()
        rows = h.rows()
        h.owner.enroll_worker(
            TOKEN,
            worker_id="worker",
            worker_epoch="new-epoch",
            key_id="new-key",
            key=b"z" * 32,
            identity=IDENTITY,
            clock_domain=CLOCK,
        )
        for operation in (lambda: h.admit(rows), lambda: h.owner.boundary_record(TOKEN, h.lease, 0)):
            with self.assertRaises(ProtocolError):
                operation()
        with self.assertRaises(ProtocolError), h.owner.publication_guard((h.boundary(),)):
            self.fail("revoked boundary must never reach snapshot publication")
        h.owner.end_session(TOKEN, h.lease)
        with self.assertRaises(RuntimeError):
            h.owner.register_session(
                TOKEN,
                h.lease,
                worker_id="worker",
                actor_id="actor",
                actor_epoch="actor-epoch",
                episode_deadline_ns=10000,
            )

    def test_revoke_cannot_reenroll_the_same_worker_epoch_or_signing_key(self):
        h = self.ready()
        h.owner.revoke_worker(TOKEN, "worker", "worker-epoch")
        with self.assertRaises(ProtocolError):
            h.owner.enroll_worker(
                TOKEN,
                worker_id="worker",
                worker_epoch="worker-epoch",
                key_id="key",
                key=KEY,
                identity=IDENTITY,
                clock_domain=CLOCK,
            )

    def test_session_end_isolated_and_old_lease_cannot_read_boundary(self):
        h = EvidenceHarness(max_sessions=2)
        h.publish_boundary()
        h.commit_chunk()
        other = {**h.lease, "session_id": "session-2", "episode_id": "episode-2", "slot_id": "slot-1"}
        h.owner.register_session(
            TOKEN, other, worker_id="worker", actor_id="actor-2", actor_epoch="actor-epoch-2", episode_deadline_ns=10000
        )
        boundary = h.boundary(lease=other)
        h.boundaries[(other["session_id"], 0)] = boundary
        self.assertEqual(h.owner.boundary_record(TOKEN, other, 0), boundary.record())
        self.assertEqual(h.owner.end_session(TOKEN, h.lease), {"ended": True, "duplicate": False})
        self.assertEqual(h.owner.end_session(TOKEN, h.lease), {"ended": True, "duplicate": True})
        with self.assertRaises(ProtocolError):
            h.owner.boundary_record(TOKEN, h.lease, 0)
        self.assertEqual(h.owner.boundary_record(TOKEN, other, 0), boundary.record())

    def test_immutable_boundary_and_exact_public_schema(self):
        h = self.ready()
        boundary = h.boundary()
        header = boundary.header
        header["lease"]["session_id"] = "changed"
        self.assertEqual(boundary.header["lease"], h.lease)
        record = boundary.record()
        self.assertEqual(validate_boundary_record(record), record)
        self.assertEqual(
            boundary.fingerprint,
            hashlib.sha256(
                b"PSM24-BOUNDARY-v1\0"
                + struct.pack(">I", len(boundary.header_json))
                + boundary.header_json
                + boundary.visual96
            ).hexdigest(),
        )
        with self.assertRaises(ProtocolError):
            PolicyBoundary(boundary.header_json, b"\0" * 383)
        with self.assertRaises(ProtocolError):
            validate_boundary_record({**record, "capability": TOKEN})
        receipt = h.complete(0)
        for changes in ({"success": False}, {"source_step": True}, {"step_completed_ns": "0110"}, {"extra": True}):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                validate_post_step_receipt({**receipt, **changes})
        self.assertNotIn(TOKEN.encode(), canonical_json(receipt))
