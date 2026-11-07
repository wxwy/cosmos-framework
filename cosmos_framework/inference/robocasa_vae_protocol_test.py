"""Synthetic stdlib protocol checks; no real encoder, env.step or GPU evidence."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import struct
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

from cosmos_framework.inference.robocasa_vae_protocol import (
    IDENTITY_FIELDS,
    LEASE_FIELDS,
    MAX_FEATURE_WIRE_BYTES,
    MAX_FRAME_WIRE_BYTES,
    RAW15_BYTES,
    RGB_BYTES,
    VERSION,
    VISUAL96_BYTES,
    Frame,
    ProtocolError,
    canonical_json,
    sign_feature,
    source_digest,
    strict_json,
    validate_frame,
    validate_identity,
    validate_lease,
    validate_raw15,
    validate_visual96,
    verify_feature,
)

RGB = bytes(range(256)) * (RGB_BYTES // 256)
RAW15 = struct.pack("<15f", *(index / 16 for index in range(-7, 8)))
VISUAL96 = struct.pack("<96f", *(index / 32 for index in range(-48, 48)))
KEY = bytes(range(32))
LEASE = {
    "server_epoch": "policy-epoch-1",
    "run_lease": "run-1",
    "manifest_digest": "a" * 64,
    "slot_id": "slot-0",
    "generation": 1,
    "task": "CloseFridge-\u51b0\u7bb1",
    "episode_index": 0,
    "attempt_id": "attempt-1",
    "session_id": "session-1",
    "episode_id": "episode-1",
}
IDENTITY = {
    "vae_weights_sha256": "b" * 64,
    "vae_config_sha256": "c" * 64,
    "preprocess_profile_sha256": "d" * 64,
    "visual96_contract_sha256": "e" * 64,
    "encoder_backend": "eager",
    "encoder_dtype": "bfloat16",
}


def make_frame(**changes):
    arguments = {
        "lease": deepcopy(LEASE),
        "chunk_id": "chunk-1",
        "request_key": "request-1",
        "source_step": 3,
        "rgb": RGB,
        "raw15": RAW15,
        "identity": deepcopy(IDENTITY),
        "episode_deadline_ns": 1000,
        "job_deadline_ns": 900,
        "clock_domain": "boot-1:monotonic:offset-0",
        "replan_deadline_ns": None,
    }
    arguments.update(changes)
    return Frame.create(**arguments)


def make_receipt(frame=None):
    return sign_feature(
        make_frame() if frame is None else frame,
        VISUAL96,
        KEY,
        worker_id="worker-1",
        worker_epoch="worker-epoch-1",
        key_id="key-1",
    )


def reference_mac(receipt, key=KEY):
    """Build the specified wire message independently of the protocol helper."""
    header = json.dumps(receipt["header"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    message = b"PSM23-VIS96-v1\x00" + len(header).to_bytes(4, "big") + header
    message += base64.b64decode(receipt["raw15_f32le_b64"])
    message += base64.b64decode(receipt["visual96_f32le_b64"])
    return hmac.new(key, message, "sha256").hexdigest()


class VAEProtocolTest(unittest.TestCase):
    def test_full_public_lease_and_identity_are_detached(self):
        self.assertEqual(
            set(LEASE_FIELDS),
            {
                "server_epoch",
                "run_lease",
                "manifest_digest",
                "slot_id",
                "generation",
                "task",
                "episode_index",
                "attempt_id",
                "session_id",
                "episode_id",
            },
        )
        lease = validate_lease(LEASE)
        identity = validate_identity(IDENTITY)
        self.assertEqual(lease, LEASE)
        self.assertEqual(set(identity), set(IDENTITY_FIELDS))
        self.assertIsNot(lease, LEASE)
        self.assertIsNot(identity, IDENTITY)

    def test_lease_rejects_missing_extra_secret_and_invalid_types(self):
        for field in LEASE_FIELDS:
            with self.subTest(missing=field):
                lease = deepcopy(LEASE)
                del lease[field]
                with self.assertRaises(ProtocolError):
                    validate_lease(lease)
        for name, value in (
            ("capability", "never-public"),
            ("generation", 0),
            ("generation", 2**63),
            ("generation", True),
            ("generation", 1.0),
            ("episode_index", -1),
            ("episode_index", False),
            ("run_lease", {}),
            ("session_id", ""),
            ("slot_id", "x" * 257),
            ("task", "invalid\nidentifier"),
            ("manifest_digest", "A" * 64),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ProtocolError):
                validate_lease({**LEASE, name: value})

    def test_identity_exact_keys_and_supported_encoder_only(self):
        for field in IDENTITY_FIELDS:
            identity = deepcopy(IDENTITY)
            del identity[field]
            with self.subTest(missing=field), self.assertRaises(ProtocolError):
                validate_identity(identity)
        for changes in (
            {"extra": "unbound"},
            {"vae_weights_sha256": "not-a-digest"},
            {"vae_config_sha256": "C" * 64},
            {"encoder_backend": "aot_b8"},
            {"encoder_dtype": "float16"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                validate_identity({**IDENTITY, **changes})
        self.assertEqual(validate_identity({**IDENTITY, "encoder_backend": "aot_b1"})["encoder_backend"], "aot_b1")

    def test_source_digest_uses_original_rgb_domain_shape_and_order(self):
        reference = hashlib.sha256(b"PSM23-RGB-v1\x00" + struct.pack(">III", 256, 512, 3) + RGB).hexdigest()
        self.assertEqual(source_digest(RGB), reference)
        self.assertNotEqual(reference, hashlib.sha256(RGB).hexdigest())
        self.assertNotEqual(
            reference, hashlib.sha256(b"PSM23-RGB-v1\x00" + struct.pack(">III", 512, 256, 3) + RGB).hexdigest()
        )
        changed = bytearray(RGB)
        changed[0] ^= 1
        self.assertNotEqual(source_digest(changed), reference)
        for invalid in (RGB[:-1], RGB + b"\0", "rgb", RGB_BYTES):
            with self.subTest(type=type(invalid).__name__), self.assertRaises(ProtocolError):
                source_digest(invalid)

    def test_frame_owns_buffers_and_nested_input_metadata(self):
        rgb, action = bytearray(RGB), bytearray(RAW15)
        lease, identity = deepcopy(LEASE), deepcopy(IDENTITY)
        frame = make_frame(rgb=memoryview(rgb), raw15=memoryview(action), lease=lease, identity=identity)
        before, fingerprint = frame.header, frame.fingerprint
        rgb[0] ^= 1
        action[0] ^= 1
        lease["session_id"] = "changed-after-submit"
        identity["vae_weights_sha256"] = "f" * 64
        public_header = frame.header
        public_header["lease"]["session_id"] = "changed-read"
        public_header["rgb_shape"][0] = 1
        self.assertEqual(frame.rgb, RGB)
        self.assertEqual(frame.raw15, RAW15)
        self.assertEqual(frame.header, before)
        self.assertEqual(frame.fingerprint, fingerprint)
        self.assertEqual(frame.header["policy_epoch"], LEASE["server_epoch"])
        self.assertIs(type(frame.rgb), bytes)
        self.assertIs(type(frame.raw15), bytes)
        self.assertFalse(hasattr(frame, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            frame.rgb = RGB

    def test_frame_fingerprint_binds_every_input_category(self):
        frame = make_frame()
        variants = (
            make_frame(lease={**LEASE, "episode_id": "episode-2"}),
            make_frame(chunk_id="chunk-2"),
            make_frame(request_key="request-2"),
            make_frame(source_step=4),
            make_frame(rgb=bytes([1]) + RGB[1:]),
            make_frame(raw15=struct.pack("<15f", *([1.0] * 15))),
            make_frame(identity={**IDENTITY, "vae_config_sha256": "f" * 64}),
            make_frame(episode_deadline_ns=1001),
            make_frame(job_deadline_ns=899),
            make_frame(replan_deadline_ns=850),
            make_frame(clock_domain="other-host"),
        )
        for other in variants:
            with self.subTest(header=other.header):
                self.assertNotEqual(other.fingerprint, frame.fingerprint)
        expected = b"PSM23-FRAME-v1\x00"
        for part in (frame.header_json, frame.rgb, frame.raw15):
            expected += len(part).to_bytes(4, "big") + part
        self.assertEqual(frame.fingerprint, hashlib.sha256(expected).hexdigest())

    def test_direct_constructor_and_validation_cannot_skip_body_integrity(self):
        frame = make_frame()
        self.assertEqual(validate_frame(frame), frame)
        for header_json, rgb, raw15 in (
            (frame.header_json, bytes([1]) + RGB[1:], RAW15),
            (frame.header_json, RGB, b"\0" * RAW15_BYTES),
            (b" " + frame.header_json, RGB, RAW15),
            (frame.header_json.decode("ascii"), RGB, RAW15),
        ):
            with self.subTest(header_type=type(header_json).__name__), self.assertRaises(ProtocolError):
                Frame(header_json, rgb, raw15)
        object.__setattr__(frame, "rgb", bytes([1]) + RGB[1:])
        with self.assertRaises(ProtocolError):
            validate_frame(frame)
        with self.assertRaises(ProtocolError):
            make_receipt(frame)

    def test_frame_wire_round_trip_is_exact_and_detached(self):
        frame = make_frame()
        wire = frame.to_wire()
        encoded = canonical_json(wire)
        self.assertLess(len(encoded), MAX_FRAME_WIRE_BYTES)
        self.assertEqual(Frame.from_wire(wire), frame)
        self.assertEqual(Frame.from_wire(encoded), frame)
        wire["header"]["lease"]["episode_id"] = "changed"
        self.assertEqual(frame.header["lease"]["episode_id"], LEASE["episode_id"])
        bad = frame.to_wire()
        bad["rgb_b64"] = base64.b64encode(bytes([1]) + RGB[1:]).decode("ascii")
        with self.assertRaises(ProtocolError):
            Frame.from_wire(bad)
        for value in ({**frame.to_wire(), "extra": 1}, {"header": frame.header}, b" " * (MAX_FRAME_WIRE_BYTES + 1)):
            with self.subTest(type=type(value).__name__), self.assertRaises(ProtocolError):
                Frame.from_wire(value)

    def test_frame_header_rejects_all_missing_or_extra_fields(self):
        frame = make_frame()
        for name in frame.header:
            header = frame.header
            del header[name]
            with self.subTest(missing=name), self.assertRaises(ProtocolError):
                Frame(canonical_json(header), RGB, RAW15)
        for changes in (
            {"extra": 1},
            {"policy_epoch": "wrong-policy"},
            {"source_step": True},
            {"source_step": -1},
            {"source_step": 3.0},
            {"source_rgb_contract": "PNG"},
            {"rgb_layout": "BGR"},
            {"rgb_shape": [512, 256, 3]},
            {"rgb_shape": [256.0, 512, 3]},
            {"rgb_shape": [True, 512, 3]},
            {"version": "PSM23-VIS96-v2"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                Frame(canonical_json({**frame.header, **changes}), RGB, RAW15)

    def test_fixed_finite_f32_bodies_on_creation_signing_and_verification(self):
        self.assertEqual(validate_raw15(RAW15), RAW15)
        self.assertEqual(validate_visual96(VISUAL96), VISUAL96)
        for count, validator in ((15, validate_raw15), (96, validate_visual96)):
            for value in (float("nan"), float("inf"), -float("inf")):
                raw = struct.pack(f"<{count}f", value, *([0.0] * (count - 1)))
                with self.subTest(count=count, value=value), self.assertRaises(ProtocolError):
                    validator(raw)
                with self.assertRaises(ProtocolError):
                    if count == 15:
                        make_frame(raw15=raw)
                    else:
                        sign_feature(make_frame(), raw, KEY, worker_id="w", worker_epoch="e", key_id="k")
                receipt = make_receipt()
                field, digest = (
                    ("raw15_f32le_b64", "raw15_sha256")
                    if count == 15
                    else (
                        "visual96_f32le_b64",
                        "visual96_sha256",
                    )
                )
                receipt[field] = base64.b64encode(raw).decode("ascii")
                receipt["header"][digest] = hashlib.sha256(raw).hexdigest()
                receipt["mac_hex"] = reference_mac(receipt)
                with self.assertRaises(ProtocolError):
                    verify_feature(receipt, KEY)
            for raw in (b"", b"\0" * (count * 4 - 1), b"\0" * (count * 4 + 1), [0.0] * count):
                with self.subTest(count=count, length=len(raw)), self.assertRaises(ProtocolError):
                    validator(raw)

    def test_exact_hmac_wire_contract_and_unicode_canonical_header(self):
        receipt = make_receipt()
        self.assertEqual(receipt["header"]["version"], VERSION)
        self.assertEqual(receipt["mac_hex"], reference_mac(receipt))
        self.assertIn(b"\\u51b0\\u7bb1", canonical_json(receipt["header"]))
        self.assertNotIn("\u51b0\u7bb1".encode("utf-8"), canonical_json(receipt["header"]))
        self.assertEqual(base64.b64decode(receipt["raw15_f32le_b64"]), RAW15)
        self.assertEqual(base64.b64decode(receipt["visual96_f32le_b64"]), VISUAL96)
        self.assertEqual(receipt["header"]["raw15_sha256"], hashlib.sha256(RAW15).hexdigest())
        self.assertEqual(receipt["header"]["visual96_sha256"], hashlib.sha256(VISUAL96).hexdigest())
        self.assertLess(len(canonical_json(receipt)), MAX_FEATURE_WIRE_BYTES)
        self.assertEqual(verify_feature(canonical_json(receipt), KEY), receipt)

    def test_each_header_field_is_mac_bound(self):
        receipt = make_receipt()
        replacements = {
            "version": "PSM23-VIS96-v2",
            "key_id": "key-2",
            "worker_id": "worker-2",
            "worker_epoch": "worker-epoch-2",
            "policy_epoch": "policy-epoch-2",
            "lease": {**LEASE, "session_id": "session-2"},
            "issued_chunk_id": "chunk-2",
            "policy_request_key": "request-2",
            "source_step": 4,
            "source_rgb_sha256": "f" * 64,
            "source_rgb_contract": "wrong-contract",
            "rgb_shape": [512, 256, 3],
            "rgb_layout": "BGR",
            "raw15_sha256": "f" * 64,
            "visual96_sha256": "f" * 64,
            "vae_weights_sha256": "f" * 64,
            "vae_config_sha256": "f" * 64,
            "preprocess_profile_sha256": "f" * 64,
            "visual96_contract_sha256": "f" * 64,
            "encoder_backend": "aot_b1",
            "encoder_dtype": "float32",
            "clock_domain": "other-host",
            "episode_deadline_ns": "1001",
            "job_deadline_ns": "901",
            "replan_deadline_ns": "850",
        }
        self.assertEqual(set(replacements), set(receipt["header"]))
        for name, value in replacements.items():
            changed = deepcopy(receipt)
            changed["header"][name] = value
            with self.subTest(field=name), self.assertRaises(ProtocolError):
                verify_feature(changed, KEY)

    def test_each_lease_field_is_mac_bound(self):
        receipt = make_receipt()
        for name, value in LEASE.items():
            changed = deepcopy(receipt)
            changed["header"]["lease"][name] = value + 1 if type(value) is int else value + "2"
            if name == "manifest_digest":
                changed["header"]["lease"][name] = "f" * 64
            if name == "server_epoch":
                changed["header"]["policy_epoch"] = changed["header"]["lease"][name]
            with self.subTest(field=name), self.assertRaises(ProtocolError):
                verify_feature(changed, KEY)

    def test_receipt_exact_fields_and_canonical_fixed_size_base64(self):
        receipt = make_receipt()
        for name in receipt:
            changed = deepcopy(receipt)
            del changed[name]
            with self.subTest(missing=name), self.assertRaises(ProtocolError):
                verify_feature(changed, KEY)
        for name, value in (
            ("extra", 1),
            ("mac_hex", "A" * 64),
            ("mac_hex", "0" * 63),
            ("raw15_f32le_b64", "!" * 80),
            ("raw15_f32le_b64", base64.b64encode(RAW15 + b"\0").decode("ascii")),
            ("visual96_f32le_b64", "!" * 512),
            ("visual96_f32le_b64", "A" * 510 + "=="),
        ):
            with self.subTest(name=name), self.assertRaises(ProtocolError):
                verify_feature({**receipt, name: value}, KEY)
        changed = deepcopy(receipt)
        changed["header"]["unknown"] = 1
        changed["mac_hex"] = reference_mac(changed)
        with self.assertRaises(ProtocolError):
            verify_feature(changed, KEY)

    def test_body_mutation_and_public_hash_are_not_signing_authority(self):
        for field, raw in (("raw15_f32le_b64", RAW15), ("visual96_f32le_b64", VISUAL96)):
            changed = make_receipt()
            mutated = bytes([raw[0] ^ 1]) + raw[1:]
            changed[field] = base64.b64encode(mutated).decode("ascii")
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                verify_feature(changed, KEY)
        receipt = make_receipt()
        receipt["header"]["source_step"] = 4
        receipt["mac_hex"] = reference_mac(receipt, hashlib.sha256(RGB).digest())
        with self.assertRaises(ProtocolError):
            verify_feature(receipt, KEY)
        for key in (b"x" * 31, b"x" * 33, "x" * 32, b"x" * 32):
            with self.subTest(key_type=type(key).__name__), self.assertRaises(ProtocolError):
                verify_feature(make_receipt(), key)

    def test_authoritative_expectations_reject_cross_session_model_and_epoch(self):
        receipt = make_receipt()
        good = {
            "expected_lease": LEASE,
            "expected_identity": IDENTITY,
            "expected_worker_id": "worker-1",
            "expected_worker_epoch": "worker-epoch-1",
            "expected_key_id": "key-1",
            "expected_clock_domain": "boot-1:monotonic:offset-0",
        }
        self.assertEqual(verify_feature(receipt, KEY, **good), receipt)
        for name, expected in (
            ("expected_lease", {**LEASE, "attempt_id": "new-attempt"}),
            ("expected_lease", {**LEASE, "generation": 2}),
            ("expected_lease", {**LEASE, "run_lease": "other-run"}),
            ("expected_lease", {**LEASE, "episode_index": 1}),
            ("expected_identity", {**IDENTITY, "vae_weights_sha256": "f" * 64}),
            ("expected_worker_id", "worker-2"),
            ("expected_worker_epoch", "worker-epoch-2"),
            ("expected_key_id", "key-2"),
            ("expected_clock_domain", "other-boot"),
        ):
            with self.subTest(name=name), self.assertRaises(ProtocolError):
                verify_feature(receipt, KEY, **{**good, name: expected})

    def test_verified_output_is_detached_and_exact_retry_is_stable(self):
        receipt = make_receipt()
        encoded = canonical_json(receipt)
        verified = verify_feature(receipt, KEY)
        self.assertEqual(canonical_json(make_receipt()), encoded)
        self.assertEqual(canonical_json(verified), encoded)
        receipt["header"]["lease"]["session_id"] = "mutated-after-verify"
        receipt["header"]["rgb_shape"][0] = 1
        self.assertEqual(verified["header"]["lease"]["session_id"], LEASE["session_id"])
        self.assertEqual(verified["header"]["rgb_shape"], [256, 512, 3])

    def test_absolute_deadlines_are_signed_strings_and_never_refreshed(self):
        frame = make_frame(replan_deadline_ns=800)
        receipt = make_receipt(frame)
        self.assertEqual(frame.header["episode_deadline_ns"], "1000")
        self.assertEqual(frame.header["job_deadline_ns"], "900")
        self.assertEqual(frame.header["replan_deadline_ns"], "800")
        for now in (0, 799):
            self.assertEqual(
                verify_feature(receipt, KEY, expected_clock_domain=frame.header["clock_domain"], now_ns=now), receipt
            )
        for now in (800, 900, 1001):
            with self.subTest(now=now), self.assertRaises(ProtocolError):
                verify_feature(receipt, KEY, expected_clock_domain=frame.header["clock_domain"], now_ns=now)
        self.assertEqual(make_receipt(Frame.from_wire(frame.to_wire())), receipt)
        background = make_receipt(make_frame())
        self.assertIsNone(background["header"]["replan_deadline_ns"])
        with self.assertRaises(ProtocolError):
            verify_feature(background, KEY, expected_clock_domain=frame.header["clock_domain"], now_ns=900)
        # A later Replan cannot extend the earlier job's deadline, even when signed.
        later_replan = make_receipt(make_frame(replan_deadline_ns=950))
        with self.assertRaises(ProtocolError):
            verify_feature(later_replan, KEY, expected_clock_domain=frame.header["clock_domain"], now_ns=900)

    def test_deadline_type_range_order_and_clock_domain_fail_closed(self):
        for changes in (
            {"job_deadline_ns": 1001},
            {"replan_deadline_ns": 1001},
            {"episode_deadline_ns": -1},
            {"job_deadline_ns": 900.0},
            {"job_deadline_ns": True},
            {"job_deadline_ns": "9e2"},
            {"job_deadline_ns": "9" * 21},
            {"clock_domain": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ProtocolError):
                make_frame(**changes)
        for now in (-1, True, 1.0):
            with self.subTest(now=now), self.assertRaises(ProtocolError):
                verify_feature(make_receipt(), KEY, expected_clock_domain="boot-1:monotonic:offset-0", now_ns=now)
        with self.assertRaises(ProtocolError):
            verify_feature(make_receipt(), KEY, now_ns=1)

    def test_strict_json_rejects_duplicate_float_nonfinite_and_oversized_wire(self):
        for raw in (
            b'{"x":1,"x":2}',
            b'{"x":{"nested":1,"nested":2}}',
            b'{"x":1.0}',
            b'{"x":1e3}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b"[]",
            b"{} trailing",
            b"\xff",
        ):
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                strict_json(raw)
        receipt = canonical_json(make_receipt())
        duplicated = b'{"header":{},' + receipt[1:]
        with self.assertRaises(ProtocolError):
            verify_feature(duplicated, KEY)
        with self.assertRaises(ProtocolError):
            verify_feature(b" " * (MAX_FEATURE_WIRE_BYTES + 1), KEY)
        self.assertEqual(strict_json(b'{"integer":9007199254740993}')["integer"], 9007199254740993)
        self.assertEqual((len(RAW15), len(VISUAL96)), (RAW15_BYTES, VISUAL96_BYTES))
