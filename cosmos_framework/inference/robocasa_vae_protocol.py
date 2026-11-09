"""Standalone Issue23 wire protocol; no model, GPU or Policy Action Path imports.

A valid MAC authenticates an enrolled encoder's bytes. It does not attest that
an environment step completed. Issuer, causal frontier, enrollment/revocation
and lifecycle checks belong to the trusted Worker/Policy owners.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any

VERSION = "PSM23-VIS96-v1"
RGB_SHAPE = (256, 512, 3)
RGB_LAYOUT = "HWC_UINT8_RGB"
RGB_CONTRACT = "PSM23-RGB-v1_HWC_UINT8_RGB"
RGB_BYTES = 256 * 512 * 3
RAW15_BYTES = 15 * 4
VISUAL96_BYTES = 96 * 4
MAX_FRAME_WIRE_BYTES = 1 << 20
MAX_FEATURE_WIRE_BYTES = 16384
MAX_HEADER_BYTES = 8192


def validate_history_k(value):
    if type(value) is not int or value not in (1, 2, 4, 8, 16):
        raise ValueError("local_ttt_history_k must be one of1/2/4/8/16")
    return value


def selected_history_step(source_step, chunk_start, history_k):
    """Selection is relative to the actual completed H16 Action chunk."""
    stride = 16 // validate_history_k(history_k)
    if (
        type(source_step) is not int
        or type(chunk_start) is not int
        or chunk_start < 0
        or chunk_start % 16
        or not 0 <= source_step - chunk_start < 16
    ):
        raise ValueError("source step must belong to its real H16 Action chunk")
    return (source_step - chunk_start) % stride == 0


LEASE_FIELDS = (
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
)
IDENTITY_FIELDS = (
    "vae_weights_sha256",
    "vae_config_sha256",
    "preprocess_profile_sha256",
    "visual96_contract_sha256",
    "encoder_backend",
    "encoder_dtype",
)
_FRAME_FIELDS = frozenset(
    (
        "version",
        "policy_epoch",
        "lease",
        "issued_chunk_id",
        "policy_request_key",
        "source_step",
        "source_rgb_sha256",
        "source_rgb_contract",
        "rgb_shape",
        "rgb_layout",
        "raw15_sha256",
        "clock_domain",
        "episode_deadline_ns",
        "job_deadline_ns",
        "replan_deadline_ns",
        *IDENTITY_FIELDS,
    )
)
_FEATURE_FIELDS = _FRAME_FIELDS | {"key_id", "worker_id", "worker_epoch", "visual96_sha256"}
_RECEIPT_FIELDS = frozenset(("header", "raw15_f32le_b64", "visual96_f32le_b64", "mac_hex"))
_FRAME_WIRE_FIELDS = frozenset(("header", "rgb_b64", "raw15_f32le_b64"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NS = re.compile(r"[0-9]{1,20}\Z")
_RGB_DOMAIN = b"PSM23-RGB-v1\0"
_MAC_DOMAIN = b"PSM23-VIS96-v1\0"
_FRAME_DOMAIN = b"PSM23-FRAME-v1\0"


class ProtocolError(ValueError):
    """Malformed, unauthenticated or mismatched protocol data."""


class EncoderDrainUnproven(RuntimeError):
    """Backend failed without proving its device work drained; owner must quarantine."""


def _object(value: Any, fields, name: str) -> dict:
    if type(value) is not dict or value.keys() != set(fields):
        raise ProtocolError(f"{name} requires exact fields")
    return value


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ProtocolError(f"invalid {name}")
    return value


def _digest(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ProtocolError(f"invalid {name}")
    return value


def _nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolError(f"invalid {name}")
    return value


def _ns(value: Any, name: str, *, input_value: bool = False) -> str:
    if input_value and type(value) is int:
        value = str(value)
    if type(value) is not str or _NS.fullmatch(value) is None:
        raise ProtocolError(f"invalid {name}")
    return value


def _owned_bytes(value: Any, length: int, name: str) -> bytes:
    try:
        view = memoryview(value)
    except TypeError as error:
        raise ProtocolError(f"{name} requires a byte buffer") from error
    if view.itemsize != 1 or view.nbytes != length:
        raise ProtocolError(f"{name} requires {length} bytes")
    return view.tobytes()


def _finite_f32(value: Any, count: int, name: str) -> bytes:
    raw = _owned_bytes(value, 4 * count, name)
    if not all(math.isfinite(item) for item in struct.unpack(f"<{count}f", raw)):
        raise ProtocolError(f"{name} contains non-finite float32")
    return raw


def validate_raw15(value: Any) -> bytes:
    """Copy exactly fifteen finite little-endian float32 action values."""
    return _finite_f32(value, 15, "raw15")


def validate_visual96(value: Any) -> bytes:
    """Copy exactly ninety-six finite little-endian float32 feature values."""
    return _finite_f32(value, 96, "visual96")


def validate_lease(value: dict) -> dict:
    """Detach the exact public SessionLease.identity(); capabilities are forbidden."""
    value = _object(value, LEASE_FIELDS, "lease")
    result = {}
    for name in LEASE_FIELDS:
        item = value[name]
        if name == "manifest_digest":
            result[name] = _digest(item, name)
        elif name == "generation":
            if type(item) is not int or not 1 <= item < 2**63:
                raise ProtocolError("invalid generation")
            result[name] = item
        elif name == "episode_index":
            result[name] = _nonnegative(item, name)
        else:
            result[name] = _identifier(item, name)
    return result


def validate_identity(value: dict) -> dict:
    """Validate identities supplied by Owner attestation, without claiming to attest files."""
    value = _object(value, IDENTITY_FIELDS, "encoder identity")
    result = {name: _digest(value[name], name) for name in IDENTITY_FIELDS[:4]}
    if type(value["encoder_backend"]) is not str or value["encoder_backend"] not in ("eager", "aot_b1"):
        raise ProtocolError("invalid encoder_backend")
    if value["encoder_dtype"] != "bfloat16":
        raise ProtocolError("invalid encoder_dtype")
    result.update(encoder_backend=value["encoder_backend"], encoder_dtype="bfloat16")
    return result


def canonical_json(value: Any) -> bytes:
    """The approved canonical encoding, after the caller's strict schema validation."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ProtocolError("invalid canonical JSON") from error


def strict_json(raw: bytes, *, max_bytes: int = MAX_FRAME_WIRE_BYTES) -> dict:
    """Bounded JSON object parsing with duplicate fields and float tokens forbidden."""
    if type(raw) is not bytes or len(raw) > max_bytes:
        raise ProtocolError("invalid JSON byte length")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError("duplicate JSON field")
            result[key] = value
        return result

    def invalid(_):
        raise ProtocolError("float/non-finite JSON token forbidden")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=invalid, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ProtocolError("invalid strict JSON") from error
    if type(value) is not dict:
        raise ProtocolError("JSON object required")
    return value


def source_digest(rgb: Any) -> str:
    """Digest original C-order HWC uint8 RGB, before resize/normalization/padding."""
    raw = _owned_bytes(rgb, RGB_BYTES, "rgb")
    return hashlib.sha256(_RGB_DOMAIN + struct.pack(">III", *RGB_SHAPE) + raw).hexdigest()


def _header(value: dict, *, feature: bool) -> dict:
    value = _object(value, _FEATURE_FIELDS if feature else _FRAME_FIELDS, "header")
    lease = validate_lease(value["lease"])
    identity = validate_identity({name: value[name] for name in IDENTITY_FIELDS})
    if value["version"] != VERSION or value["policy_epoch"] != lease["server_epoch"]:
        raise ProtocolError("version or policy_epoch mismatch")
    if (
        value["source_rgb_contract"] != RGB_CONTRACT
        or type(value["rgb_shape"]) is not list
        or any(type(item) is not int for item in value["rgb_shape"])
        or value["rgb_shape"] != list(RGB_SHAPE)
        or value["rgb_layout"] != RGB_LAYOUT
    ):
        raise ProtocolError("source RGB contract mismatch")
    result = {
        "version": VERSION,
        "policy_epoch": lease["server_epoch"],
        "lease": lease,
        "issued_chunk_id": _identifier(value["issued_chunk_id"], "issued_chunk_id"),
        "policy_request_key": _identifier(value["policy_request_key"], "policy_request_key"),
        "source_step": _nonnegative(value["source_step"], "source_step"),
        "source_rgb_sha256": _digest(value["source_rgb_sha256"], "source_rgb_sha256"),
        "source_rgb_contract": RGB_CONTRACT,
        "rgb_shape": list(RGB_SHAPE),
        "rgb_layout": RGB_LAYOUT,
        "raw15_sha256": _digest(value["raw15_sha256"], "raw15_sha256"),
        "clock_domain": _identifier(value["clock_domain"], "clock_domain"),
        "episode_deadline_ns": _ns(value["episode_deadline_ns"], "episode_deadline_ns"),
        "job_deadline_ns": _ns(value["job_deadline_ns"], "job_deadline_ns"),
        "replan_deadline_ns": None,
        **identity,
    }
    if value["replan_deadline_ns"] is not None:
        result["replan_deadline_ns"] = _ns(value["replan_deadline_ns"], "replan_deadline_ns")
    episode = int(result["episode_deadline_ns"])
    if int(result["job_deadline_ns"]) > episode:
        raise ProtocolError("job deadline exceeds original Episode deadline")
    if result["replan_deadline_ns"] is not None and int(result["replan_deadline_ns"]) > episode:
        raise ProtocolError("Replan deadline exceeds original Episode deadline")
    if feature:
        for name in ("key_id", "worker_id", "worker_epoch"):
            result[name] = _identifier(value[name], name)
        result["visual96_sha256"] = _digest(value["visual96_sha256"], "visual96_sha256")
    if len(canonical_json(result)) > MAX_HEADER_BYTES:
        raise ProtocolError("header byte limit exceeded")
    return result


def _decode_b64(value: Any, length: int, name: str) -> bytes:
    if type(value) is not str or len(value) != 4 * ((length + 2) // 3):
        raise ProtocolError(f"invalid {name} base64 length")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ProtocolError(f"invalid {name} base64") from error
    if len(raw) != length or base64.b64encode(raw).decode("ascii") != value:
        raise ProtocolError(f"noncanonical {name} base64")
    return raw


@dataclass(frozen=True, slots=True)
class Frame:
    """Owned immutable bytes; neither caller mutation nor header reads expose storage."""

    header_json: bytes
    rgb: bytes
    raw15: bytes

    def __post_init__(self):
        if type(self.header_json) is not bytes:
            raise ProtocolError("canonical header bytes required")
        header = _header(strict_json(self.header_json, max_bytes=MAX_HEADER_BYTES), feature=False)
        if canonical_json(header) != self.header_json:
            raise ProtocolError("noncanonical frame header")
        rgb = _owned_bytes(self.rgb, RGB_BYTES, "rgb")
        raw15 = validate_raw15(self.raw15)
        if (
            source_digest(rgb) != header["source_rgb_sha256"]
            or hashlib.sha256(raw15).hexdigest() != header["raw15_sha256"]
        ):
            raise ProtocolError("frame body digest mismatch")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "raw15", raw15)

    @classmethod
    def create(
        cls,
        lease: dict,
        chunk_id: str,
        request_key: str,
        source_step: int,
        rgb: Any,
        raw15: Any,
        identity: dict,
        episode_deadline_ns: int | str,
        job_deadline_ns: int | str,
        clock_domain: str,
        replan_deadline_ns: int | str | None = None,
    ) -> Frame:
        public_lease = validate_lease(lease)
        raw_rgb = _owned_bytes(rgb, RGB_BYTES, "rgb")
        raw_action = validate_raw15(raw15)
        header = {
            "version": VERSION,
            "policy_epoch": public_lease["server_epoch"],
            "lease": public_lease,
            "issued_chunk_id": chunk_id,
            "policy_request_key": request_key,
            "source_step": source_step,
            "source_rgb_sha256": source_digest(raw_rgb),
            "source_rgb_contract": RGB_CONTRACT,
            "rgb_shape": list(RGB_SHAPE),
            "rgb_layout": RGB_LAYOUT,
            "raw15_sha256": hashlib.sha256(raw_action).hexdigest(),
            "clock_domain": clock_domain,
            "episode_deadline_ns": _ns(episode_deadline_ns, "episode_deadline_ns", input_value=True),
            "job_deadline_ns": _ns(job_deadline_ns, "job_deadline_ns", input_value=True),
            "replan_deadline_ns": (
                None if replan_deadline_ns is None else _ns(replan_deadline_ns, "replan_deadline_ns", input_value=True)
            ),
            **validate_identity(identity),
        }
        return cls(canonical_json(_header(header, feature=False)), raw_rgb, raw_action)

    @property
    def header(self) -> dict:
        return strict_json(self.header_json, max_bytes=MAX_HEADER_BYTES)

    @property
    def fingerprint(self) -> str:
        parts = (self.header_json, self.rgb, self.raw15)
        return hashlib.sha256(
            _FRAME_DOMAIN + b"".join(struct.pack(">I", len(part)) + part for part in parts)
        ).hexdigest()

    def to_wire(self) -> dict:
        return {
            "header": self.header,
            "rgb_b64": base64.b64encode(self.rgb).decode("ascii"),
            "raw15_f32le_b64": base64.b64encode(self.raw15).decode("ascii"),
        }

    @classmethod
    def from_wire(cls, value: dict | bytes) -> Frame:
        if type(value) is bytes:
            value = strict_json(value, max_bytes=MAX_FRAME_WIRE_BYTES)
        value = _object(value, _FRAME_WIRE_FIELDS, "frame wire")
        header = _header(value["header"], feature=False)
        return cls(
            canonical_json(header),
            _decode_b64(value["rgb_b64"], RGB_BYTES, "rgb"),
            _decode_b64(value["raw15_f32le_b64"], RAW15_BYTES, "raw15"),
        )


def _key(value: Any) -> bytes:
    return _owned_bytes(value, 32, "HMAC key")


def validate_frame(frame: Frame) -> Frame:
    """Revalidate and detach a Frame at a trust seam; performs no clock refresh/check."""
    if type(frame) is not Frame:
        raise ProtocolError("Frame required")
    return Frame(frame.header_json, frame.rgb, frame.raw15)


def _mac(header: dict, raw15: bytes, visual96: bytes, key: bytes) -> str:
    encoded = canonical_json(header)
    message = _MAC_DOMAIN + struct.pack(">I", len(encoded)) + encoded + raw15 + visual96
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def sign_feature(
    frame: Frame,
    visual96: Any,
    key: Any,
    *,
    worker_id: str,
    worker_epoch: str,
    key_id: str,
) -> dict:
    """Sign an immutable input snapshot and finite encoded bytes using an enrolled key."""
    frame = validate_frame(frame)
    raw_visual = validate_visual96(visual96)
    header = _header(
        {
            **frame.header,
            "worker_id": worker_id,
            "worker_epoch": worker_epoch,
            "key_id": key_id,
            "visual96_sha256": hashlib.sha256(raw_visual).hexdigest(),
        },
        feature=True,
    )
    return {
        "header": header,
        "raw15_f32le_b64": base64.b64encode(frame.raw15).decode("ascii"),
        "visual96_f32le_b64": base64.b64encode(raw_visual).decode("ascii"),
        "mac_hex": _mac(header, frame.raw15, raw_visual, _key(key)),
    }


def verify_feature(
    receipt: dict | bytes,
    key: Any,
    *,
    expected_lease: dict | None = None,
    expected_identity: dict | None = None,
    expected_worker_id: str | None = None,
    expected_worker_epoch: str | None = None,
    expected_key_id: str | None = None,
    expected_clock_domain: str | None = None,
    now_ns: int | None = None,
) -> dict:
    """Validate schema/body/MAC and optional authoritative expectations; return a copy.

    Supplying the current enrolled epoch/lease is required at an admission seam;
    this stateless utility cannot detect revocation or a future causal source step.
    When supplied, now_ns must use the expected attested shared clock domain.
    """
    if type(receipt) is bytes:
        receipt = strict_json(receipt, max_bytes=MAX_FEATURE_WIRE_BYTES)
    receipt = _object(receipt, _RECEIPT_FIELDS, "feature receipt")
    header = _header(receipt["header"], feature=True)
    raw15 = validate_raw15(_decode_b64(receipt["raw15_f32le_b64"], RAW15_BYTES, "raw15"))
    visual96 = validate_visual96(_decode_b64(receipt["visual96_f32le_b64"], VISUAL96_BYTES, "visual96"))
    if hashlib.sha256(raw15).hexdigest() != header["raw15_sha256"]:
        raise ProtocolError("raw15 digest mismatch")
    if hashlib.sha256(visual96).hexdigest() != header["visual96_sha256"]:
        raise ProtocolError("visual96 digest mismatch")
    supplied_mac = _digest(receipt["mac_hex"], "mac_hex")
    if not hmac.compare_digest(supplied_mac, _mac(header, raw15, visual96, _key(key))):
        raise ProtocolError("feature MAC mismatch")
    if expected_lease is not None and header["lease"] != validate_lease(expected_lease):
        raise ProtocolError("authoritative lease mismatch")
    if expected_identity is not None:
        expected = validate_identity(expected_identity)
        if any(header[name] != expected[name] for name in IDENTITY_FIELDS):
            raise ProtocolError("attested encoder identity mismatch")
    for name, expected in (
        ("worker_id", expected_worker_id),
        ("worker_epoch", expected_worker_epoch),
        ("key_id", expected_key_id),
        ("clock_domain", expected_clock_domain),
    ):
        if expected is not None and header[name] != _identifier(expected, name):
            raise ProtocolError(f"authoritative {name} mismatch")
    if now_ns is not None:
        now = _nonnegative(now_ns, "now_ns")
        if expected_clock_domain is None:
            raise ProtocolError("expiry check requires expected attested clock_domain")
        deadlines = [int(header["episode_deadline_ns"]), int(header["job_deadline_ns"])]
        if header["replan_deadline_ns"] is not None:
            deadlines.append(int(header["replan_deadline_ns"]))
        if now >= min(deadlines):
            raise ProtocolError("original feature deadline expired")
    return {
        "header": header,
        "raw15_f32le_b64": base64.b64encode(raw15).decode("ascii"),
        "visual96_f32le_b64": base64.b64encode(visual96).decode("ascii"),
        "mac_hex": supplied_mac,
    }
