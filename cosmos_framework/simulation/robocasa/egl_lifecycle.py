"""Pinned V5 admission and receipts for the native EvalV2 worker.

The core is immutable. This module owns only startup/cleanup admission and
evidence. It never changes policy inputs, sampling, batching or Local state.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import threading
from dataclasses import asdict
from pathlib import Path

CORE_SHA256 = "d57e592c49712559f655247e0bacc66d0f57b05b4b247cfd122b77d9a6941a06"
CANDIDATE_ID = "robosuite_egl_owned_lifecycle_v5"
INTEGRATION_VERSION = "v5-production-v1"
SCHEMA = "robosuite_egl_lifecycle_receipt_v1"
CORE_PATH = Path(__file__).with_name("egl_lifecycle_v5.py")
VENDOR_PINS = {
    "binding": "c8463cddb558f22247b1c8f202443ddeabe1469e5433b5b9d0eb7b938f32f480",
    "egl": "72ff1e153da85b85a35ebd0350c4693d3b89ffd5e1f13ff49813b7a3ad4c51c6",
    "env": "da6a543615c667dbdbd972e0e1af9ab944f7209053bc6cadba5138ce3f878ad6",
}
COUNTERS = {"created", "destroyed", "mjr_freed", "explicit_retire", "fallback_close"}


def expected_binding(cfg, pid):
    """Coordinator-owned identity; CPU_TEST_FACTORY is an existing Python test boundary."""
    sources = cfg.get("source_files", {})
    return {
        "schema": SCHEMA,
        "integration_version": INTEGRATION_VERSION,
        "candidate_id": CANDIDATE_ID,
        "core_sha256": CORE_SHA256,
        "adapter_sha256": sources.get("egl_lifecycle", {}).get(
            "sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
        "pid": pid,
        "nonce": cfg["nonce"],
        "physical_slot_id": cfg["physical_slot_id"],
        "manifest_digest": cfg["manifest_digest"],
        "backend": cfg["rendering"]["backend"],
        "test_factory": cfg.get("runtime_args", {}).get("expected_runtime_digest") == "CPU_TEST_FACTORY",
    }


def validate_receipt(receipt, expected, phase, startup=None):
    """Fail closed on absent, foreign, stale, or incomplete worker receipts."""
    if phase not in ("startup", "close") or not isinstance(receipt, dict):
        raise ValueError("missing/invalid EGL lifecycle receipt")
    if any(receipt.get(key) != value for key, value in expected.items()) or receipt.get("phase") != phase:
        raise ValueError("EGL lifecycle receipt identity mismatch")
    if receipt.get("core_sha256") != CORE_SHA256 or receipt.get("errors") != []:
        raise ValueError("EGL lifecycle receipt core/failure mismatch")
    owner = receipt.get("owner_thread")
    if type(owner) is not int or owner <= 0 or receipt.get("receipt_thread") != owner:
        raise ValueError("EGL lifecycle receipt thread mismatch")
    if startup is not None:
        for key in ("owner_thread", "vendor", "reset_entry"):
            if receipt.get(key) != startup.get(key):
                raise ValueError("EGL lifecycle startup/close binding changed")
    if expected["test_factory"]:
        status = "CPU_TEST_FACTORY"
    elif expected["backend"] == "osmesa":
        status = "NOT_APPLICABLE"
    elif expected["backend"] == "egl":
        status = "INSTALLED" if phase == "startup" else "QUIESCENT"
    else:
        raise ValueError("unsupported EGL lifecycle backend")
    if receipt.get("status") != status:
        raise ValueError("EGL lifecycle receipt status mismatch")
    if status in ("CPU_TEST_FACTORY", "NOT_APPLICABLE"):
        if receipt.get("audit") is not None or receipt.get("used") is not False or receipt.get("vendor") != {}:
            raise ValueError("non-EGL/CPU receipt cannot claim native lifecycle usage")
        return receipt
    vendor = receipt.get("vendor")
    if not isinstance(vendor, dict) or set(vendor) != set(VENDOR_PINS):
        raise ValueError("incomplete EGL vendor witnesses")
    for name, digest in VENDOR_PINS.items():
        item = vendor[name]
        if (
            not isinstance(item, dict)
            or item.get("sha256") != digest
            or not isinstance(item.get("path"), str)
            or not Path(item["path"]).is_absolute()
        ):
            raise ValueError("EGL lifecycle vendor witness mismatch")
    audit = receipt.get("audit")
    if not isinstance(audit, dict) or audit.get("id") != CANDIDATE_ID:
        raise ValueError("missing V5 lifecycle audit")
    counts = audit.get("counts")
    if not isinstance(counts, dict) or not COUNTERS <= set(counts) or set(counts) - COUNTERS - {"resize_freed"}:
        raise ValueError("incomplete V5 lifecycle counters")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("invalid V5 lifecycle counters")
    if (
        any(type(audit.get(key)) is not int or audit[key] != 0 for key in ("live_egl", "live_render", "quarantined"))
        or audit.get("failures") != []
        or counts["created"] != counts["destroyed"]
        or counts["fallback_close"] != 0
        or (phase == "startup" and any(counts.values()))
    ):
        raise ValueError("V5 lifecycle is not explicitly quiescent")
    binding = receipt.get("current_binding")
    if not isinstance(binding, dict) or set(binding) != {"display", "draw", "read", "context", "api"}:
        raise ValueError("missing actual EGL binding")
    if any(type(value) is not int or value < 0 for value in binding.values()) or binding["context"] != 0:
        raise ValueError("EGL context still current at lifecycle boundary")
    if type(receipt.get("used")) is not bool or receipt["used"] != (counts["created"] > 0):
        raise ValueError("incorrect V5 lifecycle usage claim")
    return receipt


def _load_core():
    if hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA256:
        raise RuntimeError("unqualified V5 core bytes")
    core = importlib.import_module("cosmos_framework.simulation.robocasa.egl_lifecycle_v5")
    origin = getattr(getattr(core, "__spec__", None), "origin", None)
    loaded = getattr(core, "__file__", None)
    if (
        not isinstance(origin, str)
        or not isinstance(loaded, str)
        or Path(origin).resolve() != CORE_PATH.resolve()
        or Path(loaded).resolve() != CORE_PATH.resolve()
        or hashlib.sha256(Path(loaded).read_bytes()).hexdigest() != CORE_SHA256
    ):
        raise RuntimeError("foreign V5 core import origin")
    return core


def _vendor_admission():
    from robosuite.environments import base
    from robosuite.renderers.context import egl_context
    from robosuite.utils import binding_utils

    modules = {"binding": binding_utils, "egl": egl_context, "env": base}
    witnesses = {
        name: {
            "path": str(Path(module.__file__).resolve()),
            "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        }
        for name, module in modules.items()
    }
    if any(witnesses[name]["sha256"] != digest for name, digest in VENDOR_PINS.items()):
        raise RuntimeError("unqualified V5 vendor bytes")
    # A pre-install stock reset observer is allowed. Native destruction observers
    # belong after install; in particular the original Guard must never be stacked.
    for module, cls, name in (
        (binding_utils, binding_utils.MjRenderContext, "__del__"),
        (egl_context, egl_context.EGLGLContext, "free"),
    ):
        method = getattr(cls, name)
        code = getattr(method, "__code__", None)
        if (
            getattr(method, "_p01_context_guard", False)
            or code is None
            or Path(code.co_filename).resolve() != Path(module.__file__).resolve()
        ):
            raise RuntimeError("pre-existing EGL/Guard patch; install V5 in a fresh worker")
    reset = base.MujocoEnv.reset
    code = getattr(reset, "__code__", None)
    reset_path = Path(code.co_filename).resolve() if code is not None else None
    reset_entry = {
        "module": reset.__module__,
        "qualname": reset.__qualname__,
        "path": str(reset_path) if reset_path is not None else None,
        "sha256": hashlib.sha256(reset_path.read_bytes()).hexdigest()
        if reset_path is not None and reset_path.is_file()
        else None,
    }
    return witnesses, reset_entry


class WorkerLifecycle:
    def __init__(self, cfg, *, test_factory=False):
        self.cfg = cfg
        self.pid, self.thread = os.getpid(), threading.current_thread()
        self.identity = expected_binding(cfg, self.pid)
        self.identity["test_factory"] = bool(test_factory)
        self.life = None
        self.vendor, self.reset_entry = {}, None
        self.errors = []
        self.finished = None
        core = _load_core()
        backend = self.identity["backend"]
        if backend not in ("egl", "osmesa"):
            raise ValueError("unsupported worker rendering backend")
        if not test_factory and backend == "egl":
            self.vendor, self.reset_entry = _vendor_admission()
            if core.ID != CANDIDATE_ID or core.PINS != VENDOR_PINS:
                raise RuntimeError("V5 core contract changed")
            life = core.install()
            if type(life) is not core.Lifecycle or core._INSTALLED is not life or not life.installed:
                raise RuntimeError("V5 install did not return its installed Lifecycle singleton")
            self.life = life
            life.assert_quiescent()
        self.startup_receipt = self._receipt("startup")
        validate_receipt(self.startup_receipt, self.identity, "startup")
        self._publish(self.startup_receipt)

    def _owner(self):
        if os.getpid() != self.pid or threading.current_thread() is not self.thread:
            raise RuntimeError("V5 worker lifecycle used from another process/thread")

    def _receipt(self, phase):
        self._owner()
        audit = self.life.audit() if self.life is not None else None
        binding = asdict(self.life.driver.snapshot()) if self.life is not None else None
        status = (
            "CPU_TEST_FACTORY"
            if self.identity["test_factory"]
            else "NOT_APPLICABLE"
            if self.identity["backend"] == "osmesa"
            else "INSTALLED"
            if phase == "startup"
            else "QUIESCENT"
        )
        return {
            **self.identity,
            "phase": phase,
            "status": "FAILED" if self.errors else status,
            "owner_thread": self.thread.ident,
            "receipt_thread": threading.get_ident(),
            "vendor": self.vendor,
            "reset_entry": self.reset_entry,
            "audit": audit,
            "current_binding": binding,
            "used": bool(audit and audit["counts"]["created"]),
            "errors": list(self.errors),
        }

    def _publish(self, receipt):
        from cosmos_framework.evaluation.robocasa_v2.store import write

        nonce = self.identity["nonce"]
        if not isinstance(nonce, str) or not nonce or not all(c.isalnum() or c in "_-" for c in nonce):
            raise ValueError("invalid worker lifecycle receipt nonce")
        path = Path(self.cfg["task_root"]).parent / "egl_lifecycle_workers" / f"{nonce}_{receipt['phase']}.json"
        write(path, receipt, replace=receipt["phase"] == "failure")

    def failure(self, error):
        """Record failure without retrying any uncertain native release."""
        self._owner()
        self.errors.append(f"{type(error).__name__}: {error}")
        self._publish(self._receipt("failure"))

    def finish(self):
        self._owner()
        if self.finished is not None:
            return self.finished
        if self.life is not None:
            self.life.assert_quiescent()
        value = self._receipt("close")
        if not self.errors:
            validate_receipt(value, self.identity, "close", self.startup_receipt)
        self._publish(value)
        self.finished = value
        return value


def start(cfg, *, test_factory=False):
    """Call after renderer/vendor verification and before the first environment."""
    return WorkerLifecycle(cfg, test_factory=test_factory)
