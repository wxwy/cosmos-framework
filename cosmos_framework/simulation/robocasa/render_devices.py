"""Resolve physical EGL devices by PCI identity, never by CUDA ordinal equality.

Import is CPU-only. EGL is loaded only in a real EGL worker, before vendor imports.
No context is initialized on any device while enumerating its identity.
"""

from __future__ import annotations

import ctypes
import os
import re
from pathlib import Path


def normalize_pci(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value):
        raise ValueError("invalid physical PCI bus identity")
    domain, bus, slot_function = value.lower().split(":")
    if int(domain, 16) > 65535:
        raise ValueError("unsupported PCI domain")
    return f"{int(domain, 16):04x}:{bus}:{slot_function}"


class EGLDevices:
    def __init__(self):
        self.lib = ctypes.CDLL("libEGL.so.1")
        self.lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        self.lib.eglGetProcAddress.restype = ctypes.c_void_p
        self.lib.eglGetCurrentDisplay.restype = ctypes.c_void_p
        self.lib.eglGetCurrentContext.restype = ctypes.c_void_p
        self.lib.eglGetError.restype = ctypes.c_uint
        self.query_devices = self._function(
            b"eglQueryDevicesEXT",
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        )
        self.query_string = self._function(b"eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
        self.query_display = self._function(
            b"eglQueryDisplayAttribEXT", ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
        )

    def _function(self, name, result, *arguments):
        address = self.lib.eglGetProcAddress(name)
        if not address:
            raise RuntimeError(f"physical EGL identity extension missing: {name.decode()}")
        return ctypes.CFUNCTYPE(result, *arguments)(address)

    def _describe(self, device):
        # EGL_EXT_device_drm / EGL_EXT_device_drm_render_node. A missing DRM
        # identity is BLOCKED, never an excuse to assume ordinal N == GPU N.
        for attribute in (0x3377, 0x3233):
            raw = self.query_string(device, attribute)
            self.lib.eglGetError()  # Do not leave an unsupported-attribute error for the vendor.
            if raw:
                node = Path(raw.decode("utf-8", errors="strict"))
                if node.parent != Path("/dev/dri"):
                    continue
                try:
                    pci = (Path("/sys/class/drm") / node.name / "device").resolve(strict=True).name
                    return normalize_pci(pci), str(node)
                except (OSError, ValueError):
                    continue
        raise RuntimeError("EGL physical PCI identity unavailable; no ordinal fallback")

    def inventory(self):
        count = ctypes.c_int()
        if not self.query_devices(0, None, ctypes.byref(count)) or not 0 < count.value <= 256:
            raise RuntimeError("bounded EGL device enumeration failed")
        devices = (ctypes.c_void_p * count.value)()
        capacity = count.value
        if not self.query_devices(capacity, devices, ctypes.byref(count)) or not 0 < count.value <= capacity:
            raise RuntimeError("EGL devices changed during enumeration")
        result = []
        for index, device in enumerate(devices[: count.value]):
            try:
                pci, node = self._describe(device)
            except RuntimeError:
                pci = node = None  # Software/unidentifiable devices cannot be selected.
            result.append({"egl_device_index": index, "handle": device, "pci_bus_id": pci, "device_node": node})
        return result

    def current(self):
        display = self.lib.eglGetCurrentDisplay()
        if not display or not self.lib.eglGetCurrentContext():
            raise RuntimeError("no current EGL context after environment reset")
        device = ctypes.c_ssize_t()
        if not self.query_display(display, 0x322C, ctypes.byref(device)):  # EGL_DEVICE_EXT
            raise RuntimeError("cannot identify the actual EGL display device")
        pci, node = self._describe(device.value)
        return {"handle": device.value, "pci_bus_id": pci, "device_node": node}


def validate_proof(proof, plan, *, context=False):
    if not isinstance(proof, dict) or any(
        proof.get(k) != plan[k] for k in ("backend", "server_gpu_id", "render_gpu_id", "expected_device")
    ):
        raise ValueError("render device attestation differs from assigned roles")
    if plan["backend"] == "egl":
        expected = plan["expected_device"]
        actual = proof.get("actual_device", {})
        if (
            proof.get("protocol") != "egl_physical_pci_v1"
            or proof.get("physical_identity_verified") is not True
            or type(proof.get("egl_device_index")) is not int
            or proof["egl_device_index"] < 0
            or actual != expected
            or (context and proof.get("context_verified") is not True)
        ):
            raise ValueError("unproven physical EGL device/context")
    elif (
        plan["backend"] != "osmesa"
        or proof.get("protocol") != "cpu_osmesa_v1"
        or proof.get("actual_device") is not None
    ):
        raise ValueError("unproven CPU render backend")
    return proof


def configure(plan, *, api=None):
    backend = plan["backend"]
    if backend not in ("egl", "osmesa"):
        raise ValueError("unsupported render backend")
    os.environ.update(MUJOCO_GL=backend, PYOPENGL_PLATFORM=backend, CUDA_VISIBLE_DEVICES="")
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    if backend == "osmesa":
        if plan["render_gpu_id"] is not None or plan["expected_device"] is not None:
            raise ValueError("OSMesa must not allocate a GPU")
        return dict(plan, protocol="cpu_osmesa_v1", actual_device=None, context_verified=False)
    expected = plan["expected_device"]
    if expected["gpu_id"] != plan["render_gpu_id"] or normalize_pci(expected["pci_bus_id"]) != expected["pci_bus_id"]:
        raise ValueError("physical render assignment identity mismatch")
    devices = (api or EGLDevices()).inventory()
    matches = [v for v in devices if v["pci_bus_id"] == expected["pci_bus_id"]]
    if len(matches) != 1:
        raise RuntimeError("selected physical render GPU must have one provable EGL device")
    selected = matches[0]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected["egl_device_index"])
    proof = dict(
        plan,
        protocol="egl_physical_pci_v1",
        actual_device=dict(expected),
        egl_device_index=selected["egl_device_index"],
        device_node=selected["device_node"],
        physical_identity_verified=True,
        context_verified=False,
    )
    return validate_proof(proof, plan)


def verify_current(proof, *, api=None):
    if proof["backend"] != "egl":
        return proof
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(proof["egl_device_index"]):
        raise RuntimeError("render environment changed after physical device attestation")
    driver = api or EGLDevices()
    current = driver.current()
    selected = [v for v in driver.inventory() if v["egl_device_index"] == proof["egl_device_index"]]
    if (
        len(selected) != 1
        or current["handle"] != selected[0]["handle"]
        or current["pci_bus_id"] != proof["expected_device"]["pci_bus_id"]
    ):
        raise RuntimeError("actual EGL context uses a different physical GPU")
    return validate_proof(dict(proof, context_verified=True), proof, context=True)
