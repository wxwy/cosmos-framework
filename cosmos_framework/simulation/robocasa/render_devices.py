"""Select MuJoCo EGL devices without requiring vendor DRM identity strings.

Prefer verifiable PCI identity. If the driver exposes no identity for the
requested device, use the explicit EGL index and report that its physical
mapping is unverified. No server-specific PCI/address mapping is persisted.
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
        try:
            self.query_display = self._function(
                b"eglQueryDisplayAttribEXT", ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
            )
        except RuntimeError:
            # Some EGL stacks can create a Context without exposing display-device identity.
            self.query_display = None

    def _function(self, name, result, *arguments):
        address = self.lib.eglGetProcAddress(name)
        if not address:
            raise RuntimeError(f"physical EGL identity extension missing: {name.decode()}")
        return ctypes.CFUNCTYPE(result, *arguments)(address)

    def _describe(self, device):
        # Optional EGL_EXT_device_drm / EGL_EXT_device_drm_render_node identity.
        # A missing extension is not evidence of physical GPU placement.
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
        raise RuntimeError("EGL physical PCI identity unavailable")

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
                pci = node = None  # Unknown physical identity; ordinal remains selectable.
            result.append({"egl_device_index": index, "handle": device, "pci_bus_id": pci, "device_node": node})
        return result

    def current_handle(self):
        """Require an active Context; return its EGL Device handle when supported."""
        display = self.lib.eglGetCurrentDisplay()
        if not display or not self.lib.eglGetCurrentContext():
            raise RuntimeError("no current EGL context after environment reset")
        if self.query_display is None:
            return None
        device = ctypes.c_ssize_t()
        if not self.query_display(display, 0x322C, ctypes.byref(device)):  # EGL_DEVICE_EXT
            self.lib.eglGetError()
            return None
        return device.value

    def current(self):
        handle = self.current_handle()
        if handle is None:
            raise RuntimeError("cannot identify the actual EGL display device")
        pci, node = self._describe(handle)
        return {"handle": handle, "pci_bus_id": pci, "device_node": node}


def validate_proof(proof, plan, *, context=False):
    if not isinstance(proof, dict) or any(
        proof.get(k) != plan[k] for k in ("backend", "server_gpu_id", "render_gpu_id", "expected_device")
    ):
        raise ValueError("render device attestation differs from assigned roles")
    if plan["backend"] == "egl":
        index = proof.get("egl_device_index")
        if type(index) is not int or index < 0 or (context and proof.get("context_verified") is not True):
            raise ValueError("invalid EGL device/context proof")
        if proof.get("protocol") == "egl_physical_pci_v1":
            if proof.get("physical_identity_verified") is not True or proof.get("actual_device") != plan["expected_device"]:
                raise ValueError("unproven physical EGL device")
        elif proof.get("protocol") == "egl_index_unverified_v1":
            # Be honest: an EGL ordinal is NOT evidence of a physical GPU mapping.
            if (
                proof.get("physical_identity_verified") is not False
                or proof.get("actual_device") is not None
                or index != plan["render_gpu_id"]
                or proof.get("device_node") is not None
            ):
                raise ValueError("invalid unverified EGL ordinal selection")
        else:
            raise ValueError("unsupported EGL device proof protocol")
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
    if len(matches) > 1:
        raise RuntimeError("ambiguous physical EGL device identity")
    if matches:
        selected, verified = matches[0], True
    else:
        # Standard MuJoCo behavior when NVIDIA does not expose EGL DRM strings.
        # This is a configurable EGL ordinal, NOT a verified physical GPU binding.
        index = plan["render_gpu_id"]
        selected = next((v for v in devices if v["egl_device_index"] == index), None)
        if selected is None:
            raise RuntimeError("configured EGL device index is not enumerated")
        if selected["pci_bus_id"] is not None:
            raise RuntimeError("explicit EGL index conflicts with known physical PCI identity")
        verified = False
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected["egl_device_index"])
    proof = dict(
        plan,
        protocol="egl_physical_pci_v1" if verified else "egl_index_unverified_v1",
        actual_device=dict(expected) if verified else None,
        egl_device_index=selected["egl_device_index"],
        device_node=selected["device_node"] if verified else None,
        physical_identity_verified=verified,
        context_verified=False,
    )
    return validate_proof(proof, plan)


def verify_current(proof, *, api=None):
    if proof["backend"] != "egl":
        return proof
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(proof["egl_device_index"]):
        raise RuntimeError("render environment changed after EGL device selection")
    driver = api or EGLDevices()
    if proof["protocol"] == "egl_index_unverified_v1":
        # Check the current Context exists. If the vendor exposes a device handle,
        # also check it is the selected EGL device; never invent a PCI mapping.
        handle = driver.current_handle()
        if handle is not None:
            selected = [v for v in driver.inventory() if v["egl_device_index"] == proof["egl_device_index"]]
            if len(selected) != 1 or handle != selected[0]["handle"]:
                raise RuntimeError("current EGL Context is on a different enumerated device")
        return validate_proof(
            dict(proof, context_verified=True, context_device_handle_verified=handle is not None),
            proof,
            context=True,
        )
    current = driver.current()
    selected = [v for v in driver.inventory() if v["egl_device_index"] == proof["egl_device_index"]]
    if (
        len(selected) != 1
        or current["handle"] != selected[0]["handle"]
        or current["pci_bus_id"] != proof["expected_device"]["pci_bus_id"]
    ):
        raise RuntimeError("actual EGL context uses a different physical GPU")
    return validate_proof(dict(proof, context_verified=True), proof, context=True)
