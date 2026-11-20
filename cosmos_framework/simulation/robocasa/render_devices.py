"""Resolve a physical EGL render device without assuming CUDA/EGL index equality.

DRM provides a PCI identity directly. NVIDIA EGL can instead expose a
persistent device UUID while the Actor masks CUDA. An isolated driver probe
joins that UUID to PCI using EGL and CUDA in the same subprocess. The Actor
selects its own enumerated device and verifies its actual context handle.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def normalize_pci(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value):
        raise ValueError("invalid physical PCI bus identity")
    domain, bus, slot_function = value.lower().split(":")
    if int(domain, 16) > 65535:
        raise ValueError("unsupported PCI domain")
    return f"{int(domain, 16):04x}:{bus}:{slot_function}"


# Only a short-lived subprocess runs these queries with CUDA visible. The
# driver enumeration has no context creation or model imports.
_CUDA_PCI_PROBE = "--probe-cuda-pci"
_EGL_PCI_PROBE = "--probe-egl-cuda-pci"


def _cuda_driver_pci():
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cuDeviceGetCount.restype = ctypes.c_int
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDeviceGet.restype = ctypes.c_int
    lib.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.cuDeviceGetPCIBusId.restype = ctypes.c_int
    if lib.cuInit(0) != 0:
        raise RuntimeError("CUDA driver initialization failed")
    count = ctypes.c_int()
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0 or not 0 < count.value <= 256:
        raise RuntimeError("CUDA device enumeration failed")
    identity = {}
    for ordinal in range(count.value):
        device = ctypes.c_int()
        bus = ctypes.create_string_buffer(64)
        if lib.cuDeviceGet(ctypes.byref(device), ordinal) != 0 or device.value in identity:
            raise RuntimeError("invalid or repeated CUDA device handle")
        if lib.cuDeviceGetPCIBusId(bus, len(bus), device.value) != 0:
            raise RuntimeError("CUDA PCI identity lookup failed")
        identity[device.value] = normalize_pci(bus.value.decode("ascii"))
    return identity


def _cuda_handle_key(key):
    ordinal = int(key)
    if str(ordinal) != key or not 0 <= ordinal < 256:
        raise ValueError("invalid CUDA device handle")
    return ordinal


def _egl_uuid_key(key):
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}", key) or key == "0" * 32:
        raise ValueError("invalid EGL persistent device UUID")
    return key


def _driver_pci_probe(mode, parse_key, *, run):
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        result = run(
            [sys.executable, "-I", str(Path(__file__).resolve()), mode],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        raw = json.loads(result.stdout)
        if not isinstance(raw, dict) or not 0 < len(raw) <= 256:
            raise ValueError("invalid CUDA PCI inventory")
        mapping = {parse_key(key): normalize_pci(value) for key, value in raw.items()}
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("ambiguous CUDA PCI identity")
        return mapping
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        name = "CUDA" if mode == _CUDA_PCI_PROBE else "EGL/CUDA"
        raise RuntimeError(f"{name} physical PCI identity unavailable") from error


def cuda_pci_map(*, run=subprocess.run):
    """Map same-process CUDA handles to PCI for driver diagnostics."""
    return _driver_pci_probe(_CUDA_PCI_PROBE, _cuda_handle_key, run=run)


def egl_cuda_pci_map(*, run=subprocess.run):
    """Probe persistent EGL UUID -> physical PCI without exposing CUDA in the Actor."""
    return _driver_pci_probe(_EGL_PCI_PROBE, _egl_uuid_key, run=run)


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
        self.query_string = self._optional(b"eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
        self.query_cuda = self._optional(
            b"eglQueryDeviceAttribEXT",
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
        )
        self.query_binary = self._optional(
            b"eglQueryDeviceBinaryEXT",
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        self.query_display = self._optional(
            b"eglQueryDisplayAttribEXT",
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
        )

    def _function(self, name, result, *arguments):
        address = self.lib.eglGetProcAddress(name)
        if not address:
            raise RuntimeError(f"physical EGL identity extension missing: {name.decode()}")
        return ctypes.CFUNCTYPE(result, *arguments)(address)

    def _optional(self, name, result, *arguments):
        try:
            return self._function(name, result, *arguments)
        except RuntimeError:
            return None

    def _cuda_device(self, device):
        # EGL_NV_device_cuda accepts an attribute, not an EGL device string.
        if self.query_cuda is None:
            return None
        value = ctypes.c_ssize_t(-1)
        if not self.query_cuda(device, 0x323A, ctypes.byref(value)):
            self.lib.eglGetError()
            return None
        if not 0 <= value.value < 256:
            raise RuntimeError("invalid NVIDIA EGL CUDA handle")
        return int(value.value)

    def _nvidia_uuid(self, device):
        if self.query_binary is None or self.query_string is None:
            return None
        raw = self.query_string(device, 0x3055)  # EGL_EXTENSIONS
        self.lib.eglGetError()
        extensions = set((raw or b"").split())
        if not {b"EGL_NV_device_cuda", b"EGL_EXT_device_persistent_id"} <= extensions:
            return None
        data, size = (ctypes.c_ubyte * 16)(), ctypes.c_int()
        if not self.query_binary(device, 0x335C, len(data), data, ctypes.byref(size)):  # EGL_DEVICE_UUID_EXT
            self.lib.eglGetError()
            return None
        if size.value != len(data):
            raise RuntimeError("invalid NVIDIA EGL persistent device UUID size")
        return _egl_uuid_key(bytes(data).hex())

    def _describe(self, device):
        # EGL_EXT_device_drm / EGL_EXT_device_drm_render_node, if provided.
        if self.query_string is None:
            raise RuntimeError("EGL DRM identity extension unavailable")
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
        raise RuntimeError("EGL DRM PCI identity unavailable")

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
            result.append(
                {
                    "egl_device_index": index,
                    "handle": device,
                    "pci_bus_id": pci,
                    "device_node": node,
                    "drm_accessible": os.access(node, os.R_OK | os.W_OK) if node else None,
                    "cuda_device": self._cuda_device(device),
                    "egl_device_uuid": self._nvidia_uuid(device),
                }
            )
        return result

    def current(self):
        display = self.lib.eglGetCurrentDisplay()
        if not display or not self.lib.eglGetCurrentContext():
            raise RuntimeError("no current EGL context after environment reset")
        if self.query_display is None:
            return {"handle": None, "pci_bus_id": None, "cuda_device": None}
        device = ctypes.c_ssize_t()
        if not self.query_display(display, 0x322C, ctypes.byref(device)):  # EGL_DEVICE_EXT
            self.lib.eglGetError()
            return {"handle": None, "pci_bus_id": None, "cuda_device": None}
        try:
            pci, _ = self._describe(device.value)
        except RuntimeError:
            pci = None
        return {
            "handle": device.value,
            "pci_bus_id": pci,
            "cuda_device": self._cuda_device(device.value),
            "egl_device_uuid": self._nvidia_uuid(device.value),
        }


def _egl_driver_pci():
    cuda = _cuda_driver_pci()
    mapping = {}
    for device in EGLDevices().inventory():
        uuid = device["egl_device_uuid"]
        if uuid is None:
            continue
        pci = cuda.get(device["cuda_device"])
        if pci is None or uuid in mapping or device["pci_bus_id"] not in (None, pci):
            raise RuntimeError("missing, duplicate, or conflicting EGL/CUDA PCI identity")
        mapping[uuid] = pci
    return mapping


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
            or type(proof.get("egl_device_handle")) is not int
            or proof["egl_device_handle"] <= 0
            or actual != expected
            or (context and (
                proof.get("context_verified") is not True
                or proof.get("context_device_handle_verified") is not True
            ))
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
    if any(v.get("egl_device_uuid") is not None for v in devices):
        uuid_pci = egl_cuda_pci_map()
        seen = set()
        for device in devices:
            uuid = device.get("egl_device_uuid")
            if uuid is None:
                continue
            pci = uuid_pci.get(uuid)
            if uuid in seen or pci is None:
                raise RuntimeError("EGL UUID has missing or ambiguous physical PCI identity")
            if device["pci_bus_id"] not in (None, pci):
                raise RuntimeError("EGL DRM and CUDA PCI identity conflict")
            seen.add(uuid)
            device["pci_bus_id"] = pci
    matches = [
        v for v in devices
        if v["pci_bus_id"] == expected["pci_bus_id"]
        and (v.get("drm_accessible") is not False or v.get("egl_device_uuid") is not None)
    ]
    if len(matches) != 1:
        raise RuntimeError("selected physical render GPU must have one provable EGL device")
    selected = matches[0]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected["egl_device_index"])
    proof = dict(
        plan,
        protocol="egl_physical_pci_v1",
        actual_device=dict(expected),
        egl_device_index=selected["egl_device_index"],
        egl_device_handle=selected["handle"],
        egl_device_uuid=selected.get("egl_device_uuid"),
        cuda_device=selected.get("cuda_device"),
        device_node=selected["device_node"],
        physical_identity_verified=True,
        context_verified=False,
    )
    return validate_proof(proof, plan)


def verify_current(proof, *, api=None):
    if proof["backend"] != "egl":
        return proof
    if (
        os.environ.get("MUJOCO_GL") != "egl"
        or os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(proof["egl_device_index"])
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise RuntimeError("render environment changed after physical device attestation")
    # No full device enumeration / CUDA PCI subprocess on every scene reset.
    driver = api or EGLDevices()
    current = driver.current()
    if not current.get("handle"):
        raise RuntimeError("current EGL context device handle unavailable")
    if current["handle"] != proof["egl_device_handle"]:
        raise RuntimeError("actual EGL context uses a different physical GPU")
    if proof.get("egl_device_uuid") is not None and current.get("egl_device_uuid") != proof["egl_device_uuid"]:
        raise RuntimeError("actual EGL context has a different persistent device UUID")
    if current.get("pci_bus_id") is not None and current["pci_bus_id"] != proof["expected_device"]["pci_bus_id"]:
        raise RuntimeError("actual EGL context has a different physical PCI identity")
    if (
        current.get("cuda_device") is not None
        and proof.get("cuda_device") is not None
        and current["cuda_device"] != proof["cuda_device"]
    ):
        raise RuntimeError("actual EGL context uses a different CUDA device")
    return validate_proof(
        dict(
            proof,
            context_verified=True,
            context_device_handle_verified=True,
        ),
        proof,
        context=True,
    )


if __name__ == "__main__":
    commands = {_CUDA_PCI_PROBE: _cuda_driver_pci, _EGL_PCI_PROBE: _egl_driver_pci}
    if len(sys.argv) != 2 or sys.argv[1] not in commands or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise SystemExit("expected an isolated driver PCI probe invocation")
    print(json.dumps(commands[sys.argv[1]]()))
