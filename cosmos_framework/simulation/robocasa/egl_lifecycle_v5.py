"""Pinned Robosuite offscreen EGL lifecycle candidate. Process-startup only.

Scope: Linux EGL, one owning Python thread per context, ordinary non-shared
OpenGL contexts. No GC scheduling fix, no CUDA/Policy modifications. Explicit
retirement before simulation replacement plus context-correct resource free,
with thread-affinity and fail-closed cleanup. Never calls eglReleaseThread.
"""
from __future__ import annotations
import atexit
import contextlib
import ctypes
import hashlib
import logging
from pathlib import Path
import threading
import weakref
from dataclasses import dataclass

ID = "robosuite_egl_owned_lifecycle_v5"
PINS = {
    "binding": "c8463cddb558f22247b1c8f202443ddeabe1469e5433b5b9d0eb7b938f32f480",
    "egl": "72ff1e153da85b85a35ebd0350c4693d3b89ffd5e1f13ff49813b7a3ad4c51c6",
    "env": "da6a543615c667dbdbd972e0e1af9ab944f7209053bc6cadba5138ce3f878ad6",
}
LOG = logging.getLogger(__name__)
OPENGL = 0x30A2
DRAW, READ = 0x3059, 0x305A

class LifecycleError(RuntimeError):
    pass

def addr(handle):
    if handle is None:
        return 0
    if isinstance(handle, int):
        return handle
    return ctypes.cast(handle, ctypes.c_void_p).value or 0

@dataclass(frozen=True)
class Binding:
    display: int
    draw: int
    read: int
    context: int
    api: int

class NativeEGL:
    def __init__(self, vendor):
        self.vendor = vendor
        self.lib = ctypes.CDLL("libEGL.so.1")
        specs = {
            "eglGetCurrentDisplay": ([], ctypes.c_void_p),
            "eglGetCurrentContext": ([], ctypes.c_void_p),
            "eglGetCurrentSurface": ([ctypes.c_int], ctypes.c_void_p),
            "eglQueryAPI": ([], ctypes.c_uint),
            "eglBindAPI": ([ctypes.c_uint], ctypes.c_uint),
            "eglMakeCurrent": ([ctypes.c_void_p] * 4, ctypes.c_uint),
            "eglDestroyContext": ([ctypes.c_void_p] * 2, ctypes.c_uint),
            "eglGetError": ([], ctypes.c_int),
        }
        for name, (args, result) in specs.items():
            fun = getattr(self.lib, name)
            fun.argtypes, fun.restype = args, result

    def check(self, ok, name):
        if not ok:
            raise LifecycleError(f"{name} failed EGL error=0x{self.lib.eglGetError():04x}")

    def snapshot(self):
        l = self.lib
        return Binding(addr(l.eglGetCurrentDisplay()), addr(l.eglGetCurrentSurface(DRAW)),
                       addr(l.eglGetCurrentSurface(READ)), addr(l.eglGetCurrentContext()), int(l.eglQueryAPI()))

    def bind_api(self, api):
        self.check(self.lib.eglBindAPI(api), "eglBindAPI")

    def bind(self, binding):
        self.bind_api(binding.api)
        self.check(self.lib.eglMakeCurrent(binding.display, binding.draw, binding.read, binding.context),
                   "eglMakeCurrent")
        if self.snapshot() != binding:
            raise LifecycleError("actual EGL binding differs from requested binding")

    def clear(self, display):
        if display:
            self.check(self.lib.eglMakeCurrent(display, 0, 0, 0), "eglMakeCurrent(no context)")

    def restore(self, saved):
        cur = self.snapshot()
        if cur == saved:
            return
        if saved.context:
            self.bind(saved)
        else:
            if cur.context:
                self.clear(cur.display)
            self.bind_api(saved.api)
            if self.snapshot() != saved:
                raise LifecycleError("failed to restore empty EGL binding/API")

    def create(self, device_id):
        v, E = self.vendor, self.vendor.EGL
        if v.EGL_DISPLAY is None:
            display = v.create_initialized_egl_device_display(device_id=device_id)
            if not addr(display):
                raise LifecycleError("cannot initialize EGL device display")
            v.EGL_DISPLAY = display
            atexit.register(E.eglTerminate, display)
        num = ctypes.c_long()
        config = E.EGLConfig()
        self.check(E.eglChooseConfig(v.EGL_DISPLAY, v.EGL_ATTRIBUTES, config, 1, num), "eglChooseConfig")
        if num.value < 1:
            raise LifecycleError("no EGL framebuffer config")
        self.bind_api(OPENGL)
        ctx = E.eglCreateContext(v.EGL_DISPLAY, config, E.EGL_NO_CONTEXT, None)
        if not addr(ctx):
            raise LifecycleError("eglCreateContext failed")
        return addr(v.EGL_DISPLAY), ctx

    def destroy(self, display, ctx):
        self.check(self.lib.eglDestroyContext(display, addr(ctx)), "eglDestroyContext")


class Lifecycle:
    """Internal driver injection also supports fault-injection CPU tests."""
    def __init__(self, binding, egl, env, driver):
        self.b, self.e, self.env, self.driver = binding, egl, env, driver
        self.lock = threading.RLock()
        self.originals = []
        self.gl_live = weakref.WeakSet()
        self.render_live = weakref.WeakSet()
        self.all_managed = weakref.WeakSet()
        self.quarantine = []
        self.failures = []
        self.events = []
        self.counts = dict(created=0, destroyed=0, mjr_freed=0, explicit_retire=0, fallback_close=0)
        self.device_id = None
        self.installed = False

    def emit(self, event, **fields):
        # bounded process-lifetime diagnostics; no unbounded frame records
        self.events.append(dict(event=event, thread=threading.get_ident(), **fields))
        if len(self.events) > 2048:
            del self.events[:1024]

    def fail(self, obj, exc):
        # Never drop a potentially live MjrContext following failed owner binding
        # or resource deletion. Require process quarantine/restart, not GC retry.
        with self.lock:
            if not any(x is obj for x in self.quarantine):
                self.quarantine.append(obj)
            self.failures.append(repr(exc))
            LOG.error("EGL lifecycle quarantined: %s", exc)

    def healthy(self):
        if self.failures:
            raise LifecycleError("renderer process quarantined: " + self.failures[-1])

    @staticmethod
    def thread(obj):
        if getattr(obj, "_egl_owner_thread", None) is not threading.current_thread():
            raise LifecycleError("EGL object used or destroyed from a non-owner thread")

    def snapshot(self):
        b = self.driver.snapshot()
        # Only OpenGL active contexts are qualified. No accidental GLES/VG switch.
        if b.context and b.api != OPENGL:
            raise LifecycleError("active non-OpenGL API is outside qualified lifecycle scope")
        return b

    def pin_binding(self, saved):
        # Keep a managed foreign active context and its render owner alive for
        # the duration of restoration, including allocations that trigger GC.
        for gl in self.gl_live:
            if addr(getattr(gl, "_context", None)) == saved.context and saved.context:
                ref = getattr(gl, "_egl_render_ref", None)
                return (gl, ref() if ref else None)
        return ()

    def restore_saved(self, saved, pin):
        # Nested retirement (e.g. resize fails and closes the current owner)
        # may legitimately destroy our saved managed context. Never restore
        # that now-invalid handle; foreign surviving bindings remain exact.
        if pin and saved.context and not addr(getattr(pin[0], "_context", None)):
            saved = Binding(0, 0, 0, 0, saved.api)
        self.driver.restore(saved)

    @contextlib.contextmanager
    def bound(self, gl):
        self.thread(gl)
        owner = addr(getattr(gl, "_context", None))
        if not owner:
            raise LifecycleError("EGL owner has no live context")
        saved = self.snapshot()
        pin = self.pin_binding(saved)
        target = Binding(gl._egl_display, 0, 0, owner, OPENGL)
        try:
            if saved != target:
                self.driver.bind(target)
            yield
        finally:
            try:
                self.restore_saved(saved, pin)
            except BaseException as exc:
                self.fail(gl, exc)
                raise

    def gl_init(self, obj, max_width, max_height, device_id=0):
        del max_width, max_height
        with self.lock:
            self.healthy()
            obj._context = None
            obj._egl_display = 0
            obj._egl_owner_thread = threading.current_thread()
            obj._egl_state = "creating"
            obj._egl_render_ref = None
            # A separate ownership bit survives GC clearing weakrefs before
            # invoking finalizers in an arbitrary order.
            obj._egl_owned_by_render = False
            saved = self.snapshot()
            pin = self.pin_binding(saved)
            self.all_managed.add(obj)
            try:
                if self.device_id is not None and device_id != self.device_id:
                    raise LifecycleError("one EGL device selection per process; changing device unsupported")
                obj._egl_display, obj._context = self.driver.create(device_id)
                self.device_id = device_id
                self.counts["created"] += 1
                self.driver.restore(saved)
                obj._egl_state = "live"
                self.gl_live.add(obj)
                self.emit("egl_created", owner=addr(obj._context))
            except BaseException:
                # No MjrContext exists yet, so EGL-only partial cleanup is safe.
                try:
                    if addr(obj._context):
                        self._gl_free(obj, allow_creating=True)
                    self.driver.restore(saved)
                    obj._egl_state = "closed"
                except BaseException as cleanup:
                    obj._egl_state = "failed"
                    self.fail(obj, cleanup)
                raise

    def gl_current(self, obj):
        with self.lock:
            self.healthy()
            self.thread(obj)
            if getattr(obj, "_egl_state", None) != "live":
                raise LifecycleError("make_current on non-live EGL context")
            saved = self.snapshot()
            pin = self.pin_binding(saved)
            try:
                self.driver.bind(Binding(obj._egl_display, 0, 0, addr(obj._context), OPENGL))
            except BaseException:
                try:
                    self.restore_saved(saved, pin)
                except BaseException as exc:
                    self.fail(obj, exc)
                raise

    def _gl_free(self, obj, allow_creating=False, *, from_render=False):
        state = getattr(obj, "_egl_state", None)
        if state == "closed" or (state is None and not addr(getattr(obj, "_context", None))):
            return
        if state == "failed":
            raise LifecycleError("previous EGL free failed; quarantine retained")
        if state != "live" and not (allow_creating and state == "creating"):
            raise LifecycleError("unknown or reentrant EGL free")
        self.thread(obj)
        ref = getattr(obj, "_egl_render_ref", None)
        render = ref() if ref else None
        if getattr(obj, "_egl_owned_by_render", False) and not from_render:
            raise LifecycleError("must retire MjrContext before destroying its EGL owner")
        if render is not None and getattr(render, "con", None) is not None:
            raise LifecycleError("must retire MjrContext before destroying its EGL owner")
        owner = addr(obj._context)
        if not owner:
            obj._egl_state = "closed"
            return
        saved = self.snapshot()
        pin = self.pin_binding(saved)
        try:
            # EGL destroy of a noncurrent context MUST NOT release the thread,
            # unbind the foreign context, or terminate the shared display.
            if saved.context == owner:
                self.driver.clear(saved.display)
            self.driver.destroy(obj._egl_display, obj._context)
            obj._context = None
            obj._egl_owned_by_render = False
            obj._egl_state = "closed"
            self.gl_live.discard(obj)
            self.counts["destroyed"] += 1
            expected = Binding(0, 0, 0, 0, saved.api) if saved.context == owner else saved
            self.driver.restore(expected)
            self.emit("egl_destroyed", owner=owner, restored=expected.context)
        except BaseException as exc:
            obj._egl_state = "failed"
            self.fail(obj, exc)
            # If destroy did not complete, don't leave unrelated binding clobbered.
            if saved.context != owner or addr(obj._context):
                try:
                    self.driver.restore(saved)
                except BaseException as restore_error:
                    self.fail(obj, restore_error)
            raise

    def gl_free(self, obj):
        with self.lock:
            return self._gl_free(obj)

    def gl_del(self, obj):
        try:
            with self.lock:
                # GL may be finalized first in the sim/render reference cycle.
                # Never destroy its context while MjrContext still owns it.
                # Render finalizer or explicit close is the only release authority.
                if getattr(obj, "_egl_owned_by_render", False):
                    self.emit("egl_finalizer_deferred_to_render", owner=addr(getattr(obj, "_context", None)))
                    return
                self._gl_free(obj)
        except BaseException as exc:
            self.fail(obj, exc)

    def render_init(self, obj, sim, *args, **kwargs):
        with self.lock:
            self.healthy()
            # If replacing a renderer on the SAME sim, retire it before taking
            # the saved binding. Never try to restore a destroyed old owner.
            self.retire_sim_renderer(sim)
            obj._egl_owner_thread = threading.current_thread()
            obj._egl_render_state = "creating"
            saved = self.snapshot()
            pin = self.pin_binding(saved)
            self.all_managed.add(obj)
            try:
                self.orig_render_init(obj, sim, *args, **kwargs)
                obj.gl_ctx._egl_render_ref = weakref.ref(obj)
                obj.gl_ctx._egl_owned_by_render = True
                # Vendor callers (including EvalV2 verify_current) expect a
                # usable GL context after creating the first renderer. Keep
                # the new owner current only when there was no previous one.
                if saved.context:
                    self.driver.restore(saved)
                elif self.driver.snapshot().context != addr(obj.gl_ctx._context):
                    raise LifecycleError("new renderer owner not current after init")
                obj._egl_render_state = "live"
                self.render_live.add(obj)
            except BaseException:
                try:
                    self.render_close(obj)
                except BaseException as exc:
                    self.fail(obj, exc)
                finally:
                    try:
                        self.driver.restore(saved)
                    except BaseException as exc:
                        self.fail(obj, exc)
                raise

    def render_close(self, obj):
        with self.lock:
            state = getattr(obj, "_egl_render_state", None)
            if state == "closed":
                return
            if state == "failed":
                raise LifecycleError("previous render close failed; quarantine retained")
            if state not in ("creating", "live"):
                # Refuse an untracked pre-install context, rather than free it blindly.
                if state is None and not hasattr(obj, "gl_ctx") and not hasattr(obj, "con"):
                    return
                raise LifecycleError("unowned or reentrant render close")
            self.thread(obj)
            obj._egl_render_state = "closing"
            gl = getattr(obj, "gl_ctx", None)
            try:
                con = getattr(obj, "con", None)
                if con is not None:
                    if gl is None:
                        raise LifecycleError("MjrContext lacks EGL owner")
                    with self.bound(gl):
                        self.emit("mjr_free_bound", owner=addr(gl._context), active=self.driver.snapshot().context)
                        con.free()
                        obj.con = None
                        del con  # native pybind finalization occurs while owner current
                        self.counts["mjr_freed"] += 1
                if gl is not None:
                    self._gl_free(gl, from_render=True)
                    obj.gl_ctx = None
                sim = getattr(obj, "sim", None)
                if sim is not None and getattr(sim, "_render_context_offscreen", None) is obj:
                    sim._render_context_offscreen = None
                for field in ("sim", "model", "data", "scn", "cam", "vopt", "pert"):
                    if hasattr(obj, field):
                        setattr(obj, field, None)
                obj._egl_render_state = "closed"
                self.render_live.discard(obj)
            except BaseException as exc:
                obj._egl_render_state = "failed"
                self.fail(obj, exc)
                raise

    def render_del(self, obj):
        try:
            with self.lock:
                if getattr(obj, "_egl_render_state", None) not in (None, "closed"):
                    self.counts["fallback_close"] += 1
                self.render_close(obj)
        except BaseException as exc:
            self.fail(obj, exc)

    def render_operation(self, obj, original, *args, **kwargs):
        with self.lock:
            self.healthy()
            self.thread(obj)
            if getattr(obj, "_egl_render_state", None) not in ("creating", "live"):
                raise LifecycleError("render operation on closed/failed context")
            # All operations that touch MjrContext, including resize (which frees
            # old buffers) and texture upload, bind the actual owner first.
            with self.bound(obj.gl_ctx):
                return original(obj, *args, **kwargs)

    def render_resize(self, obj, width, height):
        with self.lock:
            self.healthy()
            self.thread(obj)
            if getattr(obj, "_egl_render_state", None) != "live":
                raise LifecycleError("resize on non-live context")
            if width == obj.con.offWidth and height == obj.con.offHeight:
                return
            old_freed = False
            try:
                with self.bound(obj.gl_ctx):
                    obj.model.vis.global_.offwidth = width
                    obj.model.vis.global_.offheight = height
                    old = obj.con
                    try:
                        old.free()
                    except BaseException as exc:
                        obj._egl_render_state = "failed"
                        self.fail(obj, exc)
                        raise
                    obj.con = None
                    del old
                    self.counts["mjr_freed"] += 1
                    self.counts["resize_freed"] = self.counts.get("resize_freed", 0) + 1
                    old_freed = True
                    obj._set_mujoco_context_and_buffers()
            except BaseException as exc:
                # A failed allocation after successful old-buffer free can
                # safely retire the entire renderer, including a partial new
                # MjrContext. Never retry unknown/partially failed native free.
                if old_freed and not self.failures:
                    try:
                        self.render_close(obj)
                    except BaseException as cleanup:
                        self.fail(obj, cleanup)
                elif obj._egl_render_state != "failed":
                    obj._egl_render_state = "failed"
                    self.fail(obj, exc)
                raise

    def retire_sim_renderer(self, sim):
        if sim is None:
            return
        ctx = getattr(sim, "_render_context_offscreen", None)
        if ctx is not None:
            self.render_close(ctx)
            self.counts["explicit_retire"] += 1

    def sim_add(self, sim, new):
        with self.lock:
            old = getattr(sim, "_render_context_offscreen", None)
            if old is not None and old is not new:
                self.retire_sim_renderer(sim)
            return self.orig_sim_add(sim, new)

    def sim_free(self, sim):
        with self.lock:
            self.retire_sim_renderer(sim)
            # The vendor free deletes these and invokes gc.collect. Neither GL
            # cleanup nor cycle breaking should depend on collection scheduling.
            for k in ("data", "model"):
                if hasattr(sim, k):
                    delattr(sim, k)
            sim._render_context_offscreen = None

    def env_initialize(self, env, *args, **kwargs):
        with self.lock:
            self.healthy()
            if getattr(env, "_egl_reset_failed", False):
                raise LifecycleError("failed Environment must be closed, not reused")
            # Covers hard reset for renderer != "mujoco", and XML replacement.
            try:
                self.retire_sim_renderer(getattr(env, "sim", None))
                return self.orig_env_initialize(env, *args, **kwargs)
            except BaseException:
                env._egl_reset_failed = True
                try:
                    self.retire_sim_renderer(getattr(env, "sim", None))
                except BaseException as cleanup:
                    self.fail(env, cleanup)
                raise

    def env_reset(self, env, *args, **kwargs):
        with self.lock:
            self.healthy()
            if getattr(env, "_egl_reset_failed", False):
                raise LifecycleError("failed Environment must be closed, not reused")
            try:
                return self.orig_env_reset(env, *args, **kwargs)
            except BaseException:
                env._egl_reset_failed = True
                # Failed reset is not a reusable Episode. Retire any attached
                # partially initialized renderer, preserve the original error.
                try:
                    self.retire_sim_renderer(getattr(env, "sim", None))
                except BaseException as cleanup:
                    self.fail(env, cleanup)
                raise

    def env_close(self, env, *args, **kwargs):
        with self.lock:
            try:
                return self.orig_env_close(env, *args, **kwargs)
            except BaseException:
                try:
                    self.retire_sim_renderer(getattr(env, "sim", None))
                except BaseException as cleanup:
                    self.fail(env, cleanup)
                raise

    def replace(self, cls, name, fn):
        exists = name in vars(cls)
        self.originals.append((cls, name, exists, getattr(cls, name, None)))
        setattr(cls, name, fn)

    def apply(self):
        if self.installed:
            raise LifecycleError("duplicate lifecycle installation")
        b, e, env = self.b, self.e, self.env
        R, G, S, E = b.MjRenderContext, e.EGLGLContext, b.MjSim, env.MujocoEnv
        self.orig_render_init = R.__init__
        self.orig_sim_add = S.add_render_context
        self.orig_env_initialize = E._initialize_sim
        self.orig_env_reset = E.reset
        self.orig_env_close = E.close
        # One shared reentrant lock avoids lock inversion in nested render/resize
        # and MjSim.render's original render/read pair.
        self.replace(b, "_MjSim_render_lock", self.lock)
        for cls, name, method in (
            (G, "__init__", self.gl_init), (G, "make_current", self.gl_current),
            (G, "free", self.gl_free), (G, "__del__", self.gl_del),
            (R, "__init__", self.render_init), (R, "__del__", self.render_del),
            (R, "close", self.render_close), (R, "update_offscreen_size", self.render_resize), (S, "add_render_context", self.sim_add),
            (S, "free", self.sim_free), (E, "_initialize_sim", self.env_initialize),
            (E, "reset", self.env_reset), (E, "close", self.env_close),
        ):
            self.replace(cls, name, lambda obj, *a, _fn=method, **kw: _fn(obj, *a, **kw))
        for name in ("render", "read_pixels", "upload_texture"):
            orig = getattr(R, name)
            self.replace(R, name, lambda obj, *a, _orig=orig, **kw: self.render_operation(obj, _orig, *a, **kw))
        self.installed = True
        return self

    def audit(self):
        with self.lock:
            return dict(id=ID, counts=dict(self.counts), live_egl=len(self.gl_live),
                        live_render=len(self.render_live), quarantined=len(self.quarantine),
                        failures=list(self.failures), events=list(self.events))

    def assert_quiescent(self):
        with self.lock:
            self.healthy()
            if self.gl_live or self.render_live or self.counts["created"] != self.counts["destroyed"]:
                raise LifecycleError("explicit close and balanced native EGL allocation required before shutdown")

    def uninstall(self):
        # Intended only for isolated unit tests / quiescent processes.
        with self.lock:
            self.assert_quiescent()
            if self.all_managed:
                raise LifecycleError("managed Python objects still alive; uninstall unsafe")
            for cls, name, exists, old in reversed(self.originals):
                if exists:
                    setattr(cls, name, old)
                else:
                    delattr(cls, name)
            self.originals.clear()
            self.installed = False

_INSTALLED = None
def install():
    """Must be called before creating any Robosuite EGL context in this process."""
    global _INSTALLED
    if _INSTALLED is not None:
        raise LifecycleError("candidate already installed")
    from robosuite.utils import binding_utils as b
    from robosuite.renderers.context import egl_context as e
    from robosuite.environments import base as env
    for name, m in (("binding", b), ("egl", e), ("env", env)):
        if hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() != PINS[name]:
            raise LifecycleError("unqualified vendor source: " + name)
    if b._MUJOCO_GL != "egl" or e.EGL_DISPLAY is not None:
        raise LifecycleError("install before first EGL allocation, EGL backend only")
    _INSTALLED = Lifecycle(b, e, env, NativeEGL(e)).apply()
    return _INSTALLED
