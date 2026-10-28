"""Production D5 scheduler: ordered dispatch, same-round concurrency, Core admission.

No GPUs, HTTP, inference or simulators. CPU workers send token-bound messages to
one locked coordinator; model/server/env lifetimes are specified in r2 design.
"""

from __future__ import annotations

import threading
import time
import uuid

from cosmos_framework.evaluation.robocasa_v2.store import save

OWNER_PENDING = ("skipped_init_error", "scene_admission_exhausted")


def logical_plan(manifest):
    return [(t, ep) for ep in range(manifest["episodes_per_task"]) for t in manifest["task_order"]]


def slot_disposition(store, task, ep):
    state = store.slot(task, ep)
    if state["result"] is not None:
        return "skip_valid_core"
    if state["status"] in OWNER_PENDING and not store._permit(task, ep):
        return "skip_owner_pending"
    return "execute"


class MeasuredLock:
    def __init__(self, store):
        self.lock = threading.Lock()
        self.store = store

    def __enter__(self):
        start = time.perf_counter()
        self.lock.acquire()
        self.store.record_timing("scheduler_lock_wait", time.perf_counter() - start)
        return self

    def __exit__(self, *args):
        self.lock.release()


class RoundScheduler:
    def __init__(self, store, workers):
        store._lease()
        if store.scheduler_attached:
            raise ValueError("another scheduler owns coordinator")
        if not workers or len(set(workers)) != len(workers) or any(not isinstance(w, str) or not w for w in workers):
            raise ValueError("distinct physical worker names required")
        # A process from an old coordinator is not safe just because its lock died.
        for task in store.manifest["task_order"]:
            for s in store._streams(task):
                if not store._cleaned(task, s["stream_id"]) and s != store.active_tasks.get(task):
                    raise ValueError("safe stop: prior coordinator worker cleanup required")
        self.store, self.workers = store, list(workers)
        self.invocation = uuid.uuid4().hex
        self.round = self.cursor = 0
        self.inflight = {}
        self.cancelled = False
        self._lock = MeasuredLock(store)
        self.log_index = 0
        self.resume_scan = []
        store.scheduler_attached = True
        self._record("START", {"workers": self.workers, "manifest_digest": store.manifest["digest"]})

    def _record(self, kind, data):
        save(
            self.store.root / "_internal/scheduler" / self.invocation / f"{self.log_index:06d}.json",
            {
                "kind": kind,
                "data": data,
                "round": self.round,
                "time_ns": time.time_ns(),
                "invocation": self.invocation,
                "lease_id": self.store.lease_id,
            },
        )
        self.log_index += 1

    def _check(self):
        self.store._lease()
        if self.cancelled:
            raise ValueError("scheduler cancelled; cleanup and explicit Resume required")

    def _disposition(self, task, ep):
        return slot_disposition(self.store, task, ep)

    def dispatch(self, worker):
        with self._lock:
            self._check()
            if worker not in self.workers:
                raise ValueError("unknown physical worker")
            if worker in self.inflight:
                raise ValueError("worker already has a slot")
            tasks = self.store.manifest["task_order"]
            while self.round < self.store.manifest["episodes_per_task"]:
                while self.cursor < len(tasks):
                    task, ep = tasks[self.cursor], self.round
                    disposition = self._disposition(task, ep)
                    if disposition != "execute":
                        self._record(disposition, {"task": task, "episode": ep})
                        self.resume_scan.append((task, ep, disposition))
                        self.cursor += 1
                        continue
                    # Sticky task/env ownership avoids serializing all work onto
                    # one GPU or recreating task contexts for every round.
                    target = self.workers[self.cursor % len(self.workers)]
                    if worker != target:
                        return None
                    if task not in self.store.active_tasks:
                        self.store.begin_task(task)
                    ticket = uuid.uuid4().hex
                    self.store.dispatch_tickets[(task, ep)] = ticket
                    try:
                        a = self.store.begin_episode(task, ep, dispatch_ticket=ticket)
                    except BaseException:
                        self.store.dispatch_tickets.pop((task, ep), None)
                        raise
                    claim = {
                        "dispatch_id": ticket,
                        "physical_worker": worker,
                        "task": task,
                        "episode": ep,
                        "attempt": a,
                        "invocation": self.invocation,
                    }
                    # Durable dispatch before publishing token; abandoned claims
                    # are rediscovered from store attempts, never counted as DONE.
                    self.inflight[worker] = claim
                    self._record("DISPATCH", claim)
                    self.resume_scan.append((task, ep, "execute"))
                    self.cursor += 1
                    return claim
                if self.inflight:
                    return None  # Strict round barrier, regardless of completion order.
                # Recheck authority; neither RUNNING nor an Invalid counts as DONE.
                if any(self._disposition(t, self.round) == "execute" for t in tasks):
                    raise ValueError("round lacks terminal Core/Owner-pending disposition; cancel required")
                self._record("ROUND_BARRIER", {"episode": self.round})
                self.round += 1
                self.cursor = 0
            return None

    def _owned(self, claim):
        self._check()
        if self.inflight.get(claim["physical_worker"]) != claim or claim["invocation"] != self.invocation:
            raise ValueError("stale/duplicate/misrouted scheduler claim")
        if claim["episode"] != self.round:
            raise ValueError("claim crossed round barrier")

    def commit(self, claim, row, *, staging):
        with self._lock:
            self._owned(claim)
            if (row["task"], row["episode"]) != (claim["task"], claim["episode"]):
                raise ValueError("cross-slot worker response")
            self.store.commit(row, staging=staging, dispatch_ticket=claim["dispatch_id"])

    def finish(self, claim):
        with self._lock:
            self._owned(claim)
            disposition = self._disposition(claim["task"], claim["episode"])
            if disposition == "execute":
                raise ValueError("uncommitted/Invalid is not finished; cancel required")
            self._record("FINISH", {"claim": claim, "disposition": disposition})
            self.inflight.pop(claim["physical_worker"])
            self.store.dispatch_tickets.pop((claim["task"], claim["episode"]))

    def cancel(self, reason):
        with self._lock:
            self.store._lease()
            if not isinstance(reason, str) or not reason:
                raise ValueError("explicit cancellation reason required")
            self._record("CANCEL", {"reason": reason, "inflight": list(self.inflight.values())})
            self.cancelled = True
            # Do NOT kill, close sessions, mark DONE, or advance a round here.
            # Actual native owner cleanup must provide receipts before Resume.

    @property
    def done(self):
        return not self.cancelled and self.round == self.store.manifest["episodes_per_task"] and not self.inflight
