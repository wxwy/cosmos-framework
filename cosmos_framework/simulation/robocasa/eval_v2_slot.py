"""One reusable physical Sim Slot; logical Task/stream/session are independent."""

from __future__ import annotations

import hashlib
from pathlib import Path

from cosmos_framework.evaluation.robocasa_v2.contract import SAMPLING
from cosmos_framework.evaluation.robocasa_v2.store import identifier, load
from cosmos_framework.simulation.robocasa import closed_loop_eval as native
from cosmos_framework.simulation.robocasa.eval_v2_adapter import NativeEpisode


class SimSlot:
    def __init__(self, cfg, *, factory=None):
        self.cfg, self.factory = cfg, factory
        self.state = "IDLE"
        self.history = [self.state]
        self.task = self.env = self.engine = self.attempt = self.grant = None
        self.reset_index = -1
        self.used_attempts, self.used_sessions, self.used_episodes = set(), set(), set()

    def _state(self, value):
        self.state = value
        self.history.append(value)
        self.history = self.history[-64:]

    def _ordinary_task(self, task):
        if task not in self.cfg["tasks"]:
            raise ValueError("Task outside attested Slot allowlist")
        spec = self.cfg["tasks"][task]
        path = Path(spec["dataset_dir"]) / "extras/dataset_meta.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["metadata_sha256"]:
            raise ValueError("Task metadata changed before environment construction")
        return spec

    def _make(self, task, grant):
        spec = self._ordinary_task(task)
        identifier(grant["environment_id"])
        path = Path(self.cfg["task_root"]) / task / "environments" / f"{grant['environment_id']}.json"
        if (
            load(path) != grant
            or grant["task"] != task
            or grant["physical_slot_id"] != self.cfg["physical_slot_id"]
            or grant["manifest_digest"] != self.cfg["manifest_digest"]
            or type(grant["seed"]) is not int
            or not 0 <= grant["seed"] < 2**32
        ):
            raise ValueError("foreign/uncommitted environment seed grant")
        self.env = (
            self.factory(task, spec, grant)
            if self.factory
            else native.make_env(
                spec["dataset_dir"],
                256,
                obj_split="target",
                layout_style=native.ATOMIC_SEEN_LAYOUT_STYLE,
                seed=grant["seed"],
            )
        )
        settings = dict(self.cfg["settings"], horizon=spec["horizon"])
        self.engine = NativeEpisode(self.env, settings)
        self.grant, self.reset_index, self.task = grant, -1, task

    def _claim(self, attempt, health_only):
        if attempt["lease_id"] != self.cfg["coordinator_lease_id"]:
            raise ValueError("stale coordinator lease")
        task = attempt["task"]
        stream = load(Path(self.cfg["task_root"]) / task / "streams" / f"{identifier(attempt['stream_id'])}.json")
        if any(attempt[k] != stream[k] for k in ("worker_id", "seed", "lease_id", "manifest_digest")):
            raise ValueError("foreign logical stream")
        if not health_only:
            from cosmos_framework.evaluation.robocasa_v2.contract import integer

            ep = integer(attempt["episode"], 0, self.cfg["episodes_per_task"] - 1, "Episode")
            generation = integer(attempt["generation"], 0, 100000, "attempt generation")
            path = Path(self.cfg["task_root"]) / task / "slots" / f"ep_{ep:03d}" / "attempts" / f"{generation:06d}.json"
            if load(path) != attempt:
                raise ValueError("uncommitted/foreign Episode claim")

    def switch_task(self, task, grant):
        """Fenced subtransition; requests cannot freely supply an env path/config."""
        if self.state != "CLAIMED" or self._desired_task != task:
            raise RuntimeError("Task switch requires an owned idle Slot claim")
        self._state("TASK_SWITCH")
        if self.env is not None:
            self.env.close()
            self.env = self.engine = None
        if grant is None:
            raise ValueError("new Task environment requires durable seed grant")
        self._make(task, grant)

    def start_episode(self, attempt, *, grant=None, health_only=False):
        if self.state not in ("IDLE", "CLEAN"):
            raise RuntimeError("Slot not clean; quarantine/process recovery required")
        if attempt["manifest_digest"] != self.cfg["manifest_digest"]:
            raise ValueError("foreign manifest/attempt")
        task = attempt["task"]
        self._ordinary_task(task)
        self._claim(attempt, health_only)
        if not health_only:
            token = identifier(attempt["attempt_id"])
            if token in self.used_attempts:
                raise ValueError("stale/duplicate Episode claim")
            self.used_attempts.add(token)
        self.history = [self.state]
        self._state("CLAIMED")
        try:
            self._desired_task = task
            if self.task != task or self.env is None:
                self.switch_task(task, grant)
            elif grant is not None:
                raise ValueError("same-Task environment must not be silently reseeded")
            self.attempt = dict(attempt)
            self._state("ENV_READY")
            if health_only:
                self._state("CLEAN")  # No Session/inference opened by a health-only assignment.
            return {
                "task": task,
                "physical_slot_id": self.cfg["physical_slot_id"],
                "environment_id": self.grant["environment_id"],
                "actual_seed": self.grant["seed"],
                "state": self.state,
                "attempt_id": attempt.get("attempt_id"),
            }
        except BaseException:
            self._state("FAILED_QUARANTINED")
            raise

    def initialize(self):
        if self.state not in ("ENV_READY", "ADMISSION"):
            raise RuntimeError("Slot cannot reset with an open/unowned Episode")
        self._state("RESETTING")
        self.reset_index += 1  # Includes failed and initially successful scene samples.
        value = self.engine.initialize()
        self._state("FAILED_QUARANTINED" if value.get("error") and not value.get("safe_to_retry") else "ADMISSION")
        return dict(value, sampling=self.sampling())

    def sampling(self):
        return {
            "environment_id": self.grant["environment_id"],
            "actual_seed": self.grant["seed"],
            "environment_reset_index": self.reset_index,
            "physical_slot_id": self.cfg["physical_slot_id"],
            "protocol": SAMPLING,
        }

    def finish_without_session(self):
        if self.state != "ADMISSION" or self.engine.admitted:
            raise RuntimeError("only rejected/unadmitted scenes can release without a Session")
        self._state("CLEAN")
        return {"state": self.state, "session_opened": False}

    def execute(self, attempt, *, reset_index, directory):
        if self.state != "ADMISSION" or not self.engine.admitted or attempt != self.attempt:
            raise ValueError("unadmitted/stale/cross-Slot Episode")
        expected = (
            Path(self.cfg["task_root"]).resolve()
            / attempt["task"]
            / "slots"
            / f"ep_{attempt['episode']:03d}"
            / "work"
            / attempt["attempt_id"]
        )
        if Path(directory).resolve() != expected:
            raise ValueError("output outside owned attempt staging")
        self._state("SESSION_OPEN")
        try:

            def phase(name):
                if name == "executing":
                    self._state("EXECUTING")
                elif name == "local_cleanup":
                    self._state("SESSION_CLOSING")

            response = self.engine.execute(
                attempt, reset_index=reset_index, directory=directory, sampling=self.sampling(), phase_callback=phase
            )
            session = response["local_session"]
            if (
                not session["reset_ack"]
                or session["session_id"] in self.used_sessions
                or session["episode_id"] in self.used_episodes
            ):
                raise RuntimeError("unproven cleanup or reused Local Session; Slot quarantined")
            self.used_sessions.add(session["session_id"])
            self.used_episodes.add(session["episode_id"])
            self._state("FAILED_QUARANTINED" if response.get("media_process_fault") else "CLEAN")
            from cosmos_framework.utils.rollout_evidence import atomic_json

            try:
                atomic_json(
                    Path(directory) / "slot_lifecycle.json",
                    {
                        "physical_slot_id": self.cfg["physical_slot_id"],
                        "attempt_id": attempt["attempt_id"],
                        "task": attempt["task"],
                        "states": self.history,
                    },
                )
            except OSError:
                pass  # Non-authoritative UI diagnostic cannot replace a valid Core.
            return dict(response, slot_state=self.state, sampling=self.sampling())
        except BaseException:
            self._state("FAILED_QUARANTINED")
            raise

    def close(self):
        was_clean = self.state in ("IDLE", "CLEAN")
        self._state("PROCESS_RECOVERY" if not was_clean else "CLEAN")
        if self.env is not None:
            self.env.close()
            self.env = self.engine = None
        return {"closed": True, "clean": was_clean}
