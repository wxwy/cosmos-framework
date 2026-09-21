"""Bind observed, executed evidence to the existing action-policy generation API."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.libero_pose_utils import libero_rotation_format
from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
from cosmos_framework.inference.local_memory_online import (
    EVIDENCE_VERSION,
    OnlineLocalMemory,
    OnlineMemoryRequest,
    OnlineRecentHistoryMemory,
)


class PolicyLocalMemoryAdapter:
    def __init__(self, service, *, mode="auto", max_sessions=64):
        if mode not in {"auto", "off", "required"}:
            raise ValueError("local memory mode must be auto, off, or required")
        self.service, self.mode = service, mode
        config = service.model.config
        self.history_mode = str(getattr(config, "history_mode", "none"))
        self.history_horizon = int(getattr(config, "local_history_horizon", 0))
        ttt_available = bool(getattr(config, "local_ttt_enabled", False))
        recent_available = bool(
            getattr(config, "local_history_enabled", False)
            and getattr(config, "local_history_backend", "recurrent") == "recurrent"
            and getattr(config, "local_history_canonical_evidence", False)
            and self.history_horizon > 0
        )
        window_available = self.history_mode == "window" and self.history_horizon > 0
        available = ttt_available or recent_available or window_available
        self.enabled = mode == "required" or (mode == "auto" and available)
        self.memory = None
        self.memory_kind = None
        if self.enabled and window_available:
            self.memory_kind = "native_window"
        elif self.enabled and ttt_available:
            runtime = getattr(service.model.net, "local_memory_runtime", None)
            if runtime is None:
                raise ValueError("required TTT Local Memory is absent from this checkpoint model")
            self.memory = OnlineLocalMemory(runtime.evidence_encoder, runtime.ttt_core, max_sessions=max_sessions)
            self.memory_kind = "ttt_fast_weight"
        elif self.enabled and recent_available:
            runtime = getattr(service.model.net, "local_history_runtime", None)
            if runtime is None or runtime.recurrent_backend is None:
                raise ValueError("required bounded recent-history runtime is absent from this checkpoint model")
            self.memory = OnlineRecentHistoryMemory(
                runtime.encoder,
                runtime.recurrent_backend,
                history_horizon=int(config.local_history_horizon),
                max_sessions=max_sessions,
            )
            self.memory_kind = "bounded_recent_history"
        elif self.enabled:
            raise ValueError("required Local Memory is absent from this checkpoint model")

    def _visual_summary(self, req, image):
        prep = self.service._prep_policy_item({**req, "image": image})
        first_frame = prep["video_padded"][:, :1].unsqueeze(0)
        device = next(self.memory.encoder.parameters()).device
        with torch.inference_mode():
            latent = self.service.model._encode_uint8_vision_item(first_frame.to(device))
            if latent.ndim != 5 or latent.shape[1] != 48:
                raise ValueError("canonical visual memory requires a 48-channel causal VAE latent")
            return F.adaptive_avg_pool2d(latent[0, :, 0].unsqueeze(0), (1, 2)).flatten().float().cpu()

    def _normalize_executed_action(self, raw, *, gripper_mode):
        raw = torch.as_tensor(raw, dtype=torch.float32).clone()
        if tuple(raw.shape) != (7,) or not torch.isfinite(raw).all():
            raise ValueError("executed LIBERO action must be finite xyz/axisangle/gripper [7]")
        # closed_loop_eval executes the environment command after mapping the dataset
        # gripper convention into LIBERO's [-1,1] actuator convention.  Local Memory
        # evidence must use the same raw convention as training, so invert that mapping
        # before the frozen axisangle->6d + action-normalization path.
        if gripper_mode == "zero_one":
            raw[6] = (1.0 - raw[6]) * 0.5
        elif gripper_mode == "pm_one":
            pass
        elif gripper_mode == "pm_one_flip":
            raw[6] = -raw[6]
        else:
            raise ValueError("completed evidence requires zero_one/pm_one/pm_one_flip gripper_mode")
        rotation = convert_rotation(raw[3:6].unsqueeze(0), input_format="axisangle", output_format="matrix")
        rotation = convert_rotation(rotation, input_format="matrix", output_format=libero_rotation_format("6d"))[0]
        action = torch.cat((raw[:3], rotation, raw[6:7]))
        service = self.service
        if service.action_normalization == "meanstd":
            lo, scale = service.action_mean, service.action_std
            if lo is None or scale is None or lo.numel() != 10:
                raise ValueError("executed-action normalization requires the matching 10-D training statistics")
            return (action - lo.cpu()) / scale.cpu().clamp(min=1e-8)
        lo, scale = service.action_min, service.action_range
        if lo is None or scale is None or lo.numel() != 10:
            raise ValueError("executed-action normalization requires the matching 10-D training statistics")
        return 2 * (action - lo.cpu()) / scale.cpu().clamp(min=1e-8) - 1

    def _window_payload(self, req):
        memory = req.get("local_memory")
        if not isinstance(memory, dict):
            raise ValueError("native window checkpoint requires a local_memory history request")
        if memory.get("evidence_version", EVIDENCE_VERSION) != EVIDENCE_VERSION:
            raise ValueError("unsupported local evidence version")
        if memory.get("evidence_format", "libero_rgb_action7_v1") != "libero_rgb_action7_v1":
            raise ValueError("native window requires libero_rgb_action7_v1 evidence")
        session_id = memory.get("session_id")
        episode_id = memory.get("episode_id")
        consumer_step = memory.get("consumer_step")
        if not isinstance(session_id, str) or not session_id or not isinstance(episode_id, str) or not episode_id:
            raise ValueError("native window requires non-empty session_id and episode_id")
        if not isinstance(consumer_step, int) or consumer_step < 0:
            raise ValueError("native window consumer_step must be a non-negative integer")
        rows = memory.get("evidence", [])
        if not isinstance(rows, list) or len(rows) > self.history_horizon:
            raise ValueError("native window evidence exceeds the configured history horizon")
        expected_steps = list(range(consumer_step - len(rows), consumer_step))
        source_steps = []
        frames = []
        actions = []
        image_sizes = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each native window evidence row must be an object")
            source_steps.append(row.get("source_step"))
            image = row.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError("native window evidence requires the completed pre-action image")
            prep = self.service._prep_policy_item({**req, "image": image})
            frames.append(prep["video_padded"][:, :1])
            image_sizes.append(prep["padded_image_size"])
            actions.append(
                self._normalize_executed_action(
                    row.get("executed_action"), gripper_mode=row.get("gripper_mode")
                )
            )
        if source_steps != expected_steps:
            raise ValueError(
                f"native window evidence must be the contiguous causal tail {expected_steps}, got {source_steps}"
            )
        if memory.get("reset", False) and consumer_step != 0:
            raise ValueError("native window reset is only valid at consumer_step=0")
        return {
            "session_id": session_id,
            "episode_id": episode_id,
            "consumer_step": consumer_step,
            "source_steps": source_steps,
            "frames": frames,
            "actions": actions,
            "image_sizes": image_sizes,
        }

    def _generate_native_window(self, reqs, batch, generate_fn):
        input_video_key = self.service._input_video_key()
        main_image_sizes = batch["image_size"]
        if not isinstance(main_image_sizes, torch.Tensor) or main_image_sizes.ndim != 2:
            raise ValueError("native window expects batched main image_size tensor")
        histories = [self._window_payload(req) for req in reqs]
        flat_image_sizes = []
        prefix_lengths = []
        for index, history in enumerate(histories):
            h = len(history["frames"])
            prefix_lengths.append(h)
            main_items = batch[input_video_key][index]
            if not isinstance(main_items, list) or len(main_items) != 1:
                raise ValueError("native window expects exactly one current/future vision item before prefixing")
            batch[input_video_key][index] = [*history["frames"], *main_items]
            flat_image_sizes.extend(history["image_sizes"])
            flat_image_sizes.append(main_image_sizes[index])

            main_action = batch["action"][index][0]
            if not isinstance(main_action, torch.Tensor) or main_action.ndim != 2:
                raise ValueError("native window expects one [T,D] action tensor per request")
            if h:
                history_action = torch.stack(history["actions"]).to(
                    device=main_action.device, dtype=main_action.dtype
                )
                if history_action.shape[1] > main_action.shape[1]:
                    raise ValueError("native window history action width exceeds model action width")
                padded_history = main_action.new_zeros((h, main_action.shape[1]))
                padded_history[:, : history_action.shape[1]] = history_action
                batch["action"][index] = [torch.cat([padded_history, main_action], dim=0)]

            plan = batch["sequence_plan"][index]
            plan.condition_frame_indexes_vision = [0]
            plan.condition_frame_indexes_action = list(range(h))
            plan.action_start_frame_offset = 1
            plan.has_local_memory = False

        batch["image_size"] = flat_image_sizes
        with torch.inference_mode():
            samples = generate_fn()
        actions = samples.get("action")
        if actions is None or len(actions) != len(reqs):
            raise FloatingPointError("policy returned missing actions for native window inference")
        trimmed_actions = []
        for action, h in zip(actions, prefix_lengths, strict=True):
            if not torch.isfinite(action).all() or action.shape[0] < h:
                raise FloatingPointError("policy returned invalid native-window action sequence")
            trimmed = action[h:]
            if trimmed.shape[0] != self.service.cfg.action_chunk_size:
                raise ValueError(
                    "native window generated an unexpected target action length: "
                    f"expected {self.service.cfg.action_chunk_size}, got {trimmed.shape[0]}"
                )
            trimmed_actions.append(trimmed)
        samples["action"] = trimmed_actions
        samples["_local_memory_status"] = [
            {
                "session_id": history["session_id"],
                "episode_id": history["episode_id"],
                "consumer_step": history["consumer_step"],
                "prefix_present": bool(history["source_steps"]),
                "replay": False,
                "memory_kind": "native_window",
                "retained_source_steps": history["source_steps"],
            }
            for history in histories
        ]
        return samples

    def _request(self, req):
        memory = req.get("local_memory")
        if not isinstance(memory, dict):
            raise ValueError("Local Memory checkpoint requires a local_memory session/episode/consumer_step/evidence request")
        if memory.get("evidence_version", EVIDENCE_VERSION) != EVIDENCE_VERSION:
            raise ValueError("unsupported local evidence version")
        rows = memory.get("evidence", [])
        if not isinstance(rows, list) or len(rows) > self.memory.max_evidence_steps:
            raise ValueError("invalid or oversized completed evidence list")
        fmt = memory.get("evidence_format", "canonical_features_v1")
        if fmt not in {"canonical_features_v1", "libero_rgb_action7_v1"}:
            raise ValueError("unsupported local evidence format")
        visuals, actions, steps = [], [], []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each completed evidence row must be an object")
            steps.append(row["source_step"])
            if fmt == "canonical_features_v1":
                visuals.append(torch.as_tensor(row["visual_summary"], dtype=torch.float32))
                actions.append(torch.as_tensor(row["executed_action"], dtype=torch.float32))
            else:
                visuals.append(self._visual_summary(req, row["image"]))
                actions.append(
                    self._normalize_executed_action(
                        row["executed_action"], gripper_mode=row.get("gripper_mode")
                    )
                )
        visual = torch.stack(visuals) if visuals else torch.empty(0, 96)
        action = torch.stack(actions) if actions else torch.empty(0, 10)
        return OnlineMemoryRequest(
            memory.get("session_id"),
            memory.get("episode_id"),
            memory.get("consumer_step"),
            tuple(steps),
            visual,
            action,
            memory.get("reset", False),
        )

    def generate(self, reqs, batch, generate_fn):
        if not self.enabled:
            if any(req.get("local_memory") is not None for req in reqs):
                raise ValueError("Local Memory is disabled; supplied evidence would otherwise be ignored")
            with torch.inference_mode():
                return generate_fn()
        if self.memory_kind == "native_window":
            return self._generate_native_window(reqs, batch, generate_fn)
        updates = []
        try:
            for req in reqs:
                updates.append(self.memory.prepare(self._request(req)))
            tokens = [update.token for update in updates]
            plans = batch["sequence_plan"]
            if len(plans) != len(tokens):
                raise ValueError("online prefix count differs from policy request count")
            batch["local_memory"] = tokens
            for plan, token in zip(plans, tokens, strict=True):
                plan.has_local_memory = token is not None
            with torch.inference_mode():
                samples = generate_fn()
            actions = samples.get("action")
            if actions is None or len(actions) != len(reqs) or any(not torch.isfinite(a).all() for a in actions):
                raise FloatingPointError("policy returned missing or non-finite actions; memory was not committed")
            self.memory.commit_many(tuple(updates))
            samples["_local_memory_status"] = [
                {
                    "session_id": update.session_id,
                    "episode_id": update.replacement.episode_id,
                    "consumer_step": update.replacement.consumer_step,
                    "prefix_present": update.replacement.token is not None,
                    "replay": update.replay,
                    "memory_kind": self.memory_kind,
                }
                for update in updates
            ]
            return samples
        except Exception:
            self.memory.abort_many(tuple(updates))
            raise

    def reset(self, session_id):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("reset requires a session_id")
        if self.memory is not None:
            self.memory.reset_session(session_id)

    def info(self):
        data = {} if self.memory is None else self.memory.metadata()
        if self.memory_kind == "native_window":
            cold_start = "step0 required; native history window starts empty and retains only the last H completed steps"
        elif self.memory_kind == "ttt_fast_weight":
            cold_start = "step0 required; inference fast state is initialized from checkpoint W0"
        elif self.enabled:
            cold_start = "step0 required; bounded recent-history buffer starts empty and stores no recurrent hidden state"
        else:
            cold_start = "disabled"
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "history_mode": self.history_mode,
            "evidence_version": EVIDENCE_VERSION,
            "memory_kind": data.get("memory_kind", self.memory_kind),
            "history_horizon": (
                self.history_horizon if self.memory_kind == "native_window" else data.get("history_horizon")
            ),
            "sessions": data.get("sessions", 0),
            "max_sessions": data.get("max_sessions", 0),
            "cold_start": cold_start,
        }
