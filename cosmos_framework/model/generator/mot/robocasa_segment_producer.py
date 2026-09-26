# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""单 episode/slot 的 B1 evidence producer；policy payload 由既有 RGB 路径提供。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from cosmos_framework.model.generator.mot.local_memory_segment import SegmentBatch, SegmentIdentity, SegmentProvenance
from cosmos_framework.model.generator.mot.robocasa_latent_evidence import RoboCasaLatentReader


class RoboCasaSegmentProducer:
    """T16 是 consumer 段长；raw15 必须由 V3 loader 转换，不读取 H5 robot/action。"""

    ttt_tbptt_steps = 16
    policy_chunk_length = 32
    policy_consumer_frames = 33

    def __init__(
        self,
        reader: RoboCasaLatentReader,
        *,
        episode_id: str,
        category: str,
        raw15: torch.Tensor,
        payload_at: Callable[[int], Any],
        manifest_digest: str,
        config_digest: str,
        source_digest: str,
    ) -> None:
        if episode_id != reader.episode_id or not category:
            raise ValueError("producer episode/category 与缓存身份不匹配")
        if raw15.shape != (reader.source_frames, 15) or raw15.device.type != "cpu":
            raise ValueError("raw15 必须为 V3 loader 转换的 CPU [source_frames,15]；禁止 H5 原生12D action")
        if not raw15.is_floating_point() or not torch.isfinite(raw15).all():
            raise ValueError("raw15 必须为有限浮点数，不能使用 padded64 或 state conditioning 行")
        if not callable(payload_at):
            raise ValueError("payload_at 必须是既有 native RGB consumer 的读取函数")
        self.reader, self.episode_id, self.category = reader, episode_id, category
        self._raw15 = raw15.detach().clone()
        self._payload_at = payload_at
        self._provenance = SegmentProvenance(manifest_digest, config_digest, source_digest, 0)
        # 与 upstream build_episode_spans 相同：每个 anchor 需要完整33帧。
        self.valid_consumer_count = max(0, reader.source_frames - self.policy_chunk_length)

    @property
    def segment_count(self) -> int:
        return (self.valid_consumer_count + self.ttt_tbptt_steps - 1) // self.ttt_tbptt_steps

    def identity(self, *, slot_id: int, cursor: int, segment_id: int) -> SegmentIdentity:
        if type(cursor) is not int or not 0 <= cursor < self.segment_count:
            raise ValueError("cursor 超出该 episode 的合法 consumer 段")
        return SegmentIdentity(
            slot_id,
            self.episode_id,
            self.category,
            cursor,
            segment_id,
            self._provenance.source_digest,
            training_stream_end=cursor == self.segment_count - 1,
        )

    def produce(self, identity: SegmentIdentity) -> SegmentBatch:
        expected = self.identity(slot_id=identity.slot_id, cursor=identity.cursor, segment_id=identity.segment_id)
        if identity != expected:
            raise ValueError("请求的 episode/category/source/terminal 与 producer 不匹配")
        start = identity.cursor * self.ttt_tbptt_steps
        count = min(self.ttt_tbptt_steps, self.valid_consumer_count - start)
        consumer_valid = torch.arange(self.ttt_tbptt_steps)[None] < count
        consumer_step = torch.full((1, self.ttt_tbptt_steps), -1, dtype=torch.long)
        consumer_step[0, :count] = torch.arange(start, start + count)
        evidence_valid = consumer_valid & (consumer_step > 0)
        source_step = torch.where(evidence_valid, consumer_step - 1, -1)
        current_visual = torch.zeros(1, self.ttt_tbptt_steps, 96)
        previous_visual = torch.zeros_like(current_visual)
        previous_action = self._raw15.new_zeros(1, self.ttt_tbptt_steps, 15)
        payloads: list[Any | None] = [None] * self.ttt_tbptt_steps
        for offset in range(count):
            step = start + offset
            payload = self._payload_at(step)
            if payload is None:
                raise ValueError(f"有效 consumer {step} 缺少 native payload")
            # 不修改 payload，不向主 policy 注入 latent；异常原样传播，不换样本。
            payloads[offset] = payload
            _, current_visual[0, offset] = self.reader.visual_summary(step)
            if step > 0:
                _, previous_visual[0, offset] = self.reader.visual_summary(step - 1)
                previous_action[0, offset] = self._raw15[step - 1]
        segment = SegmentBatch(
            consumer_visual_summary=current_visual,
            consumer_payload=(tuple(payloads),),
            consumer_valid=consumer_valid,
            consumer_step=consumer_step,
            evidence_visual_summary_prev=previous_visual,
            evidence_executed_action_prev=previous_action,
            evidence_valid=evidence_valid,
            evidence_source_step=source_step,
            slot_id=torch.tensor([identity.slot_id]),
            episode_id=(identity.episode_id,),
            category=(identity.category,),
            segment_provenance=SegmentProvenance(
                self._provenance.manifest_digest,
                self._provenance.config_digest,
                self._provenance.source_digest,
                identity.segment_id,
            ),
        )
        segment.validate(self.ttt_tbptt_steps)
        return segment
