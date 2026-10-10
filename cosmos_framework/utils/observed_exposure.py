# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Summarize only replay-filtered observed task counts, without importing Torch."""

from __future__ import annotations

from collections import Counter


def observed_task_exposure(records: list[dict]) -> dict:
    counts = Counter()
    audited = []
    for row in records:
        if row.get("consumer_audit_status") != "OBSERVED":
            continue
        per_task = row.get("task_consumer_counts_rank_local")
        if not isinstance(per_task, dict) or any(
            not isinstance(task, str) or type(n) is not int or n < 0 for task, n in per_task.items()
        ):
            raise ValueError("invalid observed task exposure")
        if sum(per_task.values()) != row.get("actual_consumer_identity_count"):
            raise ValueError("observed exposure disagrees with consumed identities")
        counts.update(per_task)
        audited.append(row["iteration"])
    return {
        "task_exposure_observed_rank_local": dict(sorted(counts.items())),
        "task_exposure_scope": "available_replay_filtered_records_only_not_global_or_unique_corpus",
        "task_audited_records": len(audited),
        "task_first_audited_iteration": min(audited) if audited else None,
        "task_last_audited_iteration": max(audited) if audited else None,
    }
