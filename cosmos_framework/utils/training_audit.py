# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Read-only, rank-local consumer provenance and sampled runtime diagnostics."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch


def _tensor(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def _walk(value, prefix=''):
    if isinstance(value, torch.Tensor):
        yield prefix, _tensor(value)
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name), prefix + '/' + field.name)
    elif isinstance(value, Mapping):
        keys = list(value)
        if all(isinstance(key, (str, int)) for key in keys):
            keys.sort(key=str)
        for index, key in enumerate(keys):
            label = str(key) if isinstance(key, (str, int)) else f'item_{index}'
            yield from _walk(value[key], prefix + '/' + label)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk(item, prefix + '/' + str(index))


def fast_state_fingerprint(snapshot) -> str:
    digest = hashlib.sha256()
    for identity, provenance, state in sorted(snapshot, key=lambda row: row[0].slot_id):
        # Identity and provenance in this ABI are frozen scalar dataclasses.
        for binding in (identity, provenance):
            if not dataclasses.is_dataclass(binding):
                raise TypeError('expected frozen Local identity/provenance dataclass')
            digest.update(json.dumps(dataclasses.asdict(binding), sort_keys=True, allow_nan=False).encode())
        for name, tensor in _walk(state):
            host = tensor.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(host.dtype), list(host.shape)]).encode())
            digest.update(host.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def memory_inventory(model, optimizer) -> dict[str, Any]:
    """Actual local storage bytes; the residual is NOT labelled activation memory."""
    # The official host returns OptimizersContainer, not torch.optim.Optimizer.
    # Do not call state_dict(): it can communicate/gather under distributed DCP.
    inner = getattr(optimizer, "optimizers", None)
    if inner is not None:
        if not isinstance(inner, (list, tuple)) or not all(isinstance(getattr(opt, "state", None), Mapping) for opt in inner):
            raise TypeError("unsupported optimizer container; cannot measure its state")
        optimizer_states = tuple(opt.state for opt in inner)
        optimizer_count = len(inner)
    elif isinstance(getattr(optimizer, "state", None), Mapping):
        optimizer_states = (optimizer.state,)
        optimizer_count = 1
    else:
        optimizer_states = None
        optimizer_count = None
    groups = {"parameters": tuple(model.parameters()),
              "gradients": tuple(p.grad for p in model.parameters() if p.grad is not None)}
    if optimizer_states is not None:
        groups["optimizer"] = optimizer_states
    all_cuda = {}
    report = {"scope": "local_rank_unique_storage_current_not_peak",
              "optimizer_count": optimizer_count,
              "optimizer_state_status": "OBSERVED" if optimizer_states is not None else "UNAVAILABLE",
              "optimizer_storage_bytes": None, "optimizer_cuda_storage_bytes": None}
    for group, values in groups.items():
        seen = {}
        for _, tensor in _walk(values):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            seen[key] = storage.nbytes()
            if tensor.device.type == 'cuda':
                all_cuda[key] = storage.nbytes()
        report[group + '_storage_bytes'] = sum(seen.values())
        report[group + '_cuda_storage_bytes'] = sum(n for (device, _), n in seen.items() if device.startswith('cuda'))
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        report.update(current_allocated_bytes=allocated, current_reserved_bytes=torch.cuda.memory_reserved(),
                      unattributed_current_cuda_bytes=max(0, allocated - sum(all_cuda.values())))
    else:
        report.update(current_allocated_bytes=None, current_reserved_bytes=None,
                      unattributed_current_cuda_bytes=None)
    return report


def _int(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.numel() != 1:
            raise ValueError(f'{name} must be a scalar CPU identity')
        value = value.item()
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


class ConsumerAudit:
    def __init__(self) -> None:
        text = os.environ.get('PSM_V3_STATE_AUDIT_EVERY', '100')
        if not text.isascii() or not text.isdigit():
            raise ValueError('PSM_V3_STATE_AUDIT_EVERY must be a nonnegative integer')
        self.every = int(text)
        self.iteration = None
        self.first_iteration = None
        self.rows = []
        self.exposure = Counter()
        self._published = None
        self.memory = None

    def observe(self, phase, data, model) -> None:
        iteration = data.get('iteration')
        if type(iteration) is not int or iteration < 0:
            return
        if iteration != self.iteration:
            self.iteration, self.rows, self.memory = iteration, [], None
        if phase == 'native_forward' and 'payloads' in data:
            for payload in data['payloads']:
                task = payload.get('task_class')
                if not isinstance(task, str) or not task:
                    raise ValueError('consumer task_class missing')
                self.rows.append((data['member'], data['index'], task,
                                  _int(payload.get('episode_index'), 'episode_index'),
                                  _int(payload.get('start_frame'), 'start_frame'),
                                  payload.get('source_binding_digest'), payload.get('cache_corpus_digest')))
        if phase == 'pre_optimizer' and self.every and (iteration + 1) % self.every == 0:
            self.memory = memory_inventory(model, data.get('optimizer'))

    def report(self, trainer, iteration: int, expected_count: int | None = None) -> dict[str, Any]:
        if iteration != self.iteration or self._published == iteration:
            return {'consumer_audit_status': 'NO_NEW_CONSUMER_TRACE'}
        plan = getattr(trainer, '_grouped_window').plan
        # After commit the plan is cleared. The observer keeps the pre-optimizer
        # plan; the exact count is also retained by the completed trace itself.
        expected = expected_count if expected_count is not None else getattr(plan, 'n_window', None)
        if expected is not None and len(self.rows) != expected:
            raise ValueError('consumer trace count disagrees with the grouped plan')
        counts = Counter(row[2] for row in self.rows)
        if not self.rows:
            return {'consumer_audit_status': 'NO_PAYLOAD_IDENTITIES'}
        self.exposure.update(counts)
        self.first_iteration = iteration + 1 if self.first_iteration is None else self.first_iteration
        self._published = iteration
        result = {
            'consumer_audit_status': 'OBSERVED',
            'actual_consumer_identity_sha256': hashlib.sha256(json.dumps(self.rows, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'actual_consumer_identity_count': len(self.rows),
            'task_consumer_counts_rank_local': dict(sorted(counts.items())),
            'task_exposure_since_observer_start': dict(sorted(self.exposure.items())),
            'task_exposure_start_iteration': self.first_iteration,
            'exposure_scope': 'rank_local_observed_since_process_start_not_full_corpus',
            'state_fingerprint_sampled': bool(self.every and (iteration + 1) % self.every == 0),
            'memory_inventory': self.memory,
        }
        if result['state_fingerprint_sampled']:
            started = time.perf_counter()
            result['committed_fast_state_sha256'] = fast_state_fingerprint(trainer._grouped_window.live.sidecar.snapshot())
            result['state_fingerprint_ms'] = (time.perf_counter() - started) * 1000
        catalog = getattr(getattr(trainer, '_grouped_producer', None), 'catalog', None)
        raw = getattr(catalog, 'raw', None)
        reader = getattr(raw, 'cache_reader', None)
        shared_reader = callable(getattr(reader, 'cache_stats', None))
        result['data_pipeline_execution'] = {
            'num_workers': getattr(trainer, '_grouped_num_workers', 0),
            'compact_cached_video': getattr(raw, 'compact_cached_video', None),
            'rank_shared_episode_reader': shared_reader,
            'state_audit_every': self.every,
        }
        if shared_reader:
            result['episode_cache_rank_local'] = reader.cache_stats()
        return result


def audit_event(observer, trainer, phase, data) -> None:
    if observer.rank != 0:
        return
    try:
        if not hasattr(observer, '_consumer_audit'):
            observer._consumer_audit = ConsumerAudit()
        observer._consumer_audit.observe(phase, data, trainer._grouped_window.model)
    except Exception as error:
        observer._errors.append(f'consumer_audit:{type(error).__name__}:{error}')


def audit_record(observer, trainer, iteration) -> dict[str, Any]:
    if observer.rank != 0 or not hasattr(observer, '_consumer_audit'):
        return {}
    try:
        result = observer._consumer_audit.report(
            trainer, iteration, expected_count=getattr(observer._last_plan, 'n_window', None)
        )
        return result
    except Exception as error:
        return {'consumer_audit_status': 'ERROR', 'consumer_audit_error': f'{type(error).__name__}:{error}'}
