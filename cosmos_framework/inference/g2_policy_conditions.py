"""G2 condition preparation inside the existing RoboCasa Policy transaction.

Model hooks own tokenization/UND/Reader/GEN conversion. The spatial provider owns
the *current rollout* inventory and initializes GT task bindings. Neither is
implemented by a second forward/producer here. Public tensors use B's C0 ABI.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace

import torch

from cosmos_framework.data.generator.action.utils.static_task_text import StaticTaskText
from cosmos_framework.inference.g2_session_cache import CacheBudget, SessionConditionCache, content_key
from cosmos_framework.model.generator.mot.g2_contracts import (
    G2_ABI_VERSION,
    EffectiveTextBatch,
    G2SampleIdentity,
    GlobalCacheSignature,
    SpatialView,
    TextCacheSignature,
    UNDTextFeatures,
)


@dataclass(frozen=True, kw_only=True)
class OnlineSpatialSnapshot:
    """Provider boundary only; all model/view/geometry fields are the public C0 types.

    signature lists ONLY the selected mandatory top + related object views, in
    C0 order. References may have different historical owners; every owner must
    be <= this logical state. source is explicit, never inferred from a filename.
    The provider must verify content/recipe/mask versions against its inventory.
    """

    identity: G2SampleIdentity
    signature: GlobalCacheSignature
    source: str
    mask_content_ids: tuple[str, ...]


def mask_content_identity(mask: torch.Tensor) -> str:
    """Provider metadata identity, independently checked against loaded masks on miss."""
    if mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("Reader valid mask must be bool [H_z,W_z]")
    return content_key(tuple(mask.shape), mask.detach().cpu().tolist())


def _single_rank_only():
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() != 1
    ):
        raise RuntimeError(
            "Session condition preparation supports one model rank only; distributed cache protocol is NOT_IMPLEMENTED"
        )


def _global_content_key(signature: GlobalCacheSignature) -> str:
    payload = asdict(signature)
    # Provenance remains in the per-request snapshot. A redraw with identical
    # selected content must not invalidate solely because its owner advanced.
    for view in payload["views"]:
        del view["image_owner"]
        del view["image_owner_frame"]
    return content_key(G2_ABI_VERSION, payload)


def _view_content_key(signature, index, mask_id):
    view = signature.views[index]
    return content_key(
        G2_ABI_VERSION,
        view.view_id,
        view.content_version,
        view.recipe_id,
        view.geometry_version,
        asdict(signature.projections[index]),
        asdict(signature.rasters[index]),
        signature.view_types[index],
        signature.mask_version,
        mask_id,
    )


class G2ConditionRuntime:
    """A thin cache client, not a Session scheduler or Local-memory owner.

    hooks is supplied by B. Required methods:
    tokenize_static_prompt(str)->EffectiveTextBatch(B1),
    text_signature(session_id,text)->TextCacheSignature,
    global_signature(text_signature,variant)->GlobalCacheSignature(empty views),
    prepare_text(text)->UNDTextFeatures, prepare_gen_text_kv(features),
    prepare_task_queries(features), read_global(queries,tuple[SpatialView]),
    project_global(reader), prepare_global_kv(projected),
    generation_kwargs(text_features=tuple, text_kv=tuple,
                      global_tokens=tuple, global_kv=tuple)->dict.

    generation_kwargs must return B's actual shared generator kwargs, with no
    Local overrides. cache_revision binds fixed weights AND inference config.
    The provider implements snapshot(identity, empty_signature) and
    load_view(snapshot,index)->SpatialView. R1 never calls any Global hook/provider.
    """

    def __init__(
        self,
        hooks,
        *,
        profile: str,
        provider=None,
        cache_enabled=True,
        budget: CacheBudget = CacheBudget(),
        allow_fixtures=False,
    ):
        _single_rank_only()
        if profile not in ("g2_r1", "g2_r2") or type(cache_enabled) is not bool:
            raise ValueError("explicit g2_r1/g2_r2 inference profile and cache switch required")
        if profile == "g2_r2" and provider is None:
            raise ValueError("R2 requires an actual current-rollout spatial provider; expert playback is unsupported")
        self.hooks, self.profile, self.provider = hooks, profile, provider
        self.cache_enabled, self.allow_fixtures = cache_enabled, allow_fixtures
        self.cache = SessionConditionCache(hooks.cache_revision, budget)
        self._tasks: dict[str, tuple] = {}
        self._telemetry = {
            name: deque(maxlen=1024) for name in ("queue", "text_format_tokenize", "conditions", "backbone_sampling")
        }
        self.last_request_sources: tuple[OnlineSpatialSnapshot | None, ...] = ()

    @staticmethod
    def validate_request(request):
        record = request.get("g2")
        if not isinstance(record, dict) or set(record) not in (
            {"identity", "top_variant", "static_text"},
            {"identity", "static_text"},
        ):
            raise ValueError("exact g2 identity/static_text with optional top_variant envelope required")
        if record.get("top_variant", "with_robot") not in ("with_robot", "without_robot"):
            raise ValueError("requested top variant must be explicit")
        identity = G2SampleIdentity(**record["identity"])
        if any(type(frame) is not int for frame in (identity.decision_frame, identity.logical_state_frame)):
            raise ValueError("logical condition frames must be exact integers")
        if any(
            not isinstance(value, str) or not value or len(value) > 512
            for value in asdict(identity).values()
            if isinstance(value, str)
        ):
            raise ValueError("bounded condition source/version identities required")
        task = StaticTaskText(**record["static_text"])
        if (task.episode_id, task.subtask_id, task.enhanced_instruction) != (
            identity.episode_id,
            identity.subtask_id,
            request.get("prompt"),
        ):
            raise ValueError("static text must belong to this initialized subtask and enhanced instruction")
        local = request.get("local_memory", request.get("inference", {}))
        if identity.episode_id != local.get("episode_id"):
            raise ValueError("G2 condition episode differs from the existing Session assignment")
        if "consumer_step" in local and identity.decision_frame != local["consumer_step"]:
            raise ValueError("G2 decision frame differs from the existing Local evidence frontier")
        if any(
            not isinstance(value, str) or not value or len(value) > 256
            for value in (local.get("session_id"), identity.scene_id, identity.subtask_id)
        ):
            raise ValueError("bounded Session/scene/subtask identities required")
        return identity

    def _bind_task(self, session, identity, prompt, serialized):
        if session not in self._tasks and len(self._tasks) >= self.cache.budget.max_sessions:
            raise MemoryError("bounded task bindings exhausted; End/reset existing Session first")
        binding = (
            identity.episode_id,
            identity.scene_id,
            identity.subtask_id,
            identity.task_binding_id,
            content_key(prompt, serialized),
        )
        previous = self._tasks.get(session)
        if previous is not None:
            if previous[0] != identity.episode_id or previous[1] != identity.scene_id:
                raise ValueError("new episode/scene requires existing Session End/reset")
            if previous[2] == identity.subtask_id and previous != binding:
                raise ValueError("one static instruction/GT binding per subtask; explicit task switch required")
        self._tasks[session] = binding

    def _snapshot(self, identity, signature):
        result = self.provider.snapshot(identity, signature)
        if not isinstance(result, OnlineSpatialSnapshot) or result.identity != identity:
            raise ValueError("provider returned conditions for a different logical rollout state")
        if result.source != "online_rollout" and not (self.allow_fixtures and result.source == "synthetic_fixture"):
            raise ValueError("only current-rollout spatial input is permitted; expert playback is unsupported")
        actual = result.signature
        if len(result.mask_content_ids) != len(actual.views) or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in result.mask_content_ids
        ):
            raise ValueError("provider must bind actual valid_mask content SHA256 per selected view")
        if len(actual.views) > self.cache.budget.max_entries:
            raise MemoryError("selected view inventory exceeds explicit entry budget")
        # The provider fills selected-view fields, not model/text versions.
        if replace(actual, views=(), projections=(), rasters=(), view_types=()) != signature:
            raise ValueError("provider changed model/text/geometry recipe or requested top variant")
        if not any(
            scope == "floor" and direction == "floor_top" and variant == signature.top_variant
            for scope, direction, _room, _island, variant in actual.view_types
        ):
            raise ValueError("requested mandatory top variant is missing; no fallback permitted")
        if any(v.image_owner_frame > identity.logical_state_frame for v in actual.views):
            raise ValueError("view owner belongs to a future logical spatial state")
        if any(v[4] is not None and v[4] != signature.top_variant for v in actual.view_types):
            raise ValueError("only one requested top variant may be selected")
        return result

    def _load_view(self, snapshot, index):
        view = self.provider.load_view(snapshot, index)
        signature = snapshot.signature
        if not isinstance(view, SpatialView) or (
            view.identity != signature.views[index]
            or view.projection != signature.projections[index]
            or view.raster != signature.rasters[index]
            or (view.scope, view.direction, view.room_id, view.island_id, view.top_variant)
            != signature.view_types[index]
        ):
            raise ValueError("loaded view does not match the selected content/recipe/geometry inventory")
        if mask_content_identity(view.valid_mask) != snapshot.mask_content_ids[index]:
            raise ValueError("loaded valid_mask differs from its selected content identity")
        return view

    @contextmanager
    def prepare_many(self, requests, serialized_prompts, *, recipe_ids, effective_texts=None):
        _single_rank_only()
        if len(requests) != len(serialized_prompts) or not requests or len(requests) > 32:
            raise ValueError("bounded per-sample condition routing required")
        if len(recipe_ids) != len(requests) or any(not isinstance(r, str) or not r for r in recipe_ids):
            raise ValueError("shared static prompt recipe identity required per sample")
        if effective_texts is not None and len(effective_texts) != len(requests):
            raise ValueError("one C-prepared effective text per native batch sample required")
        if getattr(self.hooks, "training", False):
            raise RuntimeError("Session inference cache must not be used for training")
        refresh = getattr(self.hooks, "refresh_revision", None)
        if refresh is not None:
            refresh()
        revision = self.hooks.cache_revision
        if revision != self.cache.revision:
            self.reload(revision)
        started = time.perf_counter_ns()
        texts, text_kvs, projected, global_kvs, sources = [], [], [], [], []
        sessions = set()
        with ExitStack() as stack:
            for index, (request, serialized, recipe_id) in enumerate(
                zip(requests, serialized_prompts, recipe_ids, strict=True)
            ):
                identity = self.validate_request(request)
                local = request.get("local_memory", request.get("inference", {}))
                session = local["session_id"]
                if session in sessions:
                    raise ValueError("one condition request per Session in a native batch")
                sessions.add(session)
                self._bind_task(
                    session, identity, request["prompt"], (serialized, recipe_id, request["g2"]["static_text"])
                )
                lease = stack.enter_context(self.cache.request(session, enabled=self.cache_enabled))
                text = (
                    effective_texts[index]
                    if effective_texts is not None
                    else self.hooks.tokenize_static_prompt(serialized)
                )
                if not isinstance(text, EffectiveTextBatch) or len(text.lengths) != 1:
                    raise ValueError("model hook must return exact effective B1 text tokens/positions")
                signature = self.hooks.text_signature(session, text)
                if (
                    not isinstance(signature, TextCacheSignature)
                    or signature.session_id != session
                    or signature.text != text
                ):
                    raise ValueError("text hook changed effective token/condition/Session identity")
                text_key = content_key(G2_ABI_VERSION, recipe_id, asdict(signature))
                features = lease.get_or_compute("text", text_key, lambda: self._fixed(self.hooks.prepare_text, text))
                if not isinstance(features, UNDTextFeatures) or features.text != text:
                    raise ValueError("UND hook output must belong to this exact effective text")
                kv = lease.get_or_compute(
                    "text_kv", text_key, lambda: self._fixed(self.hooks.prepare_gen_text_kv, features)
                )
                texts.append(features)
                text_kvs.append(kv)
                if self.profile == "g2_r1":
                    projected.append(None)
                    global_kvs.append(None)
                    sources.append(None)
                    continue
                variant = request["g2"].get("top_variant", "with_robot")
                base = self.hooks.global_signature(signature, variant)
                if (
                    not isinstance(base, GlobalCacheSignature)
                    or base.views
                    or base.text != signature
                    or base.top_variant != variant
                ):
                    raise ValueError("Global model hook must supply an empty current-version signature")
                snapshot = self._snapshot(identity, base)
                sources.append(snapshot)
                queries_key = content_key(
                    text_key,
                    identity.subtask_id,
                    identity.task_binding_id,
                    base.reader_version,
                    base.geometry_encoding_version,
                )
                queries = lease.get_or_compute(
                    "task_queries", queries_key, lambda: self._fixed(self.hooks.prepare_task_queries, features)
                )
                global_key = content_key(
                    queries_key, _global_content_key(snapshot.signature), snapshot.mask_content_ids
                )

                def build_global():
                    views = tuple(
                        replace(
                            lease.get_or_compute(
                                "view_latent",
                                _view_content_key(snapshot.signature, i, snapshot.mask_content_ids[i]),
                                lambda i=i: self._load_view(snapshot, i),
                            ),
                            identity=snapshot.signature.views[i],
                        )
                        for i in range(len(snapshot.signature.views))
                    )
                    return self._fixed(self.hooks.read_global, queries, views)

                reader = lease.get_or_compute("global", global_key, build_global)
                tokens = lease.get_or_compute(
                    "global_projection", global_key, lambda: self._fixed(self.hooks.project_global, reader)
                )
                global_kv = lease.get_or_compute(
                    "global_kv", global_key, lambda: self._fixed(self.hooks.prepare_global_kv, tokens)
                )
                projected.append(tokens)
                global_kvs.append(global_kv)
            kwargs = self.hooks.generation_kwargs(
                text_features=tuple(texts),
                text_kv=tuple(text_kvs),
                global_tokens=tuple(projected),
                global_kv=tuple(global_kvs),
            )
            if not isinstance(kwargs, dict) or any(k.startswith("_local") for k in kwargs):
                raise ValueError("Global preparation cannot override the existing Local transaction")
            self.last_request_sources = tuple(sources)  # bounded by the native batch, no per-frame dictionary
            self._telemetry["conditions"].append(time.perf_counter_ns() - started)
            sampling = time.perf_counter_ns()
            try:
                yield kwargs
            finally:
                self._telemetry["backbone_sampling"].append(time.perf_counter_ns() - sampling)

    @staticmethod
    def _fixed(call, *args):
        # Only condition preparation; the existing Local inner update runs
        # before this context and retains its own gradient-mode boundary.
        with torch.no_grad():
            return call(*args)

    def record_queue(self, seconds):
        self._telemetry["queue"].append(int(seconds * 1e9))

    def record_text_preparation(self, nanoseconds):
        self._telemetry["text_format_tokenize"].append(nanoseconds)

    def reset(self, session_id):
        self.cache.reset(session_id)
        self._tasks.pop(session_id, None)
        self.last_request_sources = ()

    def reload(self, revision):
        # Weight reload invalidates derived tensors; static subtask identity persists.
        self.cache.reload(revision)
        self.last_request_sources = ()

    def info(self):
        return {
            "profile": self.profile,
            "cache_enabled": self.cache_enabled,
            "cache": self.cache.info(),
            "model_versions": dict(getattr(self.hooks, "versions", {})),
            "task_bindings": len(self._tasks),
            "host_ns": {name: tuple(values) for name, values in self._telemetry.items()},
            "telemetry_capacity_per_stage": 1024,
            "limits": "Host elapsed only. Real model parity/latency/provider require separate M08 execution.",
        }
