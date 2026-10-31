"""Pure runtime-only options for the No-TTT CPU DataLoader, never a training identity."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LoaderProfile:
    workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    timeout_s: float | None = None
    max_packet_bytes: int = 32 * 1024**2
    max_cache_bytes: int = 2 * 1024**3

    def __post_init__(self):
        if type(self.workers) is not int or self.workers not in (0, 1, 2, 4, 8):
            raise ValueError("No-TTT loader workers must be 0/1/2/4/8")
        if type(self.persistent_workers) is not bool:
            raise ValueError("persistent_workers must be boolean")
        if self.workers == 0:
            if self.prefetch_factor is not None or self.persistent_workers or self.timeout_s is not None:
                raise ValueError("workers=0 disables prefetch/persistent/timeout")
        elif (
            type(self.prefetch_factor) is not int
            or self.prefetch_factor not in (1, 2)
            or type(self.timeout_s) not in (int, float)
            or not 0 < self.timeout_s < 3600
        ):
            raise ValueError("workers>0 requires prefetch1/2 and finite timeout in (0,3600)")
        if any(
            type(v) is not int or not 1024 <= v <= 4 * 1024**3 for v in (self.max_packet_bytes, self.max_cache_bytes)
        ):
            raise ValueError("invalid CPU loader memory bound")

    def identity(self):
        return {
            "schema": "nottt_cpu_raw_dataloader_v1",
            "start_method": "spawn",
            "pin_memory": False,
            "source_reader": "parent_verified_source",
            "transform": "parent_legacy_global",
            "source_packet_limit_bytes": min(self.max_packet_bytes, 1024**2),
            **asdict(self),
        }

    def async_identity(self):
        """Runtime provenance only; never part of training/frontier identity."""
        return {
            **self.identity(),
            "schema": "nottt_cpu_async_dataloader_v1",
            "source_reader": "worker_offline_verified_source",
            "transform": "worker_conditioned_parent_ordered_cfg_selection",
            "max_prepared_windows": 2,
            "max_inflight_packets": self.workers * (self.prefetch_factor or 0),
            "max_inflight_encoded_bytes": self.workers * (self.prefetch_factor or 0) * self.max_packet_bytes,
            "rank_mmap_budget_bytes": self.workers * self.max_cache_bytes,
            "worker_mmap_budget_bytes": self.max_cache_bytes,
        }


def profile_from_args(args):
    workers = getattr(args, "no_ttt_loader_workers", 0)
    factor = getattr(args, "no_ttt_prefetch_factor", None)
    persistent = getattr(args, "no_ttt_persistent_workers", None)
    timeout = getattr(args, "no_ttt_loader_timeout_s", None)
    if workers:
        factor = 1 if factor is None else factor
        persistent = True if persistent is None else persistent
        timeout = 120.0 if timeout is None else timeout
    else:
        persistent = False if persistent is None else persistent
    profile = LoaderProfile(workers, factor, persistent, timeout)
    if workers and (
        getattr(args, "no_ttt", False) is not True
        or (args.world_size, args.ga, args.b, args.t) != (4, 4, 8, 16)
        or not args.job_name.startswith("no_ttt_")
        or not args.job_name.endswith("_world4_ga4")
    ):
        raise ValueError("multiprocess loader requires No-TTT world4/GA4/B8/T16 job namespace")
    return profile
