"""Shared Condition Cache budget for EvalV2 and the Policy Server."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class CacheBudget:
    max_sessions: int = 32
    max_entries: int = 256
    cpu_bytes: int = 256 * 1024**2
    gpu_bytes: int = 256 * 1024**2
    handles: int = 4096

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or not 1 <= value < 2**63:
                raise ValueError(f"condition-cache budgets require a positive bounded integer for {name}")

    @classmethod
    def from_options(cls, options):
        return cls(**{name: options[key] for name, key in CACHE_OPTIONS.items() if key in options})

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict or record.keys() != CACHE_OPTIONS.keys():
            raise ValueError("exact G2 Condition Cache budget fields required")
        return cls(**record)

    def validate_placement(self, *, slots, servers, rss_limit=None, gpu_limit=None):
        if self.max_sessions < slots:
            raise ValueError("G2 Condition Cache max_sessions must cover all assigned per-Policy slots")
        if rss_limit is not None and self.cpu_bytes * servers > rss_limit:
            raise ValueError("G2 Condition Cache CPU budgets exceed owned RSS limit")
        if gpu_limit is not None and self.gpu_bytes > gpu_limit:
            raise ValueError("G2 Condition Cache GPU budget exceeds per-device owned memory limit")


DEFAULT_CACHE_BUDGET = CacheBudget()
CACHE_OPTIONS = {field.name: f"g2_cache_{field.name}" for field in fields(CacheBudget)}
