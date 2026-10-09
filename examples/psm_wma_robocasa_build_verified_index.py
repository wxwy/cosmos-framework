"""One-time CPU cold build / read-only verification of the V3 RoboCasa static index.

Run once, OUTSIDE torchrun, before the first bounded GPU smoke. The output
directory must be outside the source worktree and must not already exist.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_cache import RoboCasaExactWindowCacheCatalog
from cosmos_framework.data.generator.action.datasets.robocasa_exact_window_source import RoboCasaExactWindowSourceReader
from cosmos_framework.data.generator.action.datasets.robocasa_verified_index import VerifiedExactWindowIndex


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache-root", required=True, type=Path)
    result.add_argument("--source-root", required=True, type=Path)
    result.add_argument("--index-root", required=True, type=Path)
    result.add_argument(
        "--verify-existing", action="store_true", help="Read-only fingerprint verification; never rebuild"
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    started = time.perf_counter()
    if args.verify_existing:
        index = VerifiedExactWindowIndex.open(
            args.index_root, cache_root=args.cache_root, source_root=args.source_root
        )
        catalog = RoboCasaExactWindowCacheCatalog(args.cache_root, verified_index=index)
        index.check_catalog(catalog)
        report = {
            "status": "INDEX_VALID",
            "verified_index_hit": True,
            "episodes": len(catalog.episodes),
            "windows": catalog.stats.exact_window_count,
            "cache_corpus_digest": catalog.corpus_digest,
            "source_binding_digest": index.source_binding_digest,
        }
    else:
        if args.index_root.exists():
            raise FileExistsError(f"verified index 已存在，拒绝覆盖：{args.index_root}")
        cache = RoboCasaExactWindowCacheCatalog(args.cache_root)
        source = RoboCasaExactWindowSourceReader(cache, args.source_root)
        index = VerifiedExactWindowIndex.build(args.index_root, cache_catalog=cache, source_reader=source)
        report = {
            "status": "INDEX_BUILT",
            "verified_index_hit": False,
            "episodes": len(cache.episodes),
            "windows": cache.stats.exact_window_count,
            "source_reader_stages": source.init_timings_ms,
            "cache_corpus_digest": cache.corpus_digest,
            "source_binding_digest": index.source_binding_digest,
        }
    report["index_root"] = str(args.index_root)
    report["elapsed_s"] = time.perf_counter() - started
    print("[CorrectedV3][dataset_index] " + json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
