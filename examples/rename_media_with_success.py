#!/usr/bin/env python3
"""后处理：把已落盘的 per-episode 媒体按 ``summary.json`` 标 ``_success``/``_fail``
后缀。与 ``cosmos_framework.simulation.libero.closed_loop_eval._run_episode`` 内
现改的逻辑同步（v2 起会直接写入新名；本脚本给 v1 之前无 success/fail 命名的落盘做
回填）。

用法::

    # 单一 suite（action 必须提供 summary.json 所在目录）
    python examples/rename_media_with_success.py \
        results/libero_closed_loop_4in1_acceptance/iter_000002000/libero_spatial

    # 只回填某个 task/episode
    python examples/rename_media_with_success.py \
        results/.../libero_spatial --task 0 --episode 5

    # 预演，不真改名
    python examples/rename_media_with_success.py \
        results/.../libero_spatial --dry-run

只重命名 **summary.json 已固化** 的 episodes；不动接口、不写新 summary。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 4 类媒体根；与 closed_loop_eval.py:1303-1306 / :1334 的命名一一对应。
# `gifs` / `mp4_pred` 是单数，`comparisons` 是复数；launch 脚本默认不开
# comparison，但脚本仍会在该目录存在时回填。
# 每个根对应的扩展名：`mp4_pred` 是目录、其它都是文件（`.mp4` / `.gif`）。
MEDIA_ROOTS: tuple[tuple[str, str], ...] = (
    ("mp4", ".mp4"),
    ("gifs", ".gif"),
    ("mp4_pred", ""),  # 目录：episode_NNN/
    ("comparisons", ".gif"),
)


def _suffix(success: bool) -> str:
    return "success" if success else "fail"


def _rename_media(media_root: Path, ext: str, task_id: int, episode_idx: int, success: bool, dry_run: bool) -> str:
    """Return a one-line status string describing whether a rename happened."""
    task_dir = media_root / f"task_{task_id:03d}"
    src_name = f"episode_{episode_idx:03d}{ext}"
    src = task_dir / src_name
    if not src.exists():
        return f"  skip ({src_name}: 不存在)"
    suffix = _suffix(success)
    if ext == "":
        # 目录：直接把后缀贴到名字末尾
        dst = src.with_name(f"{src.name}_{suffix}")
    else:
        dst = src.with_name(f"{src.stem}_{suffix}{src.suffix}")
    if dst.exists():
        return f"  skip {src.name} -> {dst.name}（目标已存在，疑似已 rename）"
    if dry_run:
        return f"  dry-run {src.name} -> {dst.name}"
    src.rename(dst)
    return f"  renamed {src.name} -> {dst.name}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite_dir", type=Path, help="suite 输出目录（含 summary.json）")
    parser.add_argument("--task", type=int, default=None, help="只处理此 task_id")
    parser.add_argument("--episode", type=int, default=None, help="只处理此 episode_idx（与 --task 配套）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不真改名")
    args = parser.parse_args(argv)

    suite_dir: Path = args.suite_dir.resolve()
    summary_path = suite_dir / "summary.json"
    if not summary_path.is_file():
        print(f"!! {summary_path} 不存在；本脚本要求 summary.json 已固化", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text())

    # 仅保留 suite_dir 下确实存在的媒体根（comparison 默认不开）；每个根独立
    # 报告、互不影响，便于一眼看出哪几类有回填。
    present_roots: list[tuple[str, str, Path]] = []
    for sub, ext in MEDIA_ROOTS:
        root = suite_dir / sub
        if root.is_dir():
            present_roots.append((sub, ext, root))
    if not present_roots:
        print(f"!! {suite_dir} 下找不到任何媒体根 {[s for s, _ in MEDIA_ROOTS]}，nothing to do", file=sys.stderr)
        return 1

    renamed = 0
    skipped = 0
    for tr in summary.get("task_results", []):
        task_id = int(tr["task_id"])
        if args.task is not None and task_id != args.task:
            continue
        for er in tr.get("episode_results", []):
            episode_idx = int(er["episode"])
            if args.episode is not None and episode_idx != args.episode:
                continue
            success = bool(er.get("success"))
            for sub, ext, root in present_roots:
                line = _rename_media(root, ext, task_id, episode_idx, success, args.dry_run)
                print(f"task_{task_id:03d}/ep{episode_idx:03d} success={success} [{sub}]: {line}")
                if "renamed" in line and not args.dry_run:
                    renamed += 1
                elif "skip" in line:
                    skipped += 1
    print(f"\nsummary: {renamed} renamed, {skipped} skipped, dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
