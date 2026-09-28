# PSM-WMA V3 H3-E H100 B1 Episode-Level H5 Cache Builder — 验证报告与全量 Build 计划（2026-09-28）

- 日期：2026-09-28
- 提交方：ds（execution/validation only）
- 依据：`docs/build/PSM-WMA_V3_h3e_h100_data_asset_path_decision_2026-09-28.md` §4/§5（B1 frozen H5 contract + tiny PASS 后 GPU 全量授权）
- 范围：B1 数据资产工具；本次将已验证 builder/test 正式落入 `tools/v3/`，不修改 production loader / Local model / H3-E 训练语义>
> **2026-09-28 canonical source 路径变更备注**：canonical source renamed by owner; filesystem mv only; dataset identity unchanged。
> 原 `/mnt/data1/data_v2_0617/robocasa365_official_v30` 经 `mv` 更名为 `/mnt/data1/data_v2_0617/robocasa365_official_v30`，数据内容及冻结身份未变
> （18 atomic task classes / 9126 full episodes / 2231347 frames / fps=20 / LeRobot v3.0 / train split → 9036 / raw15 / chunk32 / left_wrist 不变）。
> 本报告所有 `robocasa365_official_v30` 引用即指向该 canonical 目录。

---

## 1. Builder 工具

- 正式位置：`tools/v3/build_robocasa_b1_h5_cache.py`（由 ds `/tmp` 验证稿落仓）
- 职责：按冻结 B1 contract 生成 episode-level H5 cache，并内置 `RoboCasaLatentReader` round-trip 校验
- 支持：
  - `--mode static`（CPU 随机 latent，schema 验证） / `--mode build`（真实 Wan2.2 VAE encode）
  - `--workers/--worker`（record 序取模分片，8 卡并行互斥）
  - 每 episode 写出后调用 `verify_episode()`（`RoboCasaLatentReader` 全契约校验）
- 关键修复（均属 `/tmp` 工具，非项目代码）：
  1. episodes parquet 相机列检查改为 `videos/{cam}/chunk_index` 全路径
  2. tqdm `_lock` AttributeError → `enable_fast_init=False`（规避 `_parallel_map` 线程池）
  3. VAE `scale=(mean,1/std)` 默认落在 cuda:0 → 随模型 `.to(device)` 一同搬移

## 2. B1 H5 冻结契约（每文件）

- 根 attrs：`episode_id`、`frame_count`、`temporal_compression_factor=4`、`source_frame_to_latent_policy="causal_endpoint"`
- 两路相机：`observation.images.robot0_agentview_left` + `observation.images.robot0_eye_in_hand`
- 每路：
  - `latents/`：fp16 `[N,48,16,16]`
  - `indices/latent_source_frame_indices/`：`0,4,8,...` + 必要 terminal `F-1`
  - `valid/`：bool `[N]` all true
- endpoint 向量：`E = list(range(0, F, 4))`，末位非 `F-1` 时追加 `F-1`；两路完全相等

## 3. 验证结果（全部 PASS）

### 3.1 CPU/static schema round-trip（3 episode，CloseFridge）

| Episode | frames | endpoints | latents fp16 [N,48,16,16] ×2cam | valid | summaries [N,96] |
| --- | --- | --- | --- | --- | --- |
| ep_000067 | 152 | 39 | ✓ | all True | finite |
| ep_000164 | 439 | 111 | ✓ | all True | finite |
| ep_000271 | 351 | 89 | ✓ | all True | finite |

- 产物：`/tmp/b1_static_cache/CloseFridge/20250816/lerobot/ep_*.h5`
- `records_total = 9036` 与冻结 manifest 一致

### 3.2 tiny 真实 build（3 episode，真实 Wan2.2 VAE encode，GPU cuda:3）

- 产物：`/tmp/b1_tiny_cache/CloseFridge/20250816/lerobot/ep_*.h5`
- 耗时：20.9s（含数据集构建 ~8s + VAE 加载 ~7s + encode）
- `RoboCasaLatentReader` round-trip：3/3 PASS

### 3.3 全帧 causality 扫描（3 episode，每帧 source_step）

- `visual_summary(s)` 对全部 F 帧逐一校验：返回的 endpoint 与 `causal_endpoint_index(endpoint_indices, s)` 完全一致
- ep_000067 / ep_000164 / ep_000271：`causality_ok=True`，summaries finite

### 3.4 8-worker 分片互斥（CPU smoke）

- 8 worker × limit 4 = 32 文件全部不同，无重复、无遗漏

## 4. 全量 9036 Build 计划（待启动）

| 项 | 值 |
| --- | --- |
| 目标路径 | `/mnt/data1/data_v2_0617/robocasa365_official_v30_wan2.2vae_latent_b1/`（`<cache>/<task>/<date>/lerobot/ep_XXXXXX.h5`） |
| 磁盘 | /mnt/data1 剩余 344G，预估 cache ~36GB |
| 并行 | 8 进程 × `--worker N --device cuda:N --workers 8`，每进程 ~1130 episode |
| 预估耗时 | 单卡稳态 ~2s/ep → 每 worker ~40min；8 卡并行墙钟 ~40min–1.5h |
| 监控 | 轮询统计 `<cache>/**/ep_*.h5` 数量 → 9036 |
| 完成后验证 | 1) 全量 `RoboCasaLatentReader` 复验；2) manifest digest 重算对比 `a8cad3f053232b348ea155f15bf79c2c9cf807dedcf39b89e246b17e43f283df` |

- 启动命令（每 worker，`nohup` 后台）：```
PYTHONPATH=/mnt/data/shenzhen/szrobot/logs/.tmp_backup/psm_wma_v3/cosmos-framework:/home/robo-shenzhen/.cache/uv/git-v0/checkouts/3854d3ea6f0ea07f/1a4316c/src \
/mnt/data/shenzhen/szrobot/logs/.tmp_backup/psm_wma/cosmos-framework/.venv/bin/python \
tools/v3/build_robocasa_b1_h5_cache.py \
  --source-root /mnt/data1/data_v2_0617/robocasa365_official_v30 \
  --cache-root /mnt/data1/data_v2_0617/robocasa365_official_v30_wan2.2vae_latent_b1 \
  --mode build --vae-path /mnt/data/shenzhen/szrobot/logs/.tmp_backup/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --device cuda:N --workers 8 --worker N \
  --report /tmp/b1_full_report_w{N}.json
```

## 5. 风险提示

- 当前 8 卡全部被训练任务占用（各 14.5GB 已用，util 53–100%）；VAE 内存占用低（~1-2GB）不会 OOM，但会在高负载卡上与训练任务争抢 compute，可能拖慢他人任务。用户已明确选择 8 卡并行。
- 全量 build 尚未启动（用户要求"先汇报再启动"）。

## 6. 阶段边界

- 未启动全量 build、未创建新 manifest、未修改 production code。
- 下一步：用户确认后启动 8 卡并行全量 build → 完成后全量复验 + manifest digest 重算。
