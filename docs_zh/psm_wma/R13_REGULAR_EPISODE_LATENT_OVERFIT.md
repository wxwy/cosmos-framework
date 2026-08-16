# R13 — Regular Episode Latent + LIBERO 单任务 Overfit

> 状态：实现口径冻结（2026-08-16）  
> 分支：`data/cache/action_r13_regular_episode_overfit`  
> 目标：不再以官方 window-local `z0` 作为当前视觉 condition；改为 MoWA-style 整 episode 因果 VAE 编码，保存 `z0,z1,...`，训练仅使用 regular latent `z1+`，先用单任务 tiny-overfit 验证 Cosmos3 是否能适应新的 condition latent 分布。

## 1. 本轮唯一研究问题

验证下述替换是否可行：

```text
Official Cosmos LIBERO
17 RGB window -> Wan VAE -> [z0, z1, z2, z3, z4]
condition = z0(single-frame prime)
future    = z1..z4(regular temporal latents)

R13
whole episode RGB -> causal Wan VAE -> [z0, z1, z2, ...]
condition = z_k
future    = z_{k+1}..z_{k+4}
k >= 1
```

其中 `z0` **仍然保存**，仅不进入 R13 正常训练样本。

本轮不同时引入 Local Memory、Global Spatial Memory、Planner、RL、额外 loss 或 streaming-VAE 性能优化。

## 2. 必须保持不变的部分

- Cosmos3 / OmniMoT 主体结构不改。
- LIBERO action 定义不改：frame-wise relative + rot6d + gripper。
- action chunk = 16。
- WAM loss / sampler 不改。
- 两相机仍按现有 Cosmos action-policy latent layout 进入模型。
- R12 exact-window cache 路径保留，不覆盖、不破坏，作为回退与 parity 参考。

## 3. Episode causal latent 的唯一时间定义

Wan2.2 temporal compression = 4：

```text
pixel frame
f0 | f1 f2 f3 f4 | f5 f6 f7 f8 | f9 f10 f11 f12 | ...
 |         |               |                 |
 v         v               v                 v
z0        z1              z2                z3
prime     regular         regular           regular
```

定义：

```text
z0        : prime latent，endpoint frame = 0
z_k(k>=1) : regular latent，endpoint frame = 4*k
```

缓存中必须显式保存 endpoint，而不能靠调用方猜测：

```python
endpoint_frame_indices = [0, 4, 8, 12, ...]
```

## 4. R13 cache artifact

建议每 episode 一个 `.pt`，沿用 R12 的工程风格：

```text
episode_000123.pt
{
  "format": "r13_regular_episode_latent_v1",
  "episode_index": 123,
  "frame_count": T,
  "temporal_compression_factor": 4,
  "views": ["image", "wrist_image"],
  "latents": {
      # 建议保持 view 维显式，不要在落盘时丢失相机语义
      "per_view": Tensor[V, Tz, C, H, W]
  },
  "indices": {
      "endpoint_frame_indices": Tensor[Tz]  # [0,4,8,...]
  },
  "metadata": {...}
}
```

约束：

- `per_view[:, 0]` 是 z0，必须保存。
- 正常 R13 dataset sampler 从 latent index 1 开始。
- 不再保存大量 overlapping 17-frame window latent。
- cache builder 对一个 episode 的每个相机只做一次 causal encode。
- VAE encode 必须从 episode frame 0 开始，不能把任意 5-frame sliding window 重新 prime 后冒充 regular latent。

## 5. 两相机训练 tensor

对 regular current latent index `k>=1`：

```text
camera 0: [z_k, z_k+1, z_k+2, z_k+3, z_k+4]
camera 1: [z_k, z_k+1, z_k+2, z_k+3, z_k+4]
```

拼接方式必须严格复用当前 R12 / OmniMoT 已验证的 camera-major layout：

```text
[cam0_t0..t4 | cam1_t0..t4]
```

最终 shape 保持当前 cached-LIBERO 模型入口：

```text
vision_latent_cache: [10, C, H, W]
```

不要在 R13 中自行发明新的 view 维 pack 方式。

## 6. Action 对齐：最重要的 off-by-one gate

对于 original Wan latent index `k>=1`：

```text
current latent z_k endpoint = frame e = 4*k
```

训练样本定义：

```text
condition : z_k
future    : z_{k+1}, z_{k+2}, z_{k+3}, z_{k+4}
action    : a_e, a_{e+1}, ..., a_{e+15}
```

即：

```text
frame endpoint e                    e+16
       |------------------------------|
       z_k  -> 16 actions ->          z_{k+4}
              ^                         ^
              a_e                       future endpoint
```

合法 sample 条件至少为：

```python
k >= 1
k + 4 < num_latents
4 * k + 16 <= episode_action_count
```

若 LIBERO action/frame 数存在末帧定义差异，以实际 parquet row count 做断言，严禁 silent clamp/pad。

## 7. Dataset 实现路线

### 7.1 新 cache reader

新增：

```text
cosmos_framework/data/generator/action/regular_episode_latent_cache.py
```

建议接口：

```python
class R13RegularEpisodeLatentCache:
    def get_training_window(
        self,
        episode_index: int,
        current_endpoint_frame: int,
        horizon_latents: int = 4,
    ) -> Tensor | None:
        """Return camera-major [10,C,H,W]; never return z0 as current."""
```

reader 必须校验：

- artifact format/version；
- endpoint index 精确命中；
- current latent index > 0；
- current + 4 future latent 全部存在；
- 两个 view 的 Tz 完全一致；
- 输出 `[10,C,H,W]`。

### 7.2 R13 dataset 不能继续按任意 RGB frame start 采样

R12 当前 dataset flat index 是任意 `window_start_frame=0,1,2,...`。

R13 必须改成 **regular endpoint aligned samples**：

```text
valid current endpoints = 4,8,12,...
```

因此给 `LIBEROLeRobotDataset` 增加显式 R13 sampling mode，而不是在 `get_window()` 内偷偷 round start index。

推荐参数：

```python
latent_cache_mode: Literal["r12_exact_window", "r13_regular_episode"]
```

R13 下：

```text
flat sample idx
 -> episode
 -> current endpoint e in {4,8,12,...}
 -> RGB/action source start = episode_start + e
 -> action = rows[e:e+16]
 -> latent = cache[z_k:z_k+5]
```

cache 命中时仍可像 R12 一样跳过 RGB decode，用 dummy video 维持 transform batch contract。

## 8. Cache builder

新增脚本：

```text
scripts/action/build_libero_r13_regular_episode_latents.py
```

命令行至少支持：

```text
--dataset-root
--output-root
--task-index
--episode-index (optional)
--image-size 256
--device cuda
--dtype bf16/fp32
--overwrite
--parity-samples
```

流程：

```text
LIBERO episode
 -> 读取全部连续 third-person RGB
 -> 读取全部连续 wrist RGB
 -> resize 256
 -> 分别从 frame0 开始做一次 Wan causal encode
 -> 得到每 view [z0,z1,...]
 -> 保存 z0 + regular latents + endpoint indices
 -> parity check
```

### parity 必须验证的对象

随机选择 `k>=1`：

```text
cached z_k
vs
从 episode frame0 重编码到 endpoint=4*k 后取得的 z_k
```

不得拿 `frames[4*k-4:4*k+1]` fresh encode 得到的局部 `z1'` 做 parity。

输出：

```text
parity.json
{
  "status": "PASS",
  "format": "r13_regular_episode_latent_v1",
  "max_abs_err": ...,
  "mean_abs_err": ...,
  "checked": ...
}
```

## 9. 单任务 tiny-overfit：直接作为第一实验

不再先额外训练一版 official per-window baseline；R12 只作为已有代码回退路径。

### 9.1 数据

- LIBERO 单 task：默认 `task_index=0`，但配置可覆写。
- tiny subset：先 16 个 aligned samples。
- 固定 subset，不 shuffle；确保每次看到完全相同的 16 个样本。
- cache 只需要先构建覆盖这批样本所在 episode；若 builder 按 task 构建全部 episode 也可。

### 9.2 配置

新增：

```text
cosmos_framework/configs/base/experiment/action/posttrain_config/
  action_policy_libero_nano_r13_regular_latent_overfit.py
```

在官方 / 当前 LIBERO Nano config 基础上只改：

```text
task_index = 0
tiny_overfit_num_samples = 16
iterable_shuffle = False
use_latent_cache = True
latent_cache_mode = r13_regular_episode
latent_cache_root = <R13 cache>
cfg_dropout_rate = 0  # overfit 诊断避免随机条件 dropout
```

其余模型、loss、action normalization、chunk_length=16 保持不变。

### 9.3 第一阶段不追求 closed-loop SR

先只看模型能否吃下新的 regular condition distribution。

Gate A：数据正确

```text
- z0 存在，但任何训练 sample current index 都不是 0
- current endpoint 均为 4 的倍数且 >=4
- action[0] 对应 endpoint frame e 的 parquet action
- future 最后一 latent endpoint = e+16
- 两相机 pack 顺序与 R12 相同
```

Gate B：训练可过拟合

至少记录：

```text
- total loss
- action loss
- vision/world loss
```

成功标准优先使用“明显、持续地向接近 0 收敛”，而不是预先拍死某个绝对阈值；实现后再根据现有 loss 尺度冻结数字阈值。

Gate C：动作记忆

在训练的 16 个固定样本上离线采样 action，反归一化后比较 target：

```text
translation error
rotation error
gripper accuracy
```

只有 Gate A/B/C 通过，才进入 closed-loop。

## 10. R13 closed-loop correctness oracle

首版推理不优先做 persistent encoder-cache 工程优化。

第一次 policy：

```text
收集 f0..f4
 -> 从 f0 开始 encode
 -> 得 z0,z1
 -> 用 z1 做 current condition
```

之后每次 policy request，为保证与 episode cache **完全相同的因果语义**，允许 correctness-first：

```text
保存本 episode 已观测 RGB history
 -> 从 f0 重编码到当前 endpoint
 -> 取最新 regular z_k
 -> policy
```

它慢，但单任务 overfit 阶段用来查正确性最安全。

R13 overfit + closed-loop SR>0 后，再把 oracle 替换为：

```text
persistent Wan encoder state
push_frames(4n frames)
get_latest_regular_latent()
```

优化前后必须做 latent parity，不能只看 shape。

## 11. 必须新增的测试

```text
tests/data/generator/action/test_r13_regular_episode_latent_cache.py
```

至少：

1. `test_prime_latent_is_saved_but_never_selected_as_current`
2. `test_regular_endpoint_mapping_z1_is_frame4`
3. `test_training_window_has_five_latents_per_view`
4. `test_two_view_camera_major_pack_order`
5. `test_action_starts_at_current_latent_endpoint`
6. `test_future_last_endpoint_matches_action_horizon`
7. `test_reject_unaligned_current_endpoint`
8. `test_reject_window_past_episode_end`

GPU/integration test：

9. `test_cached_regular_latent_matches_full_episode_prefix_encode`

## 12. 实现顺序

```text
R13-1  cache artifact schema + reader + CPU unit tests
  ↓
R13-2  whole-episode cache builder + VAE parity
  ↓
R13-3  LIBERO aligned regular-endpoint sampler
  ↓
R13-4  R13 Nano single-task tiny-overfit config
  ↓
R13-5  16-sample loss/action memorization
  ↓
R13-6  closed-loop full-history re-encode oracle
  ↓
R13-7  same-task SR > 0
  ↓
R13-8  persistent streaming VAE（只做性能优化，latent parity 必须保持）
  ↓
后续 Local Memory / Global Spatial Memory
```

## 13. Stop / rollback 条件

以下任一发生，不继续叠加 Memory/Spatial：

- cache parity 不通过；
- z/action 对齐测试不通过；
- tiny-overfit loss 无法显著下降；
- 能 memorization 但 closed-loop 仍 0-SR，且确认是 regular-latent condition distribution 导致。

这时直接切回保留的 R12 exact-window 路径定位问题，不用回滚或覆盖 R13 artifact。

## 14. 本轮最终判断标准

R13 的目的不是证明 episode cache 更快——这一点工程上几乎确定；真正要回答的是：

> **Cosmos3 action policy 能否在很小的 SFT/overfit 成本下，把“当前 condition = single-frame prime z0”适应成“当前 condition = causal regular z_k”，同时保持 future latent 与 action 的联合生成能力。**

单任务 overfit 是成本最低、归因最干净的第一验证。
