# Regular Episode Latent + LIBERO 单任务 Overfit

> 状态：实现口径冻结（2026-08-16）  
> 分支：`experiment/libero_regular_episode_latent_overfit`  
> 目标：不再以官方 window-local `z0` 作为当前视觉 condition；改为 MoWA-style 整 episode 因果 VAE 编码，保存 `z0,z1,...`，训练仅使用 regular latent `z1+`，先用单任务 tiny-overfit 验证 Cosmos3 是否能适应新的 condition latent 分布。

## 1. 本轮唯一研究问题

```text
Official Cosmos LIBERO
17 RGB window -> Wan VAE -> [z0, z1, z2, z3, z4]
condition = z0(single-frame prime)
future    = z1..z4(regular temporal latents)

Regular Episode Latent
whole episode RGB -> causal Wan VAE -> [z0, z1, z2, ...]
condition = z_k
future    = z_{k+1}..z_{k+4}
k >= 1
```

其中 `z0` 仍然保存，但不进入正常训练样本。

本轮不同时引入 Local Memory、Global Spatial Memory、Planner、RL、额外 loss 或 streaming-VAE 性能优化。

## 2. 保持不变

- Cosmos3 / OmniMoT 主体结构不改。
- LIBERO action：frame-wise relative + rot6d + gripper。
- action chunk = 16。
- WAM loss / sampler 不改。
- 两相机沿用当前 Cosmos action-policy latent layout。
- 现有 exact-window cache 路径保留，作为回退与 parity 参考。

## 3. Episode causal latent 时间定义

Wan2.2 temporal compression = 4：

```text
pixel frame
f0 | f1 f2 f3 f4 | f5 f6 f7 f8 | f9 f10 f11 f12 | ...
 |         |               |                 |
 v         v               v                 v
z0        z1              z2                z3
prime     regular         regular           regular
```

```text
z0        : prime latent，endpoint frame = 0
z_k(k>=1) : regular latent，endpoint frame = 4*k
```

缓存中显式保存：

```python
endpoint_frame_indices = [0, 4, 8, 12, ...]
```

## 4. Cache artifact

每 episode 一个 `.pt`：

```text
episode_000123.pt
{
  "format": "regular_episode_latent_v1",
  "episode_index": 123,
  "frame_count": T,
  "temporal_compression_factor": 4,
  "views": ["image", "wrist_image"],
  "latents": {
      "per_view": Tensor[V, Tz, C, H, W]
  },
  "indices": {
      "endpoint_frame_indices": Tensor[Tz]
  },
  "metadata": {...}
}
```

约束：

- `per_view[:, 0]` 是 z0，必须保存。
- 正常 dataset sampler 从 latent index 1 开始。
- 不保存大量 overlapping 17-frame window latent。
- 每个 episode 的每个相机只做一次从 frame0 开始的 causal encode。

## 5. 两相机训练 tensor

对 current latent index `k>=1`：

```text
camera 0: [z_k, z_k+1, z_k+2, z_k+3, z_k+4]
camera 1: [z_k, z_k+1, z_k+2, z_k+3, z_k+4]
```

严格复用当前已验证 camera-major layout：

```text
[cam0_t0..t4 | cam1_t0..t4]
```

最终：

```text
vision_latent_cache: [10, C, H, W]
```

## 6. Action 对齐

对于 original Wan latent index `k>=1`：

```text
current latent z_k endpoint = frame e = 4*k
condition : z_k
future    : z_{k+1}, z_{k+2}, z_{k+3}, z_{k+4}
action    : a_e, a_{e+1}, ..., a_{e+15}
```

合法 sample：

```python
k >= 1
k + 4 < num_latents
4 * k + 16 <= episode_action_count
```

末端严格按 parquet row count 断言，不做 silent clamp/pad。

## 7. Dataset 实现

新增：

```text
cosmos_framework/data/generator/action/regular_episode_latent_cache.py
```

接口：

```python
class RegularEpisodeLatentCache:
    def get_training_window(
        self,
        episode_index: int,
        current_endpoint_frame: int,
        horizon_latents: int = 4,
    ) -> Tensor | None:
        """Return camera-major [10,C,H,W]; never return z0 as current."""
```

reader 校验：

- artifact format/version；
- endpoint 精确命中；
- current latent index > 0；
- current + 4 future latents 全部存在；
- 两个 view 的 Tz 一致；
- 输出 `[10,C,H,W]`。

R13 风格的版本编号不进入接口命名。

### aligned sampling

不能继续按任意 RGB frame start 采样。有效 current endpoint：

```text
4, 8, 12, 16, ...
```

建议配置：

```python
latent_cache_mode: Literal["exact_window", "regular_episode"]
```

`regular_episode` 下：

```text
flat sample idx
 -> episode
 -> current endpoint e in {4,8,12,...}
 -> action source start = episode_start + e
 -> action = rows[e:e+16]
 -> latent = cache[z_k:z_k+5]
```

cache 命中时可以跳过 RGB decode，用 dummy video 维持 transform batch contract。

## 8. Cache builder

新增：

```text
scripts/action/build_libero_regular_episode_latents.py
```

至少支持：

```text
--dataset-root
--output-root
--task-index
--episode-index
--image-size 256
--device cuda
--dtype bf16/fp32
--overwrite
--parity-samples
```

流程：

```text
LIBERO episode
 -> 连续 third-person RGB
 -> 连续 wrist RGB
 -> resize 256
 -> 两个 view 分别从 frame0 做 causal encode
 -> 得到每 view [z0,z1,...]
 -> 保存 latents + endpoint indices
 -> parity check
```

parity 比较：

```text
cached z_k
vs
从 episode frame0 重编码到 endpoint=4*k 后取得的 z_k
```

不能拿局部 5-frame fresh encode 的结果替代。

## 9. 第一实验：LIBERO 单任务 tiny-overfit

不额外训练一版 exact-window baseline；它只作为已有回退路径。

### 数据

- 单 task，默认 `task_index=0`。
- 16 个 aligned samples。
- 固定 subset，不 shuffle。

### 配置

新增：

```text
cosmos_framework/configs/base/experiment/action/posttrain_config/
  action_policy_libero_nano_regular_latent_overfit.py
```

主要修改：

```text
task_index = 0
tiny_overfit_num_samples = 16
iterable_shuffle = False
use_latent_cache = True
latent_cache_mode = regular_episode
latent_cache_root = <cache>
cfg_dropout_rate = 0
```

其余模型、loss、action normalization、chunk_length=16 保持不变。

## 10. 实验 gate

Gate A：数据正确

```text
- z0 被保存，但训练 current 永远不是 z0
- current endpoint 为 4 的倍数且 >=4
- action[0] 对应 endpoint frame e
- future 最后一 latent endpoint = e+16
- 两相机 pack 顺序正确
```

Gate B：loss overfit

记录：

```text
total loss
action loss
vision/world loss
```

要求持续明显下降并接近训练集记忆状态。

Gate C：动作记忆

训练的 16 个样本上离线采样 action，反归一化比较：

```text
translation error
rotation error
gripper accuracy
```

Gate A/B/C 全部通过后再跑闭环。

## 11. Closed-loop correctness oracle

首版不先做 persistent encoder cache。

第一次 policy：

```text
收集 f0..f4
 -> 从 f0 encode
 -> 得 z0,z1
 -> 用 z1 做 current condition
```

之后 correctness-first：

```text
保存本 episode 已观测 RGB history
 -> 从 f0 重编码到当前 endpoint
 -> 取最新 regular z_k
 -> policy
```

闭环 SR > 0 后，再优化为：

```text
persistent Wan encoder state
push new frames
get latest regular latent
```

优化前后必须做 latent parity。

## 12. 测试

新增：

```text
tests/data/generator/action/test_regular_episode_latent_cache.py
```

至少覆盖：

1. prime latent 保存但不能作为 current
2. `z1 -> frame4`
3. 每 view current+4 future
4. camera-major pack 顺序
5. action 从 current endpoint 开始
6. future endpoint 与 16-step action horizon 对齐
7. 拒绝非 4 对齐 endpoint
8. 拒绝越界 window
9. cached regular latent 与 full-episode prefix encode parity

## 13. 实现顺序

```text
cache schema + reader
  ↓
whole-episode builder + parity
  ↓
LIBERO aligned sampler
  ↓
Nano single-task overfit config
  ↓
16-sample loss/action memorization
  ↓
closed-loop full-history re-encode oracle
  ↓
same-task SR > 0
  ↓
persistent streaming VAE optimization
  ↓
Local Memory / Global Spatial Memory
```

## 14. Stop / rollback

以下任一发生，不叠加 Memory/Spatial：

- cache parity 不通过；
- latent/action 对齐不通过；
- tiny-overfit loss 无法显著下降；
- 可 memorization 但 closed-loop 仍 0-SR，且确认问题来自 regular-latent condition distribution。

## 15. 最终判断标准

> Cosmos3 action policy 能否以很小的 SFT/overfit 成本，把 `current condition = single-frame prime z0` 适应为 `current condition = causal regular z_k`，同时保持 future latent 与 action 的联合生成能力。
