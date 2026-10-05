# Phase6A CPU/static/debug 记录（2026-10-05）

- 基线：`afb9ca8`；本阶段仅在 `phase6_cx` 工作树实现 corrected composite current Encode1、canonical raw15、完成动作后证据、Local-TTT 事务及顺序分块。
- CPU 验收：focused Phase6/既有在线 Local/评估合同 45 项通过；服务端补充 raw15/state/prompt 后对应 5 项通过。
- 静态验收：修改文件 Ruff、format、`py_compile`、`git diff --check` 通过；AST 测试禁止 corrected active modules 导入历史 B1 causal/latent 模块。
- 环境：使用已有 `/disk/rl/psm_wma/cosmos-framework/.venv`；本机 `uv run` 不识别 `tool.uv.audit`，首次尝试已中止。
- 边界：未运行真实 Wan 数值 parity、GPU、真实服务器、模拟器、18-task 或训练；Phase6B 与生产提升仍待单独授权。
