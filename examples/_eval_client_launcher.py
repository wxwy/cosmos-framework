#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shim to run closed_loop_eval from the RLinf python (has LIBERO sim) while
falling back to cosmos-framework venv site-packages for pure-python deps that
the RLinf env lacks (loguru, ...).

Run with:  /disk/rl/RLinf/.venv/bin/python examples/_eval_client_launcher.py <closed_loop_eval args>

The cosmos venv site-packages is appended at the END of sys.path so RLinf's own
(native, py3.11) packages always win; only modules absent there resolve to the
cosmos venv.
"""
import runpy
import sys

REPO = "/disk/rl/psm_wma/cosmos-framework"
COSMOS_SP = f"{REPO}/.venv/lib/python3.13/site-packages"

# cosmos_framework package itself comes from the repo (PYTHONPATH), NOT from the
# cosmos venv's site-packages copy.
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# cosmos venv site-packages last -> fallback only (pure-python modules).
if COSMOS_SP not in sys.path:
    sys.path.append(COSMOS_SP)

if __name__ == "__main__":
    sys.argv = ["closed_loop_eval", *sys.argv[1:]]
    runpy.run_module("cosmos_framework.simulation.libero.closed_loop_eval", run_name="__main__")
