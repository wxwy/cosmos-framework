#!/usr/bin/env python3
"""CPU encoder child dies when its Python encoder disappears."""

import ctypes
import os
import signal
import sys

if __name__ == "__main__":
    before = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG")
    if os.getppid() != before or before == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    executable = os.environ["PSM_EVAL_REAL_FFMPEG"]
    os.execv(executable, [executable, *sys.argv[1:]])
