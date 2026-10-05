from __future__ import annotations

import numpy as np
import pytest

from cosmos_framework.simulation.robocasa.eval_utils import canonicalize_raw15_for_env, decode_15d_to_env12


@pytest.mark.parametrize("mode", (-0.1, 0.1))
@pytest.mark.parametrize("flip", (False, True))
def test_canonical_raw15_redecodes_exact_submitted_command(mode: float, flip: bool) -> None:
    predicted = np.array(
        [2.0, -2.0, 0.7, -1.4, mode, 0.1, -0.2, 0.3, 1.0, 0.2, 0.1, 0.3, 1.0, -0.2, 2.5],
        dtype=np.float32,
    )
    submitted, canonical = canonicalize_raw15_for_env(predicted, flip)
    np.testing.assert_array_equal(submitted, decode_15d_to_env12(predicted, flip))
    np.testing.assert_allclose(decode_15d_to_env12(canonical, flip), submitted, rtol=0, atol=2e-6)
    assert canonical.shape == (15,)
    assert canonical[4] in (-1.0, 1.0)
    assert abs(canonical[14]) <= 1.0
    if mode < 0:
        np.testing.assert_array_equal(canonical[:4], np.zeros(4))
    else:
        np.testing.assert_array_equal(canonical[:4], np.array([1.0, -1.0, 0.7, -1.0], dtype=np.float32))


def test_canonical_rejects_nonfinite_or_wrong_width() -> None:
    with pytest.raises(ValueError, match="15D"):
        canonicalize_raw15_for_env(np.zeros(12), False)
    with pytest.raises(ValueError, match="有限"):
        canonicalize_raw15_for_env(np.full(15, np.nan), False)
