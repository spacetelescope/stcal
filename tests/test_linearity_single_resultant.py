import numpy as np

from stcal.linearity.linearity import linearity_correction

DQFLAGS = {
    "GOOD": 0,
    "DO_NOT_USE": 1,
    "SATURATED": 2,
    "DEAD": 1024,
    "HOT": 2048,
    "NO_LIN_CORR": 1048576,
}


def test_read_level_correction_single_resultant():
    """A single resultant uses the classic correction instead of estimating a slope."""
    nints, ngroups, nrows, ncols = 1, 1, 1, 1
    data = np.full((nints, ngroups, nrows, ncols), 10.0, dtype=np.float32)
    gdq = np.zeros_like(data, dtype=np.uint32)
    pdq = np.zeros((nrows, ncols), dtype=np.uint32)
    lin_dq = np.zeros((nrows, ncols), dtype=np.uint32)

    lin_coeffs = np.zeros((3, nrows, ncols), dtype=np.float32)
    lin_coeffs[:, 0, 0] = [0.0, 1.0, 0.01]
    ilin_coeffs = np.zeros((3, nrows, ncols), dtype=np.float32)
    ilin_coeffs[:, 0, 0] = [0.0, 1.0, -0.01]

    corrected, _, _ = linearity_correction(
        data,
        gdq,
        pdq,
        lin_coeffs,
        lin_dq,
        DQFLAGS,
        ilin_coeffs=ilin_coeffs,
        read_pattern=[[1]],
        satval=np.full((nrows, ncols), 65000.0, dtype=np.float32),
    )

    assert corrected[0, 0, 0, 0] == np.float32(11.0)
