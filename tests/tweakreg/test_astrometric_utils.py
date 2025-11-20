import contextlib
import time

import psutil
import pytest

from stcal.tweakreg.astrometric_utils import get_catalog as gc
from stcal.tweakreg._s3_catalog import get_catalog as s3gc


class TrackNetUsage:
    def __enter__(self):
        self.baseline = psutil.net_io_counters().bytes_recv

    def __exit__(self, *args, **kwargs):
        self.used = psutil.net_io_counters().bytes_recv - self.baseline

    def to_human(self):
        return psutil._common.bytes2human(self.used)


@pytest.mark.parametrize("func", [gc, s3gc], ids=["gsss", "s3"])
@pytest.mark.parametrize("sr", [0.01, 0.1, 0.2, 0.5])
def test_catalog_query_performance(func, sr):
    ra, dec = 268, 29
    net = TrackNetUsage()
    t0 = time.monotonic()
    with net:
        t = func(ra, dec, epoch=None, search_radius=sr)
    t1 = time.monotonic()
    dt = t1 - t0
    msg = f"N={len(t)}, Time={dt:0.3f} seconds, Net={net.to_human()}"
    assert False, msg
