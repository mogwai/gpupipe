"""Smoke-test an installed wheel: its Rust extension, then a CPU pipeline.

cibuildwheel runs this against every wheel it builds, without torch, so it
also checks that a CPU pipeline neither needs nor imports torch. A wheel
that lost its extension fails here instead of shipping unable to start a
pipeline.
"""
import glob
import importlib.util
import os
import sys

import numpy as np


class Source:
    def __call__(self):
        for i in range(20):
            yield {"id": i, "wave": np.full(1 << 16, i, dtype=np.float32)}  # big enough for the store


class Double:
    def __call__(self, item):
        item["sum"] = float(item["wave"].sum()) * 2
        return item


def check_extension():
    spec = importlib.util.find_spec("pipe")
    pkg_dir = spec.submodule_search_locations[0]
    found = glob.glob(os.path.join(pkg_dir, "_rustq*"))
    if not found:
        sys.exit(f"pipe._rustq missing from {pkg_dir}: the Rust extension was not built")
    from pipe import _rustq

    q = _rustq.RingQueue(4, 1 << 20)
    try:
        for msg in (b"small", b"x" * (600 << 10)):  # ring fast path, then a spill
            assert q.put(msg, True, 1.0) and q.get(True, 1.0) == msg
        assert q.get(False) is None
    finally:
        q.unlink()

    st = _rustq.Store(64 << 20, 4 << 20)
    try:
        blk = st.alloc(1 << 20)
        memoryview(blk)[:3] = b"abc"
        assert bytes(memoryview(st.adopt(*blk.share()))[:3]) == b"abc"
    finally:
        st.unlink()
    return found[0]


def check_pipeline():
    from pipe import Pipe

    pipe = Pipe(stats_interval=0)
    pipe.add(Source(), outqn=4)
    pipe.add(Double(), workers=2, outqn=4)
    got = sorted((x["id"], x["sum"]) for x in pipe)
    assert got == [(i, 2.0 * i * (1 << 16)) for i in range(20)], got
    assert "torch" not in sys.modules, "a CPU pipeline imported torch"


if __name__ == "__main__":
    os.environ.setdefault("PIPE_DRAIN_GRACE", "0.2")
    ext = check_extension()
    check_pipeline()
    print(f"ok: {ext} on Python {sys.version.split()[0]}")
