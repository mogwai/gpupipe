"""Smoke-test an installed wheel's Rust extension without torch.

cibuildwheel runs this against every wheel it builds. Importing `pipe` pulls
in torch (GBs per test environment), so this loads `pipe._rustq` straight
from the installed package and exercises the queue and the payload store.
A wheel that silently lost its extension (setuptools-rust builds it as
optional) fails here instead of shipping as the slow mp.Queue fallback.
"""
import glob
import importlib.machinery
import importlib.util
import os
import sys

spec = importlib.util.find_spec("pipe")  # locates the package without running pipe/__init__
pkg_dir = spec.submodule_search_locations[0]
found = glob.glob(os.path.join(pkg_dir, "_rustq*"))
if not found:
    sys.exit(f"pipe._rustq missing from {pkg_dir}: the Rust extension was not built")
loader = importlib.machinery.ExtensionFileLoader("pipe._rustq", found[0])
ext = importlib.util.module_from_spec(importlib.util.spec_from_loader("pipe._rustq", loader))
loader.exec_module(ext)

q = ext.RingQueue(4, 1 << 20)
try:
    for msg in (b"small", b"x" * (600 << 10)):  # ring fast path, then a spill
        assert q.put(msg, True, 1.0) and q.get(True, 1.0) == msg
    assert q.get(False) is None
finally:
    q.unlink()

st = ext.Store(64 << 20, 4 << 20)
try:
    blk, _ = st.alloc(1 << 20)
    memoryview(blk)[:3] = b"abc"
    assert bytes(memoryview(st.adopt(*blk.share()))[:3]) == b"abc"
finally:
    st.unlink()

print(f"ok: {found[0]} on Python {sys.version.split()[0]}")
