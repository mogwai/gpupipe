"""Tests for per-stage CPU affinity via add(cpus=[...]).

Each worker reports os.sched_getaffinity(0) (the core set it may run on) and
torch.get_num_threads(). Pinning is correct iff every worker's mask is exactly its
contiguous slice of the pool (and the thread count matches the slice size) — if
affinity silently failed the mask would be the whole machine instead.
"""
import os
import time

import pytest
import torch

from conftest import Collector
from pipe import Pipe, End
from pipe.workers import _cpu_chunk

N_CPU = os.cpu_count() or 1
HAS_AFFINITY = hasattr(os, "sched_getaffinity")
FULL_MACHINE = frozenset(range(N_CPU))


class Gen:
    def __init__(self, n):
        self.n = n
        self._i = 0

    def load(self):
        pass

    def __call__(self):
        if self._i >= self.n:
            return End
        it = {"id": self._i}
        self._i += 1
        return [it, End] if self._i >= self.n else it


class AffinityReporter:
    """Stamps the CPU set this worker is pinned to and its torch thread count."""

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(0.003)  # spread work so every worker in the pool gets some
        item["mask"] = tuple(sorted(os.sched_getaffinity(0)))
        item["nthreads"] = torch.get_num_threads()
        return item


def _run(stages):
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    for s, k in stages:
        pipe.add(s, **k)
    return list(pipe)


def _report(cpus, n=200, workers=1):
    res = _run([
        (Gen(n), {"workers": 1, "outqn": 64}),
        (AffinityReporter(), {"cpus": cpus, "workers": workers, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert len(res) == n, f"{len(res)} != {n}"
    return res


def _expected_masks(pool, workers):
    return {tuple(_cpu_chunk(pool, i, workers)) for i in range(workers)}


pytestmark = pytest.mark.skipif(not HAS_AFFINITY, reason="no os.sched_getaffinity (non-Linux)")


@pytest.mark.skipif(N_CPU < 4, reason="need >=4 cores")
def test_workers_pinned_to_their_slices():
    """4 workers over a 4-core pool => each pinned to exactly one distinct core."""
    pool = [0, 1, 2, 3]
    res = _report(pool, workers=4)
    seen = {r["mask"] for r in res}
    expected = _expected_masks(pool, 4)
    # every observed mask is one of the expected single-core slices...
    assert seen <= expected, f"unexpected masks: {seen - expected}"
    # ...and pinning actually happened (a failed setaffinity would report FULL_MACHINE)
    assert tuple(sorted(FULL_MACHINE)) not in seen
    # threads sized to the (1-core) slice
    assert {r["nthreads"] for r in res} == {1}


@pytest.mark.skipif(N_CPU < 8, reason="need >=8 cores")
def test_multicore_slice_sets_thread_count():
    """2 workers over an 8-core pool => each owns 4 contiguous cores, 4 threads."""
    pool = list(range(8))
    res = _report(pool, workers=2)
    seen = {r["mask"] for r in res}
    expected = _expected_masks(pool, 2)  # {(0,1,2,3), (4,5,6,7)}
    assert seen <= expected, f"unexpected masks: {seen - expected}"
    assert {r["nthreads"] for r in res} == {4}


@pytest.mark.skipif(N_CPU < 8, reason="need >=8 cores")
def test_noncontiguous_pool_is_respected():
    """A non-contiguous pool pins only to the listed cores, nothing else."""
    pool = [1, 3, 5, 7]
    res = _report(pool, workers=2)
    seen = {r["mask"] for r in res}
    for m in seen:
        assert set(m) <= set(pool), f"{m} leaked outside pool {pool}"
    assert seen <= _expected_masks(pool, 2)


@pytest.mark.skipif(N_CPU < 4, reason="need >=4 cores")
def test_pool_is_covered():
    """With enough work every worker runs, so the union of masks covers the pool.
    Loose (>=2 distinct) to avoid a startup-race flake like the GPU pool test."""
    pool = [0, 1, 2, 3]
    res = _report(pool, n=400, workers=4)
    union = set().union(*(r["mask"] for r in res))
    assert union <= set(pool)
    assert len({r["mask"] for r in res}) >= 2


@pytest.mark.skipif(N_CPU < 2, reason="need >=2 cores")
def test_single_worker_owns_whole_pool():
    pool = [0, 1]
    res = _report(pool, workers=1)
    assert {r["mask"] for r in res} == {(0, 1)}
    assert {r["nthreads"] for r in res} == {2}


@pytest.mark.skipif(N_CPU < 2, reason="need >=2 cores")
def test_more_workers_than_cores_oversubscribe():
    """workers > len(cpus): workers round-robin single cores (oversubscribed)."""
    pool = [0, 1]
    res = _report(pool, workers=4)
    seen = {r["mask"] for r in res}
    assert seen <= {(0,), (1,)}, seen
    assert {r["nthreads"] for r in res} == {1}


@pytest.mark.skipif(N_CPU < 2, reason="need >=2 cores")
def test_cpus_and_gpus_are_independent():
    """cpus= is stored/validated independently of GPU pinning; a CPU stage keeps
    is_gpu_stage False."""
    p = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    p.add(Gen(1), workers=1)
    p.add(AffinityReporter(), cpus=[0, 1])
    job = p.jobs[-1]
    assert job["cpus"] == [0, 1]
    assert job["is_gpu_stage"] is False


# --- validation at add() time (no spawning needed) ---
def _fresh_pipe():
    return Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)


def test_cpus_out_of_range_raises():
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError, match="CPU"):
        p.add(AffinityReporter(), cpus=[N_CPU + 5])


def test_empty_cpus_raises():
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError, match="empty"):
        p.add(AffinityReporter(), cpus=[])


def test_negative_cpu_raises():
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError):
        p.add(AffinityReporter(), cpus=[-1])


@pytest.mark.skipif(N_CPU < 2, reason="need >=2 cores")
def test_cpu_threads_sets_count_without_pinning():
    """cpu_threads lifts the flat 2-cap but does NOT pin (mask stays full machine)."""
    res = _run([
        (Gen(60), {"workers": 1, "outqn": 32}),
        (AffinityReporter(), {"cpu_threads": 6, "outqn": 32}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert {r["nthreads"] for r in res} == {6}
    assert {r["mask"] for r in res} == {tuple(sorted(FULL_MACHINE))}, "must stay unpinned"


def test_default_is_two_threads():
    """No cpus= / cpu_threads => the legacy 2-thread cap is preserved exactly."""
    res = _run([
        (Gen(40), {"workers": 1, "outqn": 32}),
        (AffinityReporter(), {"outqn": 32}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert {r["nthreads"] for r in res} == {2}


@pytest.mark.skipif(N_CPU < 8, reason="need >=8 cores")
def test_cpu_threads_overrides_slice_size():
    """When both are set, cpu_threads wins over the cpus= slice size; still pinned.
    The slice is 8 cores / 2 workers = 4, so a plain cpus= run would give 4 threads;
    cpu_threads=3 must override that to 3 while keeping the 4-core pin."""
    res = _run([
        (Gen(120), {"workers": 1, "outqn": 32}),
        (AffinityReporter(), {"cpus": list(range(8)), "workers": 2, "cpu_threads": 3, "outqn": 32}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert {r["nthreads"] for r in res} == {3}, "cpu_threads should override slice size"
    assert {len(r["mask"]) for r in res} == {4}, "still pinned to the 4-core slice"


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "4"])
def test_cpu_threads_validation(bad):
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError, match="cpu_threads"):
        p.add(AffinityReporter(), cpu_threads=bad)


# The autoscale/cpus interaction tests were removed with the autoscaling
# feature and are currently NOT covered anywhere that runs; restore them from
# git history when re-integrating (see PLANNED.md).
