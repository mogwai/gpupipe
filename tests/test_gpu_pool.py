"""Tests for per-stage GPU pinning via add(gpus=[...]).

Each worker reports the PHYSICAL device it was pinned to; pinning is correct iff the
set of devices the stage ran on equals the requested pool. A process worker isolates
itself with CUDA_VISIBLE_DEVICES=<one physical id>, so torch.cuda.current_device() is
always logical 0 inside it — we read the env var to recover the physical id instead.
"""
import os
import time

import pytest
import torch
from conftest import Collector

from pipe import End, Pipe

HAS_CUDA = torch.cuda.is_available()
N_GPU = torch.cuda.device_count() if HAS_CUDA else 0


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


class GpuReporter:
    """Stamps the CUDA device this worker was pinned to."""

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(0.004)  # spread work so every worker in the pool gets some
        if not HAS_CUDA:
            item["device"] = -1
            return item
        # The process worker pinned itself to exactly one physical GPU by rewriting
        # CUDA_VISIBLE_DEVICES; current_device() is logical 0, so read the env to get
        # the physical id we actually landed on.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        item["device"] = int(cvd.split(",")[0]) if cvd else torch.cuda.current_device()
        return item


def _run(stages):
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    for s, k in stages:
        pipe.add(s, **k)
    return list(pipe)


def _devices_for(pool, n=160, workers=1):
    res = _run([
        (Gen(n), {"workers": 1, "outqn": 64}),
        (GpuReporter(), {"gpus": pool, "workers": workers, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert len(res) == n, f"{len(res)} != {n}"
    return {r["device"] for r in res}


@pytest.mark.skipif(N_GPU < 2, reason="need >=2 GPUs")
def test_gpus_pool_pins_listed_devices():
    assert _devices_for([0, 1]) == {0, 1}


@pytest.mark.skipif(N_GPU < 8, reason="need 8 GPUs")
def test_gpus_pool_noncontiguous():
    assert _devices_for([3, 5, 7]) == {3, 5, 7}


@pytest.mark.skipif(N_GPU < 2, reason="need >=2 GPUs")
def test_gpus_pool_workers_multiplier():
    # workers=2 over a 2-GPU pool => 4 workers, 2 per GPU; still only those 2 devices
    assert _devices_for([0, 1], workers=2) == {0, 1}


@pytest.mark.skipif(N_GPU < 1, reason="need a GPU")
def test_single_gpu_id_pins():
    g = N_GPU - 1
    res = _run([
        (Gen(60), {"workers": 1, "outqn": 64}),
        (GpuReporter(), {"gpu_id": g, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert {r["device"] for r in res} == {g}


# --- edge cases (validation at add() time; no spawning needed) ---
def _fresh_pipe():
    return Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)


def test_gpus_out_of_range_raises():
    """Requesting a GPU that doesn't exist fails fast at add(), not in a worker."""
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)  # root
    with pytest.raises(ValueError, match="only|available|GPU"):
        p.add(GpuReporter(), gpus=[N_GPU + 5])


def test_gpu_id_out_of_range_raises():
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError):
        p.add(GpuReporter(), gpu_id=N_GPU + 3)


def test_empty_gpus_raises():
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError, match="empty"):
        p.add(GpuReporter(), gpus=[])


def test_negative_gpu_raises():
    p = _fresh_pipe()
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError):
        p.add(GpuReporter(), gpus=[-1])


def test_explicit_gpus_no_cuda_raises():
    """An EXPLICIT gpus= request on a box with no GPUs raises — no silent CPU
    degrade (simulated by forcing gpu_count=0)."""
    p = _fresh_pipe()
    p.gpus = 0  # pretend no CUDA
    p.add(Gen(1), workers=1)
    with pytest.raises(ValueError, match="no CUDA GPUs|CUDA GPU"):
        p.add(GpuReporter(), gpus=[0, 1], workers=2)


def test_pergpu_no_cuda_falls_back_to_cpu():
    """pergpu=True is the only adaptive mode: 0 GPUs -> CPU fallback (with a loud
    warning, not silent)."""
    p = _fresh_pipe()
    p.gpus = 0
    p.add(Gen(1), workers=1)
    p.add(GpuReporter(), pergpu=True, workers=2)
    job = p.jobs[-1]
    assert job["gpus"] is None and not job["is_gpu_stage"]
    assert job["num_workers"] == 2


# --- logical->physical resolution (no GPUs needed) ---
# A worker pins itself by rewriting CUDA_VISIBLE_DEVICES to one device. When the
# process was LAUNCHED with a restricted CUDA_VISIBLE_DEVICES, its logical gpu ids
# must be resolved back through that visible set — otherwise `CVD=4,5,6,7 pergpu=True`
# pins workers onto physical 0..3 (the excluded cards).
@pytest.mark.parametrize("gpu_id,inherited,expected", [
    (0, "4,5,6,7", "4"),   # regression: was "0"
    (3, "4,5,6,7", "7"),
    (1, "2,3", "3"),
    (0, None, "0"),        # unrestricted -> logical == physical
    (5, "", "5"),
    (2, "0,1,2,3,4,5,6,7", "2"),
    (0, "6", "6"),         # single inherited device
    (0, " 4 , 5 ", "4"),   # tolerate whitespace
    (9, "4,5", "9"),       # out-of-range -> fall back to raw id, don't crash
])
def test_resolve_physical_gpu(gpu_id, inherited, expected):
    from pipe.workers import _resolve_physical_gpu
    assert _resolve_physical_gpu(gpu_id, inherited) == expected


@pytest.mark.skipif(N_GPU < 1, reason="need a GPU")
def test_duplicate_gpu_oversubscribes():
    """gpus=[g,g] is allowed: two workers pinned to the same GPU."""
    g = N_GPU - 1
    res = _run([
        (Gen(80), {"workers": 1, "outqn": 64}),
        (GpuReporter(), {"gpus": [g, g], "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert len(res) == 80
    assert {r["device"] for r in res} == {g}


@pytest.mark.skipif(N_GPU < 2, reason="need >=2 GPUs")
def test_pergpu_spreads_across_gpus():
    # pergpu == gpus=range(N): every worker pinned to a distinct GPU. (Which GPUs
    # actually get work depends on scheduling/startup, so assert valid + multiple
    # rather than exactly-all, which races on slow CUDA-context startup.)
    res = _run([
        (Gen(400), {"workers": 1, "outqn": 64}),
        (GpuReporter(), {"pergpu": True, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ])
    assert len(res) == 400
    seen = {r["device"] for r in res}
    assert seen <= set(range(N_GPU)) and len(seen) >= 2, seen
