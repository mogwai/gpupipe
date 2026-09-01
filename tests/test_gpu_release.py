"""A finished GPU worker must release its VRAM while the pipeline is still running.

Workers park after End until the whole pipe stops (so tensors they put on the
torch.mp queues stay valid). Only the process needs to survive for that. Before
the fix a parked GPU worker also kept its weights and the caching allocator's
whole reserved pool until the last stage finished -- 20-30 GB per card sitting
idle through every drain on the uvr2 pipe.

Shape: a fast GPU stage that pins ~1 GiB in load(), feeding a deliberately slow
CPU stage. The GPU stage finishes in well under a second and parks; the slow
stage keeps the pipe alive for several seconds. We sample the GPU worker's
memory from nvidia-smi (keyed by pid -- the only view of a child's usage from
outside it) while the pipe is still producing, and require that it drops to
context-only before the run ends. Unpatched, it stays at ~1.1 GiB throughout.
"""
import os
import shutil
import subprocess
import time

import pytest
import torch

from pipe import End, Pipe

HAS_CUDA = torch.cuda.is_available()
HAS_SMI = shutil.which("nvidia-smi") is not None

BLOB_BYTES = 1 << 30  # 1 GiB pinned on the card by the GPU stage
CONTEXT_ONLY_MIB = 900  # a bare primary context is ~500 MiB on Blackwell; leave slack
N_ITEMS = 14
SLOW_S = 0.35  # per item downstream => the pipe lives ~5 s after the GPU stage is done


class Gen:
    def __init__(self, n):
        self.n, self._i = n, 0

    def load(self):
        pass

    def __call__(self):
        if self._i >= self.n:
            return End
        it = {"id": self._i}
        self._i += 1
        return [it, End] if self._i >= self.n else it


class GpuHog:
    """Pins BLOB_BYTES for the worker's whole life; trivial per-item work."""

    def load(self):
        for _ in range(60):  # tolerate a busy card: retry the pin for ~a minute
            try:
                self._blob = torch.empty(BLOB_BYTES, dtype=torch.uint8, device="cuda")
                break
            except torch.cuda.OutOfMemoryError:
                time.sleep(1)
        else:
            raise RuntimeError("could not pin the test blob on the GPU")
        torch.cuda.synchronize()

    def __call__(self, item):
        item["touched"] = int(self._blob[item["id"]].item())
        return item


class GpuHogWithHook(GpuHog):
    """Same, but offers the lib's on_park() contract: the stage frees its own
    GPU state. This is the path a model stage should take (a few frees and an
    empty_cache()); the plain GpuHog case covers stages that never heard of it."""

    def on_park(self):
        del self._blob
        torch.cuda.empty_cache()
        return None


class GpuScratch:
    """No pinned state; each item allocates and drops ~1 GiB of transient GPU
    memory, which the caching allocator keeps. Models a stage whose live footprint
    is small but whose cached pool is large -- the separator between batches."""

    def load(self):
        pass

    def __call__(self, item):
        t = torch.empty(BLOB_BYTES, dtype=torch.uint8, device="cuda")
        t[item["id"]] = 1
        item["touched"] = int(t[item["id"]].item())
        del t  # freed to the caching allocator, NOT to the driver
        return item


class Slow:
    def __init__(self, delay=SLOW_S):
        self.delay = delay

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(self.delay)
        return item


def _mib_by_pid() -> dict[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    res = {}
    for line in out.splitlines():
        if "," in line:
            pid, mib = line.split(",")
            res[int(pid)] = int(mib)
    return res


def _run_and_sample(pipe, stage_name, period=0.2):
    """Iterate the pipe to completion while a thread samples the named GPU
    worker's memory from nvidia-smi. Returns (n_out, samples[(t, MiB)])."""
    import threading

    samples, stop = [], threading.Event()
    gpu_pid = [None]
    t0 = time.time()

    def sampler():
        while not stop.is_set():
            if gpu_pid[0] is None:
                for p, _wid, stage in pipe.worker_info:
                    if stage == stage_name:
                        gpu_pid[0] = p.pid
            if gpu_pid[0] is not None:
                samples.append((time.time() - t0, _mib_by_pid().get(gpu_pid[0])))
            time.sleep(period)

    th = threading.Thread(target=sampler, daemon=True)
    n_out = 0
    for _ in pipe:
        if not th.is_alive() and not stop.is_set():
            th.start()  # workers exist once the first item is out
        n_out += 1
    total = time.time() - t0
    stop.set()
    th.join(timeout=2)
    assert gpu_pid[0] is not None, f"{stage_name} worker pid not found in worker_info"
    return n_out, total, samples


@pytest.mark.skipif(not (HAS_CUDA and HAS_SMI), reason="needs a GPU and nvidia-smi")
@pytest.mark.parametrize("hog", [GpuHog, GpuHogWithHook], ids=["no-hook", "on_park"])
def test_parked_gpu_worker_releases_vram_before_pipeline_ends(hog):
    """Case 1: the GPU stage finishes and PARKS while the slow tail keeps the pipe
    alive. Its pinned state must go at the park, not at pipeline exit."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Gen(N_ITEMS), workers=1, outqn=64)
    pipe.add(hog(), gpu_id=0, outqn=64)  # deep queue: it never blocks, it finishes
    pipe.add(Slow(), workers=1, outqn=None)

    n_out, _total, samples = _run_and_sample(pipe, hog.__name__)
    assert n_out == N_ITEMS
    seen = [m for _, m in samples if m is not None]
    assert seen, f"never saw the GPU worker in nvidia-smi: {samples}"
    peak, low = max(seen), min(seen)
    assert peak >= BLOB_BYTES / 2**20, f"blob never showed up on the card (peak {peak} MiB)"
    assert low < CONTEXT_ONLY_MIB, (
        f"parked GPU worker kept {low} MiB while the pipe was still running "
        f"(peak {peak} MiB); samples={[(round(t, 1), m) for t, m in samples]}"
    )


STALL_ITEMS = 4
STALL_SLOW_S = 3.0  # > the 2 s stall threshold in queues._put_retry


@pytest.mark.skipif(not (HAS_CUDA and HAS_SMI), reason="needs a GPU and nvidia-smi")
def test_worker_stalled_on_full_downstream_queue_releases_pool():
    """Case 2: the GPU stage is done computing but BLOCKED on put() because the
    downstream queue is full (outqn=1) and the consumer is slow. It never reaches
    End, so the park release cannot help; the stall hook must return its cached
    pool while it waits -- well before the run ends."""
    pipe = Pipe(raise_errors=True, stats_interval=0, health_check_interval=0)
    pipe.add(Gen(STALL_ITEMS), workers=1, outqn=64)
    pipe.add(GpuScratch(), gpu_id=0, outqn=1)  # one slot: every put after the first stalls
    pipe.add(Slow(STALL_SLOW_S), workers=1, outqn=None)

    n_out, total, samples = _run_and_sample(pipe, "GpuScratch")
    assert n_out == STALL_ITEMS
    seen = [(t, m) for t, m in samples if m is not None]
    assert seen, f"never saw the GPU worker in nvidia-smi: {samples}"
    peak = max(m for _, m in seen)
    assert peak >= BLOB_BYTES / 2**20, f"scratch never showed up on the card (peak {peak} MiB)"
    # Before the fix the pool is held until the very end. Require a release while
    # at least a quarter of the run is still ahead.
    early = [m for t, m in seen if t < 0.75 * total]
    assert early and min(early) < CONTEXT_ONLY_MIB, (
        f"stalled GPU worker held its pool (min {min(early) if early else None} MiB in the "
        f"first 75% of a {total:.1f}s run); samples={[(round(t, 1), m) for t, m in seen]}"
    )
