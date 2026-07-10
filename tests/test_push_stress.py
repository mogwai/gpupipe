"""Thorough edge-case + large-job tests for worker.push(stage, item).

Invariants checked everywhere:
  - every generated item reaches the collector EXACTLY once (no loss, no dup)
  - render_count == 1 + retries   (one initial pass + one re-render per retry)
  - retries == the number this item was supposed to fail

The fail count is a pure function of the item id (id % MOD), so the retry cycle
is deterministic and always converges.
"""
import os

# push()-back tests need the full production drain window: with a short
# PIPE_DRAIN_GRACE (set for speed in conftest) the push target can drain and
# exit before retried items land, leaving the pusher spinning on a full queue.
os.environ["PIPE_DRAIN_GRACE"] = "3.0"

import time

import pytest
import torch
from conftest import Collector

from pipe import End, Pipe

MOD = 5  # item `i` must be retried (i % MOD) times before it passes


def _run(stages, sequential=False, use_shm=False, timeout=120):
    pipe = Pipe(
        sequential=sequential,
        raise_errors=True,
        stats_interval=0,
        health_check_interval=0,
        use_shm=use_shm,
    )
    for stage, kwargs in stages:
        pipe.add(stage, **kwargs)
    out, start = [], time.time()
    for item in pipe:
        out.append(item)
        if time.time() - start > timeout:
            raise TimeoutError("pipeline timeout")
    return out


# === workers (module-level so spawn can pickle them) ===
class Gen:
    def __init__(self, n):
        self.n = n
        self._i = 0

    def load(self):
        pass

    def __call__(self):
        if self._i >= self.n:
            return End
        it = {"id": self._i, "render_count": 0, "b_count": 0, "retries": 0}
        self._i += 1
        return [it, End] if self._i >= self.n else it


class Stamp:
    """Render-like stage: increments a counter every time it sees an item."""

    def __init__(self, key="render_count"):
        self.key = key

    def load(self):
        pass

    def __call__(self, item):
        item[self.key] = item.get(self.key, 0) + 1
        return item


class RetryCheck:
    """Fails (pushes back to `target`) until the item has been retried id%MOD times."""

    def __init__(self, target, mod=MOD):
        self.target = target
        self.mod = mod

    def load(self):
        pass

    def __call__(self, item):
        if item["retries"] < item["id"] % self.mod:
            item["retries"] += 1
            self.push(self.target, item)
            return None
        return item


class RunRetryCheck:
    """Same logic but via a run() worker (exercises push from run()/pull/put)."""

    def __init__(self, target, mod=MOD):
        self.target = target
        self.mod = mod

    def load(self):
        pass

    def run(self):
        while True:
            items = self.pull(8)
            if not items:
                break
            for item in items:
                if item["retries"] < item["id"] % self.mod:
                    item["retries"] += 1
                    self.push(self.target, item)
                else:
                    self.put(item)


class TensorStamp:
    """Render stage that also mutates a tensor payload, to prove tensors survive
    the round trip back through push()."""

    def load(self):
        pass

    def __call__(self, item):
        item["render_count"] = item.get("render_count", 0) + 1
        item["x"] = item["x"] + 1
        return item


class TensorGen:
    def __init__(self, n):
        self.n = n
        self._i = 0

    def load(self):
        pass

    def __call__(self):
        if self._i >= self.n:
            return End
        it = {"id": self._i, "x": torch.tensor([float(self._i)]), "retries": 0}
        self._i += 1
        return [it, End] if self._i >= self.n else it


class BadPush:
    def __init__(self, stage):
        self.stage = stage

    def load(self):
        pass

    def __call__(self, item):
        self.push(self.stage, item)  # expected to raise
        return None


class BatchStamp:
    def load(self):
        pass

    def __call__(self, batch):
        for item in batch:
            item["render_count"] = item.get("render_count", 0) + 1
        return batch


class DeepCheck:
    """Everyone retries exactly 8x — stresses cycle depth."""

    def load(self):
        pass

    def __call__(self, item):
        if item["retries"] < 8:
            item["retries"] += 1
            self.push(1, item)
            return None
        return item


class ClassChecker(RetryCheck):
    """Pushes back by passing the worker CLASS (not index/name)."""

    def __call__(self, item):
        if item["retries"] < item["id"] % self.mod:
            item["retries"] += 1
            self.push(Stamp, item)  # <-- the class itself
            return None
        return item


class SparseCheck:
    """Realistic low-retry-rate gate: only ~1/EVERY of items fail, each retried
    up to MAXR times. The cycle stays tiny, so default queue sizes never
    saturate even at large N — this is the intended production regime."""

    EVERY = 8
    MAXR = 3

    def load(self):
        pass

    def __call__(self, item):
        target = self.MAXR if item["id"] % self.EVERY == 0 else 0
        if item["retries"] < target:
            item["retries"] += 1
            self.push(1, item)
            return None
        return item


# === assertions ===
def _check(results, n, render_key="render_count"):
    assert len(results) == n, f"count {len(results)} != {n} (loss or dup!)"
    by_id = {r["id"]: r for r in results}
    assert set(by_id) == set(range(n)), "missing/extra ids"
    for i, r in by_id.items():
        want = i % MOD
        assert r["retries"] == want, f"id {i}: retries {r['retries']} != {want}"
        assert r[render_key] == 1 + want, (
            f"id {i}: {render_key} {r[render_key]} != {1 + want}"
        )


# === tests ===
@pytest.mark.parametrize("sequential", [True, False])
def test_variable_retry_counts(sequential):
    """Items retry 0..MOD-1 times depending on id; all converge, none lost/dup."""
    n = 200
    # Cyclic queues sized > n so the retry cycle can never saturate (see the
    # back-edge sizing rule in PIPE_REFERENCE).
    stages = [
        (Gen(n), {"workers": 1, "outqn": 256}),
        (Stamp(), {"workers": 1 if sequential else 3, "outqn": 256}),
        (RetryCheck(target=1), {"workers": 1 if sequential else 3, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    _check(_run(stages, sequential=sequential), n)


def test_push_skips_multiple_stages():
    """Check at stage 3 pushes back to stage 1, so BOTH stage 1 and stage 2 re-run."""
    n = 150
    stages = [
        (Gen(n), {"workers": 1, "outqn": 256}),
        (Stamp("render_count"), {"workers": 2, "outqn": 256}),  # stage 1
        (Stamp("b_count"), {"workers": 2, "outqn": 256}),  # stage 2
        (RetryCheck(target=1), {"workers": 2, "outqn": 64}),  # stage 3 -> back to 1
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = _run(stages, sequential=False)
    _check(results, n, render_key="render_count")
    # stage 2 ran the same number of times as stage 1 (both on the back-edge path)
    for r in results:
        assert r["b_count"] == 1 + r["id"] % MOD, r


@pytest.mark.parametrize("sequential", [True, False])
def test_push_by_class(sequential):
    """push() accepts the worker class itself as the target."""
    n = 120
    stages = [
        (Gen(n), {"workers": 1, "outqn": 256}),
        (Stamp(), {"workers": 1 if sequential else 2, "outqn": 256}),
        (ClassChecker(target=1), {"workers": 1 if sequential else 2, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    _check(_run(stages, sequential=sequential), n)


def test_push_from_run_worker():
    n = 150
    stages = [
        (Gen(n), {"workers": 1, "outqn": 256}),
        (Stamp(), {"workers": 2, "outqn": 256}),
        (RunRetryCheck(target=1), {"workers": 2, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    _check(_run(stages, sequential=False), n)


def test_push_with_batched_render():
    """Render stage uses framework batching; pushed-back items get re-batched."""
    n = 200
    stages = [
        (Gen(n), {"workers": 1, "outqn": 256}),
        (BatchStamp(), {"workers": 2, "outqn": 256, "batch": 8}),
        (RetryCheck(target=1), {"workers": 2, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    _check(_run(stages, sequential=False), n)


def test_tensor_payload_survives_pushback():
    """A tensor in the item is correctly carried back through push()."""
    n = 120
    stages = [
        (TensorGen(n), {"workers": 1, "outqn": 256}),
        (TensorStamp(), {"workers": 2, "outqn": 256}),
        (RetryCheck(target=1), {"workers": 2, "outqn": 64}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = _run(stages, sequential=False)
    assert len(results) == n
    for r in results:
        want = r["id"] % MOD
        assert r["render_count"] == 1 + want
        # x started at id, +1 per render pass
        assert r["x"].item() == r["id"] + (1 + want), r


@pytest.mark.parametrize("bad_stage", [0, 99])
def test_push_invalid_stage_raises(bad_stage):
    stages = [
        (Gen(5), {"workers": 1, "outqn": 16}),
        (Stamp(), {"workers": 1, "outqn": 16}),
        (BadPush(bad_stage), {"workers": 1, "outqn": 16}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    with pytest.raises(ValueError):
        _run(stages, sequential=True)


def test_large_job_realistic_retry_rate():
    """Large job in the PRODUCTION regime: 5000 items, many workers, a low retry
    fraction (~1/8 fail, up to 3x). Default-sized queues; proves no loss/dup at
    scale with a back-edge under realistic load."""
    n = 5000
    stages = [
        (Gen(n), {"workers": 1, "outqn": 256}),
        (Stamp(), {"workers": 4, "outqn": 256}),
        (SparseCheck(), {"workers": 4, "outqn": 256}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = _run(stages, sequential=False, timeout=300)
    assert len(results) == n
    assert {r["id"] for r in results} == set(range(n))
    for r in results:
        want = SparseCheck.MAXR if r["id"] % SparseCheck.EVERY == 0 else 0
        assert r["retries"] == want, r
        assert r["render_count"] == 1 + want, r


def test_large_job_all_retry_sized_queues():
    """Large job where EVERY item retries (mod 5) — only safe because the cyclic
    queues are sized > N. Stresses the back-edge under maximal cycling."""
    n = 2000
    stages = [
        (Gen(n), {"workers": 1, "outqn": 4096}),  # back-edge landing queue
        (Stamp(), {"workers": 4, "outqn": 4096}),
        (RetryCheck(target=1), {"workers": 4, "outqn": 256}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    _check(_run(stages, sequential=False, timeout=300), n)


def test_high_retry_depth():
    """Force every item through MANY retries (mod high) to stress cycle depth."""
    n = 300
    # Every item cycles 8x, so up to all n live in the cycle at once -> the
    # back-edge landing queue (Gen.outqn) + forward queue (Stamp.outqn) must
    # exceed n, else the cycle deadlocks.
    stages = [
        (Gen(n), {"workers": 1, "outqn": 512}),
        (Stamp(), {"workers": 3, "outqn": 512}),
        (DeepCheck(), {"workers": 3, "outqn": 128}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = _run(stages, sequential=False)
    assert len(results) == n
    assert {r["id"] for r in results} == set(range(n))
    for r in results:
        assert r["retries"] == 8
        assert r["render_count"] == 9
