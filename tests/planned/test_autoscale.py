"""Autoscaling tests for pipe framework — PLANNED FEATURE, currently disabled.

Autoscaling has been pulled from the live code path (implementation preserved in
`src/pipe/_planned/autoscale.py`; see `PLANNED.md`). These tests exercise the
`Pipe(autoscale=...)` / `add(autoscale=..., min_workers=..., max_workers=...)`
API, which no longer exists, so the whole module is skipped and the
`tests/planned/` directory is excluded from collection in pyproject.toml.
Re-enable alongside re-integrating the feature.

Run with: pytest test_autoscale.py -n auto
"""
import random
import time

import pytest

from pipe import End, Pipe

pytestmark = pytest.mark.skip(
    reason="autoscaling is a planned feature, not currently wired into Pipe"
)


class Generator:
    """Simple generator returning items one at a time."""
    def __init__(self, n_items: int, delay: float = 0):
        self.n_items = n_items
        self.delay = delay
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        if self.delay:
            time.sleep(self.delay)
        item = {"id": self._idx}
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class BatchGenerator:
    """Generator that returns items in batches."""
    def __init__(self, n_items: int, batch_size: int = 10):
        self.n_items = n_items
        self.batch_size = batch_size
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        items = []
        for _ in range(min(self.batch_size, self.n_items - self._idx)):
            items.append({"id": self._idx})
            self._idx += 1
        if self._idx >= self.n_items:
            items.append(End)
        return items


class SlowWorker:
    """Worker with configurable delay."""
    def __init__(self, delay: float = 0.01):
        self.delay = delay

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(self.delay)
        item["processed"] = True
        return item


class FastWorker:
    """Worker with no delay."""
    def load(self):
        pass

    def __call__(self, item):
        item["processed"] = True
        return item


class Collector:
    """Simple pass-through collector."""
    def load(self):
        pass

    def __call__(self, item):
        return item


class VerySlowWorker:
    """Worker with 100ms delay for autoscale testing."""
    def load(self):
        pass
    def __call__(self, item):
        time.sleep(0.1)
        return item


class VariableSlowWorker:
    """Worker with variable processing time based on item."""
    def load(self):
        pass
    def __call__(self, item):
        delay = 0.02 + (item["id"] % 10) * 0.01
        time.sleep(delay)
        return item


class WorkGeneratingWorker:
    """Worker that expands items (generates more work)."""
    def __init__(self, factor: int = 3):
        self.factor = factor
    def load(self):
        pass
    def __call__(self, item):
        time.sleep(0.02)
        return [{"id": item["id"], "sub": i} for i in range(self.factor)]


class BufferingBatcher:
    """Batcher that accumulates items and releases in batches."""
    def __init__(self, size: int = 5):
        self.size = size
        self.buffer = []
    def load(self):
        pass
    def __call__(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.size:
            result = self.buffer
            self.buffer = []
            return result
        return None

    def flush(self):
        if self.buffer:
            result = self.buffer.copy()
            self.buffer = []
            for item in result:
                yield item


class BurstGenerator:
    """Generator that produces items in bursts then pauses."""
    def __init__(self, n_items: int, burst_size: int = 20, pause: float = 0.5):
        self.n_items = n_items
        self.burst_size = burst_size
        self.pause = pause
        self._idx = 0
        self._in_burst = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        if self._in_burst >= self.burst_size:
            time.sleep(self.pause)
            self._in_burst = 0

        item = {"id": self._idx}
        self._idx += 1
        self._in_burst += 1

        if self._idx >= self.n_items:
            return [item, End]
        return item


class SporadicGenerator:
    """Produces items in sporadic bursts with gaps."""
    def __init__(self, n_items: int):
        self.n_items = n_items
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        if self._idx > 0 and self._idx % 10 == 0:
            time.sleep(0.3)
        item = {"id": self._idx}
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class VariableWorkGenerator:
    """Generates items with highly variable processing requirements."""
    def __init__(self, n_items: int):
        self.n_items = n_items
        self._idx = 0
        random.seed(123)

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        work_ms = random.expovariate(1 / 30)
        work_ms = max(1, min(200, work_ms))

        item = {"id": self._idx, "work_ms": work_ms}
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class VariableWorker:
    def load(self):
        pass

    def __call__(self, item):
        time.sleep(item["work_ms"] / 1000)
        item["processed"] = True
        return item


def test_autoscale_slow_worker():
    """Test that slow workers trigger autoscaling."""
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 20), outqn=30)
    pipe.add(VerySlowWorker(), workers=1, outqn=50, autoscale=True, max_workers=8)
    pipe.add(Collector(), workers=1, outqn=0)

    start = time.time()
    results = list(pipe)
    elapsed = time.time() - start

    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))

    print(f"Autoscale test completed in {elapsed:.1f}s with {pipe.stage_worker_counts[1].value} workers")


def test_autoscale_respects_downstream():
    """Test that autoscaling doesn't scale when downstream is the bottleneck."""
    n_items = 50

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 10), outqn=20)
    pipe.add(FastWorker(), workers=1, outqn=10, autoscale=True, max_workers=8)
    pipe.add(VerySlowWorker(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items

    print(f"Downstream bottleneck test: stage 1 has {pipe.stage_worker_counts[1].value} workers")


def test_autoscale_variable_work():
    """Test autoscaling with variable processing times."""
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 10), outqn=30)
    pipe.add(VariableSlowWorker(), workers=1, outqn=50, autoscale=True, max_workers=6)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))
    print(f"Variable work test: stage 1 scaled to {pipe.stage_worker_counts[1].value} workers")


def test_autoscale_with_work_expansion():
    """Test autoscaling when upstream expands work (generates more items)."""
    n_items = 30
    expand_factor = 3

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(n_items), outqn=50)
    pipe.add(WorkGeneratingWorker(expand_factor), workers=1, outqn=100)
    pipe.add(VerySlowWorker(), workers=1, outqn=50, autoscale=True, max_workers=4)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items * expand_factor
    print(f"Work expansion test: stage 2 scaled to {pipe.stage_worker_counts[2].value} workers")


def test_autoscale_with_batcher():
    """Test autoscaling with a batcher in the pipeline."""
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(n_items), outqn=30)
    pipe.add(BufferingBatcher(10), workers=1, outqn=20)
    pipe.add(VerySlowWorker(), workers=1, outqn=50, autoscale=True, max_workers=6)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    print(f"Batcher test: stage 2 scaled to {pipe.stage_worker_counts[2].value} workers")


def test_autoscale_multi_stage():
    """Test autoscaling with multiple autoscale-enabled stages."""
    n_items = 80

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 20), outqn=40)
    pipe.add(SlowWorker(0.05), workers=1, outqn=40, autoscale=True, max_workers=4)
    pipe.add(SlowWorker(0.05), workers=1, outqn=40, autoscale=True, max_workers=4)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    print(f"Multi-stage autoscale: stage 1={pipe.stage_worker_counts[1].value}, stage 2={pipe.stage_worker_counts[2].value} workers")


def test_autoscale_no_scaling_fast_workers():
    """Test that fast workers don't unnecessarily scale."""
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(n_items, delay=0.02), outqn=20)
    pipe.add(FastWorker(), workers=1, outqn=50, autoscale=True, max_workers=8)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert pipe.stage_worker_counts[1].value <= 2, f"Unexpected scaling to {pipe.stage_worker_counts[1].value} workers"
    print(f"Fast workers test: stayed at {pipe.stage_worker_counts[1].value} workers (expected ~1)")


def test_autoscale_max_workers_limit():
    """Test that autoscaling respects max_workers limit."""
    n_items = 200

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 50), outqn=100)
    pipe.add(VerySlowWorker(), workers=1, outqn=100, autoscale=True, max_workers=2)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert pipe.stage_worker_counts[1].value <= 2, f"Exceeded max_workers: {pipe.stage_worker_counts[1].value}"
    print(f"Max workers limit test: capped at {pipe.stage_worker_counts[1].value} workers")


def test_autoscale_up_and_down():
    """Test that workers can scale up during bursts and scale down during pauses."""
    n_items = 60

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BurstGenerator(n_items, burst_size=20, pause=0.3), outqn=30)
    pipe.add(SlowWorker(0.05), workers=2, outqn=50, autoscale=True, max_workers=4)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))
    print(f"Scale up/down test: ended with {pipe.stage_worker_counts[1].value} workers")


def test_autoscale_slow_consumer():
    """Test autoscaling when consumer iterates slowly.

    Slow consumer = output queue fills up, but that's OK - we're keeping up.
    Should NOT over-scale just because output is full.
    """
    n_items = 50

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 10), outqn=20)
    pipe.add(FastWorker(), workers=1, outqn=10, autoscale=True, max_workers=4)

    results = []
    worker_counts = []
    for item in pipe:
        results.append(item)
        worker_counts.append(pipe.stage_worker_counts[1].value)
        time.sleep(0.1)

    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))

    max_workers_seen = max(worker_counts)
    print(f"Slow consumer test: max workers seen = {max_workers_seen}")
    assert max_workers_seen <= 2, f"Over-scaled to {max_workers_seen} workers with slow consumer"


def test_autoscale_fast_consumer():
    """Test autoscaling when consumer iterates as fast as possible.

    Fast consumer = output queue drains quickly, workers should scale
    based on input queue pressure only.
    """
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 20), outqn=30)
    pipe.add(VerySlowWorker(), workers=1, outqn=50, autoscale=True, max_workers=4)

    results = []
    start = time.time()
    for item in pipe:
        results.append(item)

    elapsed = time.time() - start
    assert len(results) == n_items

    final_workers = pipe.stage_worker_counts[1].value
    print(f"Fast consumer test: {elapsed:.1f}s, final workers = {final_workers}")
    assert final_workers >= 2, f"Under-scaled: only {final_workers} workers"


def test_autoscale_settling():
    """Test that autoscaling settles and doesn't thrash.

    After initial scaling, worker count should stabilize.
    """
    n_items = 200

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 10), outqn=30)
    pipe.add(SlowWorker(0.03), workers=1, outqn=30, autoscale=True, max_workers=6)

    results = []
    worker_history = []
    last_change_idx = 0
    prev_count = 1

    for i, item in enumerate(pipe):
        results.append(item)
        current = pipe.stage_worker_counts[1].value
        worker_history.append(current)
        if current != prev_count:
            last_change_idx = i
            prev_count = current

    assert len(results) == n_items

    settled_at = last_change_idx / n_items
    print(f"Settling test: worker history sample = {worker_history[::20]}")
    print(f"Settling test: last change at item {last_change_idx} ({settled_at:.0%})")

    oscillations = 0
    for i in range(2, len(worker_history)):
        if worker_history[i-2] < worker_history[i-1] > worker_history[i]:
            oscillations += 1
        elif worker_history[i-2] > worker_history[i-1] < worker_history[i]:
            oscillations += 1

    print(f"Settling test: {oscillations} oscillations detected")
    assert oscillations < 5, f"Too many oscillations ({oscillations}) - autoscaler is thrashing"


def test_autoscale_cpu_limit():
    """Test that max_workers caps scaling even under heavy load."""
    n_items = 100
    max_allowed = 3

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 50), outqn=60)
    pipe.add(VerySlowWorker(), workers=1, outqn=50, autoscale=True, max_workers=max_allowed)

    max_seen = 1
    results = []
    for item in pipe:
        results.append(item)
        current = pipe.stage_worker_counts[1].value
        max_seen = max(max_seen, current)

    assert len(results) == n_items
    assert max_seen <= max_allowed, f"Exceeded max_workers: {max_seen} > {max_allowed}"
    print(f"CPU limit test: max workers = {max_seen} (limit was {max_allowed})")


def test_autoscale_no_underscale_during_work():
    """Test that workers don't scale down while there's still work to do."""
    n_items = 80

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 40), outqn=50)
    pipe.add(SlowWorker(0.05), workers=3, outqn=50, autoscale=True, max_workers=4)

    results = []
    min_workers_while_working = 10

    for item in pipe:
        results.append(item)
        current = pipe.stage_worker_counts[1].value
        if len(results) > 10 and len(results) < n_items - 5:
            min_workers_while_working = min(min_workers_while_working, current)

    assert len(results) == n_items
    print(f"No underscale test: min workers during work = {min_workers_while_working}")
    assert min_workers_while_working >= 2, f"Under-scaled to {min_workers_while_working} while work remained"


def test_autoscale_multi_bottleneck():
    """Test autoscaling with multiple potential bottlenecks.

    Stage 1: fast
    Stage 2: slow (bottleneck)
    Stage 3: fast

    Only stage 2 should scale up.
    """
    n_items = 60

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 20), outqn=30)
    pipe.add(FastWorker(), workers=1, outqn=20, autoscale=True, max_workers=4)
    pipe.add(VerySlowWorker(), workers=1, outqn=30, autoscale=True, max_workers=4)
    pipe.add(FastWorker(), workers=1, outqn=0, autoscale=True, max_workers=4)

    results = list(pipe)
    assert len(results) == n_items

    stage1_workers = pipe.stage_worker_counts[1].value
    stage2_workers = pipe.stage_worker_counts[2].value
    stage3_workers = pipe.stage_worker_counts[3].value

    print(f"Multi-bottleneck: stage1={stage1_workers}, stage2={stage2_workers}, stage3={stage3_workers}")

    assert stage2_workers >= stage1_workers, "Slow stage should scale more than fast upstream"
    assert stage1_workers <= 2, f"Fast stage 1 over-scaled to {stage1_workers}"


def test_autoscale_consumer_pause_resume():
    """Test behavior when consumer pauses then resumes.

    Simulates real-world scenario where downstream processing varies.
    """
    n_items = 60

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(n_items), outqn=20)
    pipe.add(SlowWorker(0.02), workers=1, outqn=20, autoscale=True, max_workers=4)

    results = []
    for i, item in enumerate(pipe):
        results.append(item)
        if 20 <= i < 30:
            time.sleep(0.2)

    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))
    print(f"Pause/resume test: final workers = {pipe.stage_worker_counts[1].value}")


def test_autoscale_empty_periods():
    """Test scaling when input has empty periods (upstream produces nothing)."""
    n_items = 40

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(SporadicGenerator(n_items), outqn=15)
    pipe.add(FastWorker(), workers=2, outqn=20, autoscale=True, max_workers=4)

    results = list(pipe)
    assert len(results) == n_items

    final_workers = pipe.stage_worker_counts[1].value
    print(f"Empty periods test: final workers = {final_workers}")


def test_autoscale_all_items_processed():
    """Stress test: verify no items lost under heavy autoscaling."""
    n_items = 500

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 50), outqn=100)
    pipe.add(SlowWorker(0.01), workers=1, outqn=100, autoscale=True, max_workers=8)
    pipe.add(FastWorker(), workers=1, outqn=100, autoscale=True, max_workers=4)

    results = []
    for item in pipe:
        results.append(item)
        if len(results) % 50 == 0:
            time.sleep(0.05)

    assert len(results) == n_items, f"Lost items: got {len(results)}, expected {n_items}"
    assert {r["id"] for r in results} == set(range(n_items)), "Missing item IDs"
    print(f"Stress test: all {n_items} items processed, stage1={pipe.stage_worker_counts[1].value}, stage2={pipe.stage_worker_counts[2].value}")


def test_autoscale_min_workers_respected():
    """Test that min_workers (initial worker count) is respected."""
    n_items = 30
    initial_workers = 3

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(Generator(n_items, delay=0.05), outqn=10)
    # min_workers defaults to 1, so without setting it the autoscaler may legitimately
    # scale below the initial count once it's idle long enough. Pin the floor to verify
    # min_workers is honoured (otherwise this only passed by finishing before scale-down).
    pipe.add(FastWorker(), workers=initial_workers, outqn=20, autoscale=True,
             min_workers=initial_workers, max_workers=6)

    min_seen = initial_workers
    results = []
    for item in pipe:
        results.append(item)
        current = pipe.stage_worker_counts[1].value
        min_seen = min(min_seen, current)

    assert len(results) == n_items
    assert min_seen >= initial_workers, f"Dropped below min_workers: {min_seen} < {initial_workers}"
    print(f"Min workers test: min seen = {min_seen} (started with {initial_workers})")


def test_autoscale_threaded_workers():
    """Test that autoscaling works with threaded workers."""
    n_items = 80

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 20), outqn=30)
    pipe.add(SlowWorker(0.05), workers=1, thread=True, outqn=30, autoscale=True, min_workers=1, max_workers=4)
    pipe.add(Collector(), workers=1, outqn=0)

    results = []
    for item in pipe:
        results.append(item)

    assert len(results) == n_items
    final_workers = pipe.stage_worker_counts[1].value
    print(f"Threaded autoscale: {len(results)} items, final workers={final_workers}")
    assert final_workers >= 1


def test_autoscale_min_max_workers():
    """Test that min_workers and max_workers are respected."""
    n_items = 60

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, 30), outqn=40)
    pipe.add(SlowWorker(0.03), workers=3, outqn=40, autoscale=True, min_workers=2, max_workers=5)
    pipe.add(Collector(), workers=1, outqn=0)

    worker_history = []
    results = []
    for item in pipe:
        results.append(item)
        worker_history.append(pipe.stage_worker_counts[1].value)

    assert len(results) == n_items

    min_seen = min(worker_history)
    max_seen = max(worker_history)
    print(f"Min/max test: min={min_seen}, max={max_seen} (limits: 2-5)")

    assert min_seen >= 2, f"Dropped below min_workers: {min_seen}"
    assert max_seen <= 5, f"Exceeded max_workers: {max_seen}"


def test_autoscale_global():
    """Test that Pipe(autoscale=True) enables autoscaling for all stages."""
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0, autoscale=True, max_workers_per_stage=4)
    pipe.add(BatchGenerator(n_items, 50), outqn=60)
    pipe.add(SlowWorker(0.15), workers=1, outqn=40)
    pipe.add(Collector(), workers=1, outqn=0)

    assert pipe.jobs[0]["autoscale"] == True, "Stage 0 should have autoscale enabled"
    assert pipe.jobs[1]["autoscale"] == True, "Stage 1 should have autoscale enabled"
    assert pipe.jobs[2]["autoscale"] == True, "Stage 2 should have autoscale enabled"

    assert pipe.jobs[1]["max_workers"] == 4, f"Stage 1 max_workers should be 4, got {pipe.jobs[1]['max_workers']}"

    results = []
    worker_history = []
    for item in pipe:
        results.append(item)
        worker_history.append(pipe.stage_worker_counts[1].value)

    assert len(results) == n_items

    max_workers = max(worker_history)
    print(f"Global autoscale test: {len(results)} items, max_workers seen={max_workers}")
    assert max_workers > 1, f"Expected scaling with global autoscale, but max workers was {max_workers}"


def test_autoscale_global_with_override():
    """Test that per-stage settings override global autoscale."""
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0, autoscale=True, max_workers_per_stage=8)
    pipe.add(BatchGenerator(20, 10), outqn=20)
    pipe.add(SlowWorker(0.01), workers=2, outqn=20, max_workers=3)
    pipe.add(SlowWorker(0.01), workers=1, outqn=20, autoscale=False)
    pipe.add(Collector(), workers=1, outqn=0)

    assert pipe.jobs[1]["autoscale"] == True
    assert pipe.jobs[1]["max_workers"] == 3, f"Stage 1 should use override max_workers=3, got {pipe.jobs[1]['max_workers']}"

    assert pipe.jobs[2]["autoscale"] == False, f"Stage 2 should have autoscale=False from explicit override, got {pipe.jobs[2]['autoscale']}"

    assert pipe.jobs[3]["autoscale"] == True, "Stage 3 should inherit global autoscale=True"


def test_cpu_metric_accuracy():
    """Test that _get_cpu_usage accurately reflects actual CPU load."""
    import os
    import subprocess

    # _get_cpu_usage moved out of the live pipe.monitors with the feature
    from pipe._planned.autoscale import _get_cpu_usage

    idle1, total1 = _get_cpu_usage()
    time.sleep(0.3)
    idle2, total2 = _get_cpu_usage()
    baseline = 1.0 - ((idle2 - idle1) / (total2 - total1))
    print(f"Baseline CPU: {baseline:.1%}")

    if baseline > 0.7:
        pytest.skip(f"host CPU already saturated ({baseline:.0%}); cannot measure load delta")

    n_burners = min(os.cpu_count(), 8)
    procs = []
    for _ in range(n_burners):
        p = subprocess.Popen(
            ["dd", "if=/dev/urandom", "of=/dev/null", "bs=1M"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(p)

    try:
        time.sleep(0.5)

        idle1, total1 = _get_cpu_usage()
        time.sleep(0.5)
        idle2, total2 = _get_cpu_usage()
        under_load = 1.0 - ((idle2 - idle1) / (total2 - total1))
        print(f"Under load CPU: {under_load:.1%}")

        assert under_load > baseline + 0.2, f"CPU metric should increase under load: {baseline:.1%} -> {under_load:.1%}"

    finally:
        for p in procs:
            p.terminate()
            p.wait()

    time.sleep(0.3)
    idle1, total1 = _get_cpu_usage()
    time.sleep(0.3)
    idle2, total2 = _get_cpu_usage()
    after_load = 1.0 - ((idle2 - idle1) / (total2 - total1))
    print(f"After load CPU: {after_load:.1%}")

    assert after_load < under_load - 0.1, f"CPU should drop after load ends: {under_load:.1%} -> {after_load:.1%}"


def test_variable_workload_settling():
    """Test that autoscaling settles with highly variable workloads."""
    n_items = 100
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(VariableWorkGenerator(n_items), outqn=30)
    pipe.add(VariableWorker(), workers=1, outqn=30, autoscale=True, max_workers=6)

    results = []
    worker_history = []
    for item in pipe:
        results.append(item)
        worker_history.append(pipe.stage_worker_counts[1].value)

    assert len(results) == n_items

    changes = sum(1 for i in range(1, len(worker_history)) if worker_history[i] != worker_history[i-1])
    print(f"Variable workload: {len(results)} items, {changes} worker changes, final={worker_history[-1]}")
    assert changes < 20, f"Too many worker changes ({changes}) - not settling"


class ScaleDownRaceGenerator:
    """Generator that creates conditions to trigger scale-down during processing.

    Produces a burst of items, then pauses to trigger scale-down,
    while workers are still processing the burst.
    """
    def __init__(self, n_items: int, burst_size: int = 30, pause_after_burst: float = 1.0):
        self.n_items = n_items
        self.burst_size = burst_size
        self.pause_after_burst = pause_after_burst
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        item = {"id": self._idx}
        self._idx += 1

        # After first burst, pause to trigger scale-down
        if self._idx == self.burst_size:
            time.sleep(self.pause_after_burst)

        if self._idx >= self.n_items:
            return [item, End]
        return item


class MediumSlowWorker:
    """Worker with 50ms delay - slow enough to accumulate work."""
    def load(self):
        pass

    def __call__(self, item):
        time.sleep(0.05)
        item["processed"] = True
        return item


def test_autoscale_down_no_item_loss():
    """Test that scaling down doesn't cause item loss.

    This test catches a race condition where:
    1. Autoscaler signals worker to stop and decrements stage_worker_count
    2. Stopped worker exits and increments finished_workers
    3. finished_workers >= stage_worker_count triggers "all workers done"
    4. But other workers are still processing items!

    The bug: stage_worker_count is decremented BEFORE the worker actually exits,
    causing premature "all workers finished" signal.
    """
    n_items = 200

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    # Generate all items quickly
    pipe.add(BatchGenerator(n_items, batch_size=n_items), outqn=250)
    # Start with many workers, slow processing - autoscaler will scale down
    pipe.add(MediumSlowWorker(), workers=4, outqn=100, autoscale=True, min_workers=1, max_workers=6)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    # The critical assertion: ALL items must be processed
    assert len(results) == n_items, f"Item loss detected: got {len(results)}, expected {n_items}"
    assert {r["id"] for r in results} == set(range(n_items)), "Missing item IDs"

    print(f"Scale-down race test: {len(results)} items, final workers={pipe.stage_worker_counts[1].value}")


def test_autoscale_down_during_processing_stress():
    """Stress test: repeatedly trigger scale-down while items are being processed.

    Uses burst-pause-burst pattern to maximize chance of hitting the race condition.
    """
    n_items = 300

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(ScaleDownRaceGenerator(n_items, burst_size=50, pause_after_burst=0.8), outqn=150)
    pipe.add(SlowWorker(0.03), workers=4, outqn=100, autoscale=True, min_workers=1, max_workers=8)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    assert len(results) == n_items, f"Item loss during scale-down: got {len(results)}, expected {n_items}"
    assert {r["id"] for r in results} == set(range(n_items)), "Missing item IDs"

    print(f"Scale-down stress: {len(results)} items processed")


class RapidBurstGenerator:
    """Alternates between bursts and pauses rapidly."""
    def __init__(self, n_items: int, burst_size: int = 25, pause: float = 0.3):
        self.n_items = n_items
        self.burst_size = burst_size
        self.pause = pause
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        # Every burst_size items, pause briefly
        if self._idx > 0 and self._idx % self.burst_size == 0:
            time.sleep(self.pause)

        item = {"id": self._idx}
        self._idx += 1

        if self._idx >= self.n_items:
            return [item, End]
        return item


def test_autoscale_rapid_up_down_no_loss():
    """Test rapid scaling up and down doesn't lose items.

    Creates conditions for rapid autoscaling oscillation to maximize
    chance of hitting race conditions in worker coordination.
    """
    n_items = 400

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(RapidBurstGenerator(n_items, burst_size=25, pause=0.3), outqn=60)
    pipe.add(SlowWorker(0.02), workers=3, outqn=80, autoscale=True, min_workers=1, max_workers=6)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    assert len(results) == n_items, f"Lost items in rapid up/down: got {len(results)}, expected {n_items}"
    assert {r["id"] for r in results} == set(range(n_items)), "Missing item IDs after rapid scaling"

    print(f"Rapid up/down: {len(results)} items, workers={pipe.stage_worker_counts[1].value}")


def test_autoscale_concurrent_scale_down_multiple_stages():
    """Test concurrent scale-down across multiple stages.

    If multiple stages scale down simultaneously, coordination bugs
    can cause items to be lost between stages.
    """
    n_items = 200

    # Use BurstThenPauseGenerator which is already defined at module level
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BurstThenPauseGenerator(n_items, burst_size=50, pause_duration=0.5), outqn=80)
    pipe.add(SlowWorker(0.02), workers=3, outqn=60, autoscale=True, min_workers=1, max_workers=4)
    pipe.add(SlowWorker(0.02), workers=3, outqn=60, autoscale=True, min_workers=1, max_workers=4)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    assert len(results) == n_items, f"Multi-stage scale-down lost items: got {len(results)}, expected {n_items}"
    assert {r["id"] for r in results} == set(range(n_items)), "Missing IDs in multi-stage test"

    print(f"Multi-stage scale-down: {len(results)} items")


@pytest.mark.parametrize("initial_workers,n_items", [
    (2, 100),
    (4, 200),
    (6, 300),
    (8, 400),
])
def test_autoscale_down_parametric(initial_workers, n_items):
    """Parametric test for scale-down with various worker counts.

    Tests the race condition with different numbers of initial workers
    to maximize coverage of edge cases.
    """
    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BatchGenerator(n_items, batch_size=n_items), outqn=n_items + 50)
    pipe.add(
        MediumSlowWorker(),
        workers=initial_workers,
        outqn=100,
        autoscale=True,
        min_workers=1,
        max_workers=initial_workers + 2
    )
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    assert len(results) == n_items, f"workers={initial_workers}: got {len(results)}, expected {n_items}"
    assert {r["id"] for r in results} == set(range(n_items))


class SlowStartGenerator:
    """Generates items where first batch takes longer to process."""
    def __init__(self, n_items: int, slow_count: int = 20):
        self.n_items = n_items
        self.slow_count = slow_count
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        process_time = 0.1 if self._idx < self.slow_count else 0.01
        item = {"id": self._idx, "process_time": process_time}
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class VariableProcessTimeWorker:
    """Processes items according to their process_time field."""
    def load(self):
        pass

    def __call__(self, item):
        time.sleep(item.get("process_time", 0.01))
        item["processed"] = True
        return item


def test_scale_down_triggers_premature_end_signal():
    """Test that verifies the bug: scale-down can trigger premature end signal.

    BUG DESCRIPTION:
    When autoscaler scales down, it decrements stage_worker_count BEFORE the
    worker actually exits. When that worker then exits:
    - finished_workers increments to N
    - current_worker_count was already decremented to N
    - finished_workers >= current_worker_count triggers "all workers done"
    - But other workers are still processing!

    This test creates conditions where:
    1. Workers are processing slow items
    2. Autoscaler decides to scale down (queue empties)
    3. Stopped worker exits before other workers finish their items
    4. If bug exists: remaining items are lost
    """
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(SlowStartGenerator(n_items, slow_count=20), outqn=150)
    # Start with 4 workers - autoscaler may scale down when queue empties
    pipe.add(VariableProcessTimeWorker(), workers=4, outqn=100, autoscale=True, min_workers=1, max_workers=6)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    # Critical: verify ALL items processed
    assert len(results) == n_items, (
        f"PREMATURE END SIGNAL BUG: got {len(results)}, expected {n_items}. "
        f"Scale-down likely triggered 'all workers done' while workers still processing."
    )
    assert {r["id"] for r in results} == set(range(n_items))


def test_scale_down_worker_count_coordination():
    """Test worker count is tracked correctly during scale-down.

    Verifies that when a worker receives stop signal, the coordination
    between stage_worker_count and stage_end_counter remains correct.
    """
    import ctypes
    from multiprocessing import Value

    # Simulate the bug scenario directly
    stage_worker_count = Value(ctypes.c_int, 2)  # Start with 2 workers
    stage_end_counter = Value(ctypes.c_int, 0)   # No workers finished yet

    # Simulate autoscaler scaling down: decrements count BEFORE worker exits
    # This is what _signal_worker_to_stop does at line 1015
    stage_worker_count.value -= 1  # Now 1

    # Worker 1 finishes (the one that received stop signal)
    with stage_end_counter.get_lock():
        stage_end_counter.value += 1
        finished_workers = stage_end_counter.value  # = 1

    current_worker_count = stage_worker_count.value  # = 1

    # BUG: This condition triggers "all workers done" prematurely!
    all_done_triggered = finished_workers >= current_worker_count

    # Worker 0 is still processing, but we already signaled "all done"
    # This is the race condition

    # After worker 0 finishes:
    with stage_end_counter.get_lock():
        stage_end_counter.value += 1
        finished_workers_after = stage_end_counter.value  # = 2

    # Now finished_workers (2) > current_worker_count (1)
    # This shows worker 0 finished AFTER "all done" was signaled

    print(f"After scale-down signal: worker_count={current_worker_count}, finished={finished_workers}")
    print(f"'All done' triggered prematurely: {all_done_triggered}")
    print(f"After all workers exit: finished={finished_workers_after}")

    # This assertion documents the bug - it SHOULD fail once the bug is fixed
    # Currently it passes because this IS the buggy behavior
    assert all_done_triggered == True, "Bug demo: premature 'all done' signal"
    assert finished_workers_after > current_worker_count, (
        "Bug demo: worker finished after 'all done' was signaled"
    )


class BurstThenPauseGenerator:
    """Generates burst of items then pauses to trigger scale-down consideration."""
    def __init__(self, n_items: int, burst_size: int = 50, pause_duration: float = 0.3):
        self.n_items = n_items
        self.burst_size = burst_size
        self.pause_duration = pause_duration
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        # After burst, pause to trigger autoscaler scale-down
        if self._idx == self.burst_size:
            time.sleep(self.pause_duration)

        item = {"id": self._idx}
        self._idx += 1

        if self._idx >= self.n_items:
            return [item, End]
        return item


class ScaleDownTriggerGenerator:
    """Generator designed to trigger scale-down during processing.

    Pattern:
    1. Generate items slowly at first (allows workers to scale up if needed)
    2. Stop generating for 8+ seconds (triggers scale-down after 5 samples)
    3. Continue generating remaining items

    The key is workers must still be processing items during the pause,
    so we use slow_items that take a long time to process.
    """
    def __init__(self, n_items: int, pause_after: int = 30, pause_duration: float = 8.0):
        self.n_items = n_items
        self.pause_after = pause_after
        self.pause_duration = pause_duration
        self._idx = 0
        self._paused = False

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        # After pause_after items, wait long enough to trigger scale-down (5+ seconds)
        if self._idx == self.pause_after and not self._paused:
            self._paused = True
            print(f"Generator pausing for {self.pause_duration}s to trigger scale-down...")
            time.sleep(self.pause_duration)
            print("Generator resuming...")

        item = {"id": self._idx}
        self._idx += 1

        if self._idx >= self.n_items:
            return [item, End]
        return item


class VerySlowProcessingWorker:
    """Worker that takes 200ms per item - slow enough that items queue up."""
    def __init__(self, process_time: float = 0.2):
        self.process_time = process_time

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(self.process_time)
        item["processed"] = True
        return item


class FixedTimeWorker:
    """Worker that takes fixed time per item."""
    def __init__(self, process_time: float = 0.05):
        self.process_time = process_time

    def load(self):
        pass

    def __call__(self, item):
        time.sleep(self.process_time)
        item["processed"] = True
        return item


def test_scale_down_with_slow_processing_worker():
    """Reproduce the exact scenario from the ESB alignment bug.

    Scenario:
    - 2 workers at AudioDownloader stage
    - Autoscaler scales down from 2 to 1
    - Worker 1 exits (1/1) and triggers "all workers finished"
    - Worker 0 was still processing and finishes later (2/1)
    - Result: 200 items in, only 132 processed
    """
    n_items = 200

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(BurstThenPauseGenerator(n_items, burst_size=50, pause_duration=0.5), outqn=100)
    pipe.add(FixedTimeWorker(0.05), workers=2, outqn=100, autoscale=True, min_workers=1, max_workers=4)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    # This is the critical assertion - the ESB bug caused 200->132 items
    assert len(results) == n_items, (
        f"ESB BUG REPRODUCED: {len(results)}/{n_items} items processed. "
        f"Scale-down caused premature 'all workers finished' signal."
    )
    assert {r["id"] for r in results} == set(range(n_items))


def test_scale_down_race_condition():
    """Test that specifically triggers and catches the scale-down race condition.

    This test is designed to:
    1. Start with multiple workers (4)
    2. Generate items slowly at first, then pause for 8+ seconds
    3. During pause, input queue empties -> autoscaler decides to scale down
    4. BUT workers are still processing items from before the pause
    5. If bug exists: scaled-down worker triggers "all done" prematurely

    The bug: _signal_worker_to_stop decrements stage_worker_count BEFORE
    the worker actually exits, causing finished_workers >= stage_worker_count
    to trigger prematurely.
    """
    n_items = 100

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    # Generate 30 items, pause 8s (triggers scale-down), then remaining 70
    pipe.add(ScaleDownTriggerGenerator(n_items, pause_after=30, pause_duration=8.0), outqn=50)
    # 4 workers with slow processing - scale-down will be triggered during pause
    # min_workers=1 allows scaling down from 4
    pipe.add(VerySlowProcessingWorker(0.2), workers=4, outqn=100, autoscale=True, min_workers=1, max_workers=6)
    pipe.add(Collector(), workers=1, outqn=0)

    results = list(pipe)

    # If the bug exists, we'll get fewer than n_items because "all workers done"
    # was signaled prematurely when a scaled-down worker exited
    assert len(results) == n_items, (
        f"SCALE-DOWN RACE CONDITION: got {len(results)}/{n_items} items. "
        f"Premature 'all workers finished' signal caused item loss."
    )
    assert {r["id"] for r in results} == set(range(n_items))
    print(f"Scale-down race test: {len(results)} items, final workers={pipe.stage_worker_counts[1].value}")


# === regression tests for autoscaling bugs found in review ===

def test_per_stage_autoscale_uses_global_max_default():
    """Per-stage autoscale opt-in (global off) should still use max_workers_per_stage."""
    pipe = Pipe(autoscale=False, max_workers_per_stage=7, stats_interval=0)
    pipe.add(Generator(1), outqn=10)
    pipe.add(FastWorker(), workers=1, autoscale=True, outqn=10)  # opt-in, no max_workers
    pipe.add(Collector(), workers=1, outqn=0)
    assert pipe.jobs[1]["autoscale"] is True
    assert pipe.jobs[1]["max_workers"] == 7    # not actual_workers*4 == 4


def test_threaded_stage_autoscale_disabled():
    """Threaded stages can't be autoscaled by spawning processes; must be disabled."""
    pipe = Pipe(autoscale=True, stats_interval=0)
    pipe.add(Generator(1), outqn=10)
    pipe.add(FastWorker(), workers=2, thread=True, autoscale=True, outqn=10)
    pipe.add(Collector(), workers=1, outqn=0)
    assert pipe.jobs[1]["autoscale"] is False


def test_autoscale_no_worker_id_collision_after_down_up():
    """After a scale-down, scale-up must not reuse a still-live worker id."""
    from pipe.workers import _spawn_additional_worker

    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(Generator(5), outqn=10)
    pipe.add(FastWorker(), workers=2, outqn=10)   # stage 1: worker_0, worker_1
    pipe.add(Collector(), workers=1, outqn=0)
    pipe.start()
    try:
        def stage1_ids():
            return [wid for _, wid, _ in pipe.worker_info if wid.startswith("stage_1_worker")]

        assert sorted(stage1_ids()) == ["stage_1_worker_0", "stage_1_worker_1"]

        # simulate the autoscaler: scale-down decrements only the live count...
        with pipe.stage_worker_counts[1].get_lock():
            pipe.stage_worker_counts[1].value -= 1
        # ...then scale-up must allocate a fresh id, not reuse live worker_1
        _spawn_additional_worker(pipe, 1, pipe.jobs[1])

        ids = stage1_ids()
        assert len(ids) == len(set(ids)), f"duplicate worker ids: {ids}"
        assert "stage_1_worker_2" in ids
    finally:
        pipe.stop(force=True)
