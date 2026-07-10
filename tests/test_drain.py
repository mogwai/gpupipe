"""Test drain mode: first Ctrl+C drains queues, second force-stops."""
import time
from queue import Empty

from pipe import End, Pipe


class SlowRoot:
    def __init__(self, n=100):
        self.i = 0
        self.n = n

    def __call__(self):
        self.i += 1
        if self.i > self.n:
            return End
        time.sleep(0.01)
        return self.i


class Double:
    def __call__(self, x):
        return x * 2


class FastSeqRoot:
    """Emits sequential ints 1..n so we can detect missing items."""
    def __init__(self, n=80):
        self.i = 0
        self.n = n

    def __call__(self):
        self.i += 1
        if self.i > self.n:
            return End
        return self.i


class SlowSink:
    def __init__(self, delay=0.03):
        self.delay = delay

    def __call__(self, x):
        time.sleep(self.delay)
        return x


def test_drain_stops_root_but_drains_queue():
    p = Pipe(sequential=False, stats_interval=0, health_check_interval=0)
    p.add(SlowRoot(n=200), workers=1, outqn=50)
    p.add(Double(), workers=2, outqn=50)
    p.start()

    results = []
    for item in p:
        results.append(item)
        if len(results) == 5:
            # Simulate first Ctrl+C
            p.drain_event.set()

    # Root should have stopped generating soon after drain_event was set,
    # but items already in the queue should have been processed
    assert len(results) >= 5, f"Expected >=5, got {len(results)}"
    # Should NOT have all 200 items since root was stopped early
    assert len(results) < 200, f"Expected <200, got {len(results)}"
    # All items should be doubled
    for r in results:
        assert r % 2 == 0, f"Expected even number, got {r}"
    print(f"PASS: drain got {len(results)} items (root stopped, queue drained)")


def test_drain_sequential_unaffected():
    p = Pipe(sequential=True, stats_interval=0)
    p.add(SlowRoot(n=20), workers=1, outqn=10)
    p.add(Double(), workers=1, outqn=10)

    results = list(p)
    assert len(results) == 20
    print("PASS: sequential mode unaffected")


def test_normal_completion_still_works():
    p = Pipe(sequential=False, stats_interval=0, health_check_interval=0)
    p.add(SlowRoot(n=30), workers=1, outqn=50)
    p.add(Double(), workers=2, outqn=50)

    results = list(p)
    assert len(results) == 30, f"Expected 30, got {len(results)}"
    assert all(r % 2 == 0 for r in results)
    print("PASS: normal completion works")


def test_drain_param_rejects_gap():
    """drain=False stages must form a contiguous prefix from stage 0."""
    p = Pipe(sequential=False, stats_interval=0, health_check_interval=0)
    p.add(SlowRoot(n=10))                  # auto drain=False (stage 0)
    p.add(Double(), drain=True)            # stage 1
    p.add(Double(), drain=True)            # stage 2
    p.add(Double(), drain=False)           # stage 3 — gap, should reject

    raised = False
    try:
        p.start()
    except ValueError as e:
        raised = True
        assert "contiguous prefix" in str(e), f"unexpected msg: {e}"
    finally:
        try:
            p._stop(force=True)
        except Exception:
            pass
    assert raised, "expected ValueError for non-contiguous drain=False prefix"
    print("PASS: drain=False gap rejected")


def test_drain_param_accepts_contiguous_prefix():
    """drain=False on stages 0,1,2 then drain=True after is allowed."""
    p = Pipe(sequential=False, stats_interval=0, health_check_interval=0)
    p.add(SlowRoot(n=5))                   # stage 0 auto False
    p.add(Double(), drain=False)           # stage 1
    p.add(Double(), drain=False)           # stage 2
    p.add(Double(), drain=True)            # stage 3

    results = list(p)
    # 5 items, doubled 3 times = original * 8
    assert len(results) == 5, f"expected 5, got {len(results)}"
    assert sorted(results) == [8, 16, 24, 32, 40], f"got {sorted(results)}"
    print("PASS: contiguous prefix accepted, normal run completes")


def test_drain_false_middle_stops_on_drain():
    """When drain triggers, drain=False middle stages stop reading new items.
    Items already past those stages should still complete via the drain=True tail.
    """
    p = Pipe(sequential=False, stats_interval=0, health_check_interval=0)
    p.add(SlowRoot(n=200), workers=1, outqn=50)
    p.add(Double(), workers=1, outqn=50, drain=False)  # stage 1: stops on drain
    p.add(SlowSink(delay=0.01), workers=2, outqn=50)   # stage 2: drains in-flight
    p.start()

    results = []
    for item in p:
        results.append(item)
        if len(results) == 3:
            p.drain_event.set()

    # Stage 1 stops shortly after drain. Stage 2 finishes whatever stage 1 already
    # delivered. So we should have a small number of items, well under 200.
    assert len(results) >= 3, f"got only {len(results)}"
    assert len(results) < 200, f"drain didn't stop stage 1: got {len(results)}"
    for r in results:
        assert r % 2 == 0
    print(f"PASS: drain=False middle stage stopped, got {len(results)} items")


def test_should_stop_does_not_drop_inflight_put():
    """Regression: a worker mid out_queue.put retry must commit its item when
    should_stop=1 is set, not silently drop it.

    Setup: single root stage with tiny outqn=2 and NO consumer. After start,
    the root emits 2 items (queue fills) and blocks in the put-retry loop on
    item 3. We set should_stop=1, wait long enough for the put timeout (0.1s)
    to fire and the loop to re-check should_stop, then drain the queue.

    With the bug: item 3 is dropped (we get only [1, 2]).
    With the fix: item 3 stays in the put loop until we drain item 1 or 2,
    then lands in the queue (we get [1, 2, 3]).
    """
    from pipe.shm import _item_from_shm

    p = Pipe(sequential=False, stats_interval=0, health_check_interval=0)
    p.add(FastSeqRoot(n=20), workers=1, outqn=2)
    p.start()

    # Wait for spawn (~2s) and root to fill its queue + block in put-retry on item 3.
    time.sleep(3.0)

    # Signal graceful stop. Without the fix, the put-retry loop will exit on
    # its next iteration (after the 0.1s put timeout) and drop the in-flight item.
    p.should_stop.value = 1

    # Wait LONGER than the put timeout (0.1s) so the buggy loop has time to
    # observe should_stop and drop. With the fix, the loop keeps retrying.
    time.sleep(0.5)

    # Now drain the queue. As we pull, free space lets the (still-blocked-with-fix)
    # put loop deliver its in-flight item.
    results = []
    raws = []
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            raw = p.queues[-1].get(timeout=0.3)
        except Empty:
            if all(not proc.is_alive() for proc in p.processes):
                break
            continue
        raws.append(repr(raw)[:80])
        item = _item_from_shm(raw)
        if item is End or item is None:
            continue
        results.append(item)
        if len(results) >= 3:
            # We only need to verify item 3 made it through.
            break
    alive_after = [p.is_alive() for p in p.processes]
    p._stop(force=True)

    ids = sorted(r for r in results if isinstance(r, int))
    assert ids, f"no items delivered. raws={raws}, alive_after={alive_after}"
    # With outqn=2 the queue will have held items 1 and 2; item 3 was the one
    # blocked in put-retry. If 3 is missing, the in-flight item was dropped.
    assert 1 in ids and 2 in ids, f"queue should have held items 1,2; got {ids}"
    assert 3 in ids, f"in-flight item 3 was dropped from put-retry; got {ids}"
    print(f"PASS: items {ids} delivered, in-flight item 3 not dropped")


