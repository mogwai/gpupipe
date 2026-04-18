"""Test drain mode: first Ctrl+C drains queues, second force-stops."""
import time
from pipe import Pipe, End


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


