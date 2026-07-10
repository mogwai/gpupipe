"""Tests for generator/yield support in Pipe workers."""
import time
import pytest
from pipe import Pipe
from conftest import Collector


class YieldingRootGenerator:
    """Root generator that uses yield instead of return."""
    def __init__(self, n_items: int):
        self.n_items = n_items

    def load(self):
        pass

    def __call__(self):
        for i in range(self.n_items):
            yield {"id": i}


class YieldingExpander:
    """Middle worker that yields multiple items per input.

    Note: Workers don't need to handle 'end' - framework does it.
    """
    def __init__(self, factor: int = 3):
        self.factor = factor

    def load(self):
        pass

    def __call__(self, item):
        for i in range(self.factor):
            yield {"id": item["id"], "sub": i}


class PassThrough:
    """Simple pass-through worker. No 'end' handling needed."""
    def load(self):
        pass

    def __call__(self, item):
        return item


def run_pipeline(stages: list[tuple], timeout: float = 60, sequential: bool = False) -> list[dict]:
    """Run pipeline and collect results."""
    pipe = Pipe(sequential=sequential, raise_errors=True, stats_interval=0, health_check_interval=0)
    for stage, kwargs in stages:
        pipe.add(stage, **kwargs)
    results = []
    start = time.time()
    for item in pipe:
        results.append(item)
        if time.time() - start > timeout:
            raise TimeoutError(f"Pipeline timeout after {timeout}s")
    return results


class TestYieldingRootGenerator:
    """Tests for root workers using yield."""

    def test_yield_root_sequential_mode(self):
        """Yielding root generator works in sequential mode."""
        n_items = 20
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 50}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=True)
        assert len(results) == n_items
        assert {r["id"] for r in results} == set(range(n_items))

    def test_yield_root_process_mode(self):
        """Yielding root generator works with process workers."""
        n_items = 50
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 100}),
            (PassThrough(), {"workers": 2, "outqn": 50}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=False)
        assert len(results) == n_items
        assert {r["id"] for r in results} == set(range(n_items))

    def test_yield_root_threaded_mode(self):
        """Yielding root generator works with threaded workers."""
        n_items = 50
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 100}),
            (PassThrough(), {"workers": 4, "outqn": 50, "thread": True}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=False)
        assert len(results) == n_items
        assert {r["id"] for r in results} == set(range(n_items))

    def test_yield_root_empty(self):
        """Yielding root generator with zero items."""
        stages = [
            (YieldingRootGenerator(0), {"outqn": 10}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=True)
        assert len(results) == 0


class TestYieldingMiddleWorker:
    """Tests for middle workers using yield."""

    def test_yield_expander_sequential_mode(self):
        """Yielding expander works in sequential mode."""
        n_items = 10
        factor = 3
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 50}),
            (YieldingExpander(factor), {"workers": 1, "outqn": 50}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=True)
        assert len(results) == n_items * factor
        for i in range(n_items):
            parent_results = [r for r in results if r["id"] == i]
            assert len(parent_results) == factor
            assert {r["sub"] for r in parent_results} == set(range(factor))

    def test_yield_expander_process_mode(self):
        """Yielding expander works with process workers."""
        n_items = 20
        factor = 3
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 100}),
            (YieldingExpander(factor), {"workers": 2, "outqn": 100}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=False)
        assert len(results) == n_items * factor
        for i in range(n_items):
            parent_results = [r for r in results if r["id"] == i]
            assert len(parent_results) == factor

    def test_yield_expander_threaded_mode(self):
        """Yielding expander works with threaded workers."""
        n_items = 20
        factor = 3
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 100}),
            (YieldingExpander(factor), {"workers": 4, "outqn": 100, "thread": True}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=False)
        assert len(results) == n_items * factor
        for i in range(n_items):
            parent_results = [r for r in results if r["id"] == i]
            assert len(parent_results) == factor


class TestCombinedYieldPatterns:
    """Tests combining yield in root and middle workers."""

    def test_yield_root_and_expander_sequential(self):
        """Both root and middle use yield - sequential mode."""
        n_items = 5
        factor = 4
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 50}),
            (YieldingExpander(factor), {"workers": 1, "outqn": 50}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=True)
        assert len(results) == n_items * factor

    def test_yield_root_and_expander_process(self):
        """Both root and middle use yield - process mode."""
        n_items = 15
        factor = 3
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 100}),
            (YieldingExpander(factor), {"workers": 2, "outqn": 100}),
            (PassThrough(), {"workers": 2, "outqn": 50}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=False)
        assert len(results) == n_items * factor

    def test_yield_root_and_expander_threaded(self):
        """Both root and middle use yield - threaded mode."""
        n_items = 15
        factor = 3
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 100}),
            (YieldingExpander(factor), {"workers": 4, "outqn": 100, "thread": True}),
            (PassThrough(), {"workers": 2, "outqn": 50, "thread": True}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=False)
        assert len(results) == n_items * factor


class TestYieldWithNone:
    """Tests that yield with None values works correctly (filtering)."""

    class FilteringExpander:
        """Yields some items, skips others with None. No 'end' handling needed."""
        def __init__(self, skip_mod: int = 2):
            self.skip_mod = skip_mod

        def load(self):
            pass

        def __call__(self, item):
            for i in range(5):
                if i % self.skip_mod == 0:
                    yield None
                else:
                    yield {"id": item["id"], "sub": i}

    def test_yield_with_none_filtering(self):
        """Yielded None values are filtered out."""
        n_items = 10
        stages = [
            (YieldingRootGenerator(n_items), {"outqn": 50}),
            (self.FilteringExpander(skip_mod=2), {"workers": 1, "outqn": 50}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages, sequential=True)
        # 5 items per input, but indices 0, 2, 4 are skipped (mod 2 == 0)
        # So we get indices 1, 3 = 2 items per input
        assert len(results) == n_items * 2


@pytest.mark.parametrize("run", range(3))
def test_yield_stress_process(run):
    """Stress test yielding with process workers - multiple runs."""
    n_items = 100
    factor = 5
    stages = [
        (YieldingRootGenerator(n_items), {"outqn": 200}),
        (YieldingExpander(factor), {"workers": 4, "outqn": 500}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, sequential=False)
    assert len(results) == n_items * factor


@pytest.mark.parametrize("run", range(3))
def test_yield_stress_threaded(run):
    """Stress test yielding with threaded workers - multiple runs."""
    n_items = 100
    factor = 5
    stages = [
        (YieldingRootGenerator(n_items), {"outqn": 200}),
        (YieldingExpander(factor), {"workers": 8, "outqn": 500, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, sequential=False)
    assert len(results) == n_items * factor
