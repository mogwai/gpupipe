from conftest import Generator, Batcher, SlowWorker, Collector, run_pipeline


def test_basic_pipeline():
    """Basic pipeline with single workers."""
    n_items = 100
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (SlowWorker(0.001), {"workers": 1, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_multi_worker_stage():
    """Pipeline with multiple workers at one stage."""
    n_items = 100
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (SlowWorker(0.02), {"workers": 2, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_multiple_multi_worker_stages():
    """Multiple stages with multiple workers each."""
    n_items = 100
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 50}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))
