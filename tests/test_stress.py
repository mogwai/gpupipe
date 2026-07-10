"""Stress tests for pipe framework.

Run with: pytest tests/test_stress.py -n auto
"""
import pytest
from conftest import Batcher, BatchGenerator, Collector, FastWorker, Generator, SlowWorker, run_pipeline


@pytest.mark.parametrize("run", range(5))
def test_fast_generator_slow_workers_threaded(run):
    """Generator finishes immediately, slow workers must process all items."""
    n_items = 50
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 100}),
        (SlowWorker(0.05), {"workers": 4, "outqn": 100, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_fast_generator_slow_workers_process(run):
    """Generator finishes immediately, slow process workers must process all items."""
    n_items = 50
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 100}),
        (SlowWorker(0.05), {"workers": 4, "outqn": 100, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_instant_dump_very_slow_workers_threaded(run):
    """All items dumped instantly, workers take 100ms each - tests end signal timing."""
    n_items = 20
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 50}),
        (SlowWorker(0.1), {"workers": 4, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=30)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


@pytest.mark.parametrize("run", range(5))
def test_instant_dump_very_slow_workers_process(run):
    """All items dumped instantly, process workers take 100ms each - tests end signal timing."""
    n_items = 20
    stages = [
        (BatchGenerator(n_items, 20), {"outqn": 50}),
        (SlowWorker(0.1), {"workers": 4, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=30)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


def test_stress_slow_workers():
    """Slow workers to maximize race condition window."""
    n_items = 50
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 100}),
        (SlowWorker(0.1), {"workers": 2, "outqn": 100}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items


def test_stress_many_items():
    """Stress test with many items."""
    n_items = 500
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 100}),
        (Batcher(32), {"workers": 1, "outqn": 100}),
        (SlowWorker(0.05), {"workers": 2, "outqn": 100}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_stress_large_threaded():
    """Stress test with many items and threaded workers."""
    n_items = 1000
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 200}),
        (SlowWorker(0.005), {"workers": 8, "outqn": 200, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=120)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_stress_large_process():
    """Stress test with many items and process workers."""
    n_items = 1000
    stages = [
        (BatchGenerator(n_items, 50), {"outqn": 200}),
        (SlowWorker(0.005), {"workers": 8, "outqn": 200, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages, timeout=120)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_many_pipelines_sequential():
    """Create 20 pipelines sequentially - test cleanup."""
    for _ in range(20):
        stages = [
            (Generator(10), {"outqn": 10}),
            (FastWorker(), {"workers": 2, "outqn": 10, "thread": True}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages)
        assert len(results) == 10


def test_many_pipelines_sequential_process():
    """Create 20 pipelines sequentially with process workers - test cleanup."""
    for _ in range(20):
        stages = [
            (Generator(10), {"outqn": 10}),
            (FastWorker(), {"workers": 2, "outqn": 10, "thread": False}),
            (Collector(), {"workers": 1, "outqn": 0}),
        ]
        results = run_pipeline(stages)
        assert len(results) == 10
