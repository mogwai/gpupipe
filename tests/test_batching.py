"""Batching tests for pipe framework.

Run with: pytest test_batching.py -n auto
"""

from conftest import (
    Batcher,
    BatchGenerator,
    Collector,
    ExpandWorker,
    FastWorker,
    FlushingBatcher,
    Generator,
    SlowWorker,
    run_pipeline,
)


def test_partial_batch_flush():
    """25 items, batch size 10 - last 5 need flush."""
    n_items = 25
    stages = [
        (BatchGenerator(n_items, 5), {"outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_partial_batch_flush_process():
    """25 items, batch size 10 - last 5 need flush with process workers."""
    n_items = 25
    stages = [
        (BatchGenerator(n_items, 5), {"outqn": 50}),
        (Batcher(10), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_flushing_batcher():
    """Worker that uses flush() mechanism."""
    n_items = 55
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (FlushingBatcher(10), {"workers": 1, "outqn": 50}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_large_batch_small_data():
    """Batch size larger than total items."""
    n_items = 5
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (Batcher(100), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_large_batch_small_data_process():
    """Batch size larger than total items with process workers."""
    n_items = 5
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (Batcher(100), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_consecutive_batchers():
    """Batcher -> Batcher with different sizes."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 10), {"outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_consecutive_batchers_process():
    """Batcher -> Batcher with different sizes with process workers."""
    n_items = 100
    stages = [
        (BatchGenerator(n_items, 10), {"outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_queue_size_one():
    """Maximum backpressure - queue size 1."""
    n_items = 20
    stages = [
        (Generator(n_items), {"outqn": 1}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 1, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_queue_size_one_process():
    """Maximum backpressure - queue size 1 with process workers."""
    n_items = 20
    stages = [
        (Generator(n_items), {"outqn": 1}),
        (SlowWorker(0.01), {"workers": 2, "outqn": 1, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_expand_then_batch():
    """Expander -> Batcher."""
    n_items = 10
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (ExpandWorker(3), {"workers": 1, "outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items * 3  # 10 * 3 = 30


def test_expand_then_batch_process():
    """Expander -> Batcher with process workers."""
    n_items = 10
    stages = [
        (Generator(n_items), {"outqn": 50}),
        (ExpandWorker(3), {"workers": 1, "outqn": 50}),
        (Batcher(7), {"workers": 1, "outqn": 50}),
        (FastWorker(), {"workers": 2, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    assert len(results) == n_items * 3  # 10 * 3 = 30
