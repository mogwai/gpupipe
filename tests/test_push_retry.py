"""Tests for worker.push(stage, item): sending an item BACK to an earlier stage.

Models the real use case: a downstream check (WER) that, on failure, pushes the
item back to an earlier stage (render) for another attempt, bounded by a retry
counter on the item.
"""
import os

# push()-back tests need the full production drain window: with a short
# PIPE_DRAIN_GRACE (set for speed in conftest) the push target can drain and
# exit before retried items land, leaving the pusher spinning on a full queue.
os.environ["PIPE_DRAIN_GRACE"] = "3.0"

import time

from conftest import Collector, Generator

from pipe import Pipe


def _run(stages, sequential):
    pipe = Pipe(
        sequential=sequential,
        raise_errors=True,
        stats_interval=0,
        health_check_interval=0,
    )
    for stage, kwargs in stages:
        pipe.add(stage, **kwargs)
    results = []
    start = time.time()
    for item in pipe:
        results.append(item)
        if time.time() - start > 60:
            raise TimeoutError("pipeline timeout")
    return results


class Renderer:
    """Stamps how many times it has 'rendered' each item."""

    def load(self):
        pass

    def __call__(self, item):
        item["render_count"] = item.get("render_count", 0) + 1
        return item


class Checker:
    """Fails items whose id % 3 == 0 until they've been retried MAX times, then
    passes everything. A failure pushes the item back to the Renderer stage
    (index 1) with retries incremented."""

    MAX = 2

    def __init__(self, render_stage=1):
        self.render_stage = render_stage

    def load(self):
        pass

    def __call__(self, item):
        item["retries"] = item.get("retries", 0)
        if item["id"] % 3 == 0 and item["retries"] < self.MAX:
            item["retries"] += 1
            self.push(self.render_stage, item)  # back to Renderer
            return None
        return item


def _check(results, n_items):
    assert len(results) == n_items, f"{len(results)} != {n_items}"
    by_id = {r["id"]: r for r in results}
    assert set(by_id) == set(range(n_items))
    for i, r in by_id.items():
        if i % 3 == 0:
            assert r["retries"] == Checker.MAX, (i, r)
            assert r["render_count"] == Checker.MAX + 1, (i, r)  # 1 initial + MAX re-renders
        else:
            assert r["retries"] == 0, (i, r)
            assert r["render_count"] == 1, (i, r)


def test_push_retry_sequential():
    n_items = 60
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (Renderer(), {"workers": 1, "outqn": 50}),
        (Checker(render_stage=1), {"workers": 1, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = _run(stages, sequential=True)
    _check(results, n_items)


def test_push_retry_multiprocessing():
    n_items = 60
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (Renderer(), {"workers": 2, "outqn": 50}),
        (Checker(render_stage=1), {"workers": 2, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = _run(stages, sequential=False)
    _check(results, n_items)


class NamedChecker(Checker):
    """Like Checker but pushes back by stage NAME instead of index."""

    def __call__(self, item):
        item["retries"] = item.get("retries", 0)
        if item["id"] % 3 == 0 and item["retries"] < self.MAX:
            item["retries"] += 1
            self.push("Renderer", item)  # by name
            return None
        return item


def test_push_by_stage_name():
    """push() resolves a stage name to its index."""
    n_items = 30
    stages = [
        (Generator(n_items), {"workers": 1, "outqn": 50}),
        (Renderer(), {"workers": 1, "outqn": 50}),
        (NamedChecker(), {"workers": 1, "outqn": 50}),
        (Collector(), {"workers": 1, "outqn": None}),
    ]
    results = _run(stages, sequential=False)
    _check(results, n_items)
