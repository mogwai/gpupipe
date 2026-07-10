"""Complex pipeline pattern tests - deep pipelines, multi-stage, realistic workloads.

Run with: pytest test_pipeline_patterns.py -n auto
"""
from conftest import (
    AudioChunker,
    AudioSegmentGenerator,
    BatchGenerator,
    Collector,
    DownloadSimulator,
    FastWorker,
    Generator,
    GPUBatcher,
    GPUEncoder,
    ResultAggregator,
    SlowWorker,
    run_pipeline,
)

from pipe import Pipe

# === DEEP PIPELINE TESTS ===

def test_deep_pipeline():
    """10 stages - end must propagate through all."""
    n_items = 50
    stages = [(Generator(n_items), {"outqn": 30})]
    for _ in range(8):
        stages.append((FastWorker(), {"workers": 2, "outqn": 30, "thread": True}))
    stages.append((Collector(), {"workers": 1, "outqn": 0}))

    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


def test_deep_pipeline_process():
    """10 stages with process workers - end must propagate through all."""
    n_items = 50
    stages = [(Generator(n_items), {"outqn": 30})]
    for _ in range(8):
        stages.append((FastWorker(), {"workers": 2, "outqn": 30, "thread": False}))
    stages.append((Collector(), {"workers": 1, "outqn": 0}))

    results = run_pipeline(stages)
    assert len(results) == n_items
    assert {r["id"] for r in results} == set(range(n_items))


# === MULTI-STAGE FAST TO SLOW TESTS ===

def test_multi_stage_fast_to_slow_process():
    """Multiple stages: fast generator -> slow stage -> slower stage (process workers)."""
    n_items = 30
    stages = [
        (BatchGenerator(n_items, 30), {"outqn": 50}),
        (SlowWorker(0.02), {"workers": 2, "outqn": 50, "thread": False}),
        (SlowWorker(0.05), {"workers": 4, "outqn": 50, "thread": False}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


def test_multi_stage_fast_to_slow_threaded():
    """Multiple stages: fast generator -> slow stage -> slower stage (threaded workers)."""
    n_items = 30
    stages = [
        (BatchGenerator(n_items, 30), {"outqn": 50}),
        (SlowWorker(0.02), {"workers": 2, "outqn": 50, "thread": True}),
        (SlowWorker(0.05), {"workers": 4, "outqn": 50, "thread": True}),
        (Collector(), {"workers": 1, "outqn": 0}),
    ]
    results = run_pipeline(stages)
    expected = set(range(n_items))
    got = {r["id"] for r in results}
    assert len(results) == n_items, f"Missing items: {expected - got}"
    assert got == expected


# === REALISTIC AUDIO PIPELINE WORKERS ===


# === REALISTIC AUDIO PIPELINE TESTS ===

def test_realistic_audio_pipeline():
    """Test a realistic audio processing pipeline like encode-pipe.py."""
    n_items = 30

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(AudioSegmentGenerator(n_items, min_duration=10, max_duration=120), outqn=5)
    pipe.add(DownloadSimulator(), workers=3, outqn=10)
    pipe.add(AudioChunker(chunk_seconds=30), workers=2, outqn=50)
    pipe.add(GPUBatcher(batch_size=8), workers=1, outqn=20)
    pipe.add(GPUEncoder(), workers=1, outqn=30)
    pipe.add(ResultAggregator(), workers=1, outqn=0)

    results = []
    for item in pipe:
        if isinstance(item, dict) and item.get("complete"):
            results.append(item)

    complete_count = len([r for r in results if r.get("complete")])
    print(f"Realistic pipeline: {complete_count}/{n_items} files completed")
    assert complete_count >= n_items * 0.9, f"Too few completions: {complete_count}"


# test_realistic_audio_pipeline_autoscale was removed with the autoscaling
# feature; restore it from git history when re-integrating (see PLANNED.md).
