"""Complex pipeline pattern tests - deep pipelines, multi-stage, realistic workloads.

Run with: pytest test_pipeline_patterns.py -n auto
"""
import time
import hashlib
import random

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from conftest import (
    Generator,
    BatchGenerator,
    SlowWorker,
    FastWorker,
    Collector,
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

class AudioSegmentGenerator:
    """Simulates database retrieval of audio segments with variable lengths (10s to 40min)."""
    def __init__(self, n_items: int, min_duration: float = 10.0, max_duration: float = 600.0):
        self.n_items = n_items
        self.min_duration = min_duration
        self.max_duration = max_duration
        self._idx = 0
        random.seed(42)

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return "end"

        duration = random.expovariate(1 / 60)
        duration = max(self.min_duration, min(self.max_duration, duration))

        sample_rate = 16000
        n_samples = int(duration * sample_rate)

        item = {
            "id": self._idx,
            "duration": duration,
            "n_samples": n_samples,
            "source_url": f"s3://bucket/audio_{self._idx}.wav",
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, "end"]
        return item


class DownloadSimulator:
    """Simulates S3 download with variable latency based on file size."""
    def __init__(self, bytes_per_second: float = 50_000_000):
        self.bytes_per_second = bytes_per_second

    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            return item

        n_bytes = item["n_samples"] * 2
        download_time = n_bytes / self.bytes_per_second
        download_time *= random.uniform(0.8, 1.5)
        time.sleep(min(download_time, 0.5))

        item["audio_bytes"] = bytes(random.getrandbits(8) for _ in range(min(n_bytes, 10000)))
        return item


class AudioChunker:
    """Chunks long audio into fixed-size segments (like Fluac does)."""
    def __init__(self, chunk_seconds: float = 30.0, overlap_seconds: float = 1.0):
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            return item

        duration = item["duration"]
        n_chunks = max(1, int(duration / self.chunk_seconds))

        chunks = []
        for i in range(n_chunks):
            chunk = {
                "parent_id": item["id"],
                "chunk_idx": i,
                "n_chunks": n_chunks,
                "chunk_duration": min(self.chunk_seconds, duration - i * self.chunk_seconds),
                "audio_hash": hashlib.sha256(item.get("audio_bytes", b"") + str(i).encode()).hexdigest()[:16],
            }
            chunks.append(chunk)

        return chunks


class GPUBatcher:
    """Collects items into batches for GPU processing."""
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.buffer = []

    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            if self.buffer:
                batch = self.buffer.copy()
                self.buffer = []
                return [{"batch": batch, "batch_size": len(batch)}, "end"]
            return "end"

        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            batch = self.buffer.copy()
            self.buffer = []
            return {"batch": batch, "batch_size": len(batch)}
        return None

    def flush(self):
        if self.buffer:
            batch = self.buffer.copy()
            self.buffer = []
            yield {"batch": batch, "batch_size": len(batch)}


class GPUEncoder:
    """Simulates GPU encoding with variable processing time based on batch."""
    def __init__(self, ms_per_second_audio: float = 5.0):
        self.ms_per_second_audio = ms_per_second_audio
        self.model = None

    def load(self):
        if HAS_TORCH:
            self.model = torch.nn.Linear(256, 256).cuda() if torch.cuda.is_available() else torch.nn.Linear(256, 256)
        time.sleep(0.1)

    def __call__(self, item):
        if item == "end":
            return item

        batch = item["batch"]
        total_duration = sum(c.get("chunk_duration", 1.0) for c in batch)

        process_time = (total_duration * self.ms_per_second_audio) / 1000

        if HAS_TORCH and self.model is not None:
            x = torch.randn(len(batch), 256)
            if torch.cuda.is_available():
                x = x.cuda()
            for _ in range(int(process_time * 100)):
                x = self.model(x)
                x = torch.relu(x)
        else:
            time.sleep(process_time)

        results = []
        for chunk in batch:
            results.append({
                "parent_id": chunk["parent_id"],
                "chunk_idx": chunk["chunk_idx"],
                "n_chunks": chunk["n_chunks"],
                "encoded": True,
                "code_hash": hashlib.sha256(chunk["audio_hash"].encode()).hexdigest()[:16],
            })
        return results


class ResultAggregator:
    """Reassembles chunks back into complete files (like Process in encode-pipe.py)."""
    def __init__(self):
        self.pending = {}
        self.completed = 0

    def load(self):
        pass

    def __call__(self, item):
        if item == "end":
            results = []
            for parent_id, data in self.pending.items():
                results.append({
                    "id": parent_id,
                    "complete": False,
                    "chunks_received": len(data["chunks"]),
                    "chunks_expected": data["n_chunks"],
                })
            self.pending = {}
            if results:
                return results + ["end"]
            return "end"

        parent_id = item["parent_id"]
        chunk_idx = item["chunk_idx"]
        n_chunks = item["n_chunks"]

        if parent_id not in self.pending:
            self.pending[parent_id] = {"chunks": [], "n_chunks": n_chunks}

        self.pending[parent_id]["chunks"].append(item)

        if len(self.pending[parent_id]["chunks"]) >= n_chunks:
            chunks = self.pending[parent_id]["chunks"]
            del self.pending[parent_id]
            self.completed += 1

            combined_hash = hashlib.sha256(
                "".join(c["code_hash"] for c in sorted(chunks, key=lambda x: x["chunk_idx"])).encode()
            ).hexdigest()[:16]

            return {
                "id": parent_id,
                "complete": True,
                "n_chunks": n_chunks,
                "combined_hash": combined_hash,
            }

        return None


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


def test_realistic_audio_pipeline_autoscale():
    """Test realistic pipeline with autoscaling on the slow stages."""
    n_items = 50

    pipe = Pipe(debug=False, stats_interval=0, health_check_interval=0)
    pipe.add(AudioSegmentGenerator(n_items, min_duration=5, max_duration=60), outqn=10)
    pipe.add(DownloadSimulator(), workers=2, outqn=20, autoscale=True, max_workers=6)
    pipe.add(AudioChunker(chunk_seconds=15), workers=1, outqn=100)
    pipe.add(GPUBatcher(batch_size=4), workers=1, outqn=30)
    pipe.add(GPUEncoder(ms_per_second_audio=10), workers=1, outqn=50, autoscale=True, max_workers=2)
    pipe.add(ResultAggregator(), workers=1, outqn=0)

    results = []
    for item in pipe:
        if isinstance(item, dict) and item.get("complete"):
            results.append(item)

    download_workers = pipe.stage_worker_counts[1].value
    gpu_workers = pipe.stage_worker_counts[4].value
    complete_count = len([r for r in results if r.get("complete")])

    print(f"Autoscale pipeline: {complete_count}/{n_items} completed, download={download_workers}, gpu={gpu_workers} workers")
    assert complete_count >= n_items * 0.9
