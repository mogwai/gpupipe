"""Shared worker classes and helper functions for pipe tests."""
import time
import hashlib
import random
import pytest

# planned/ holds suites for preserved-but-unwired features (see PLANNED.md).
# collect_ignore is anchored to this conftest's directory, so the exclusion
# holds from any invocation cwd (unlike addopts --ignore, which is
# cwd-relative). The suites also carry their own module-level skip marks.
collect_ignore = ["planned"]

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from pipe import Pipe, End


# === BASIC WORKER CLASSES ===

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


class FilterWorker:
    """Drops items where id is odd."""
    def load(self):
        pass

    def __call__(self, item):
        if item["id"] % 2 == 0:
            return item
        return None


class ExpandWorker:
    """Expands each item into multiple."""
    def __init__(self, factor: int = 3):
        self.factor = factor

    def load(self):
        pass

    def __call__(self, item):
        return [{"id": item["id"], "sub": i, "original_id": item["id"]} for i in range(self.factor)]


class Batcher:
    """Batches items, with flush support."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = []

    def load(self):
        pass

    def __call__(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.size:
            result = self.buffer.copy()
            self.buffer = []
            return result
        return None

    def flush(self):
        if self.buffer:
            result = self.buffer.copy()
            self.buffer = []
            for item in result:
                yield item


class FlushingBatcher:
    """Batcher that uses flush() mechanism."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = []

    def load(self):
        pass

    def __call__(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.size:
            result = self.buffer.copy()
            self.buffer = []
            return result
        return None

    def flush(self):
        if self.buffer:
            result = self.buffer.copy()
            self.buffer = []
            for item in result:
                yield item


class Collector:
    """Simple pass-through collector."""
    def load(self):
        pass

    def __call__(self, item):
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
            result = self.buffer
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


# === REALISTIC WORKLOAD CLASSES ===

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
            return End

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
            return [item, End]
        return item


class DownloadSimulator:
    """Simulates S3 download with variable latency based on file size."""
    def __init__(self, bytes_per_second: float = 50_000_000):
        self.bytes_per_second = bytes_per_second

    def load(self):
        pass

    def __call__(self, item):
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
        parent_id = item["parent_id"]
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

    def flush(self):
        for parent_id, data in self.pending.items():
            yield {
                "id": parent_id,
                "complete": False,
                "chunks_received": len(data["chunks"]),
                "chunks_expected": data["n_chunks"],
            }
        self.pending = {}


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


# === TENSOR WORKER CLASSES (require torch) ===

class TensorGenerator:
    """Generator that produces tensors of various sizes."""
    def __init__(self, n_items: int, tensor_size: tuple = (64, 128)):
        self.n_items = n_items
        self.tensor_size = tensor_size
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        item = {
            "id": self._idx,
            "tensor": torch.randn(self.tensor_size),
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class TensorProcessor:
    """Worker that performs compute-intensive tensor operations."""
    def __init__(self, iterations: int = 10):
        self.iterations = iterations

    def load(self):
        pass

    def __call__(self, item):
        tensor = item["tensor"]
        for _ in range(self.iterations):
            tensor = tensor @ tensor.T
            tensor = torch.softmax(tensor, dim=-1)
        item["processed"] = True
        item["result_shape"] = tuple(tensor.shape)
        return item


class TensorBatcher:
    """Batches tensors by stacking them."""
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.buffer = []

    def load(self):
        pass

    def __call__(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            return self._make_batch()
        return None

    def _make_batch(self):
        ids = [item["id"] for item in self.buffer]
        tensors = torch.stack([item["tensor"] for item in self.buffer])
        self.buffer = []
        return {"ids": ids, "batch_tensor": tensors}

    def flush(self):
        if self.buffer:
            yield self._make_batch()


class BatchTensorProcessor:
    """Processes batched tensors with compute-intensive operations."""
    def __init__(self, iterations: int = 5):
        self.iterations = iterations

    def load(self):
        pass

    def __call__(self, item):
        batch_tensor = item["batch_tensor"]
        for _ in range(self.iterations):
            batch_tensor = torch.nn.functional.normalize(batch_tensor, dim=-1)
            batch_tensor = batch_tensor * 2 - 1
        item["processed"] = True
        return item


class TensorUnbatcher:
    """Unbatches tensor batches back to individual items."""
    def load(self):
        pass

    def __call__(self, item):
        ids = item["ids"]
        batch_tensor = item["batch_tensor"]
        results = []
        for i, id_ in enumerate(ids):
            results.append({"id": id_, "tensor": batch_tensor[i], "processed": True})
        return results


class LargeTensorGenerator:
    """Generator producing larger tensors to stress memory/compute."""
    def __init__(self, n_items: int, tensor_size: tuple = (256, 256)):
        self.n_items = n_items
        self.tensor_size = tensor_size
        self._idx = 0

    def load(self):
        pass

    def __call__(self):
        if self._idx >= self.n_items:
            return End
        item = {
            "id": self._idx,
            "tensor": torch.randn(self.tensor_size),
            "metadata": {"size": self.tensor_size, "dtype": "float32"},
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class HeavyTensorProcessor:
    """Simulates heavy compute on tensors (like model inference)."""
    def __init__(self, compute_ms: float = 50):
        self.compute_ms = compute_ms

    def load(self):
        pass

    def __call__(self, item):
        tensor = item["tensor"]
        start = time.time()
        while (time.time() - start) * 1000 < self.compute_ms:
            tensor = torch.nn.functional.relu(tensor)
            tensor = torch.nn.functional.normalize(tensor, dim=-1)
        item["processed"] = True
        item["compute_ms"] = (time.time() - start) * 1000
        return item


# === HELPER FUNCTIONS ===

def run_pipeline(stages: list[tuple], timeout: float = 60, health_check: int = 0) -> list[dict]:
    """Run pipeline and collect results."""
    pipe = Pipe(debug=False, raise_errors=True, stats_interval=0, health_check_interval=health_check)
    for stage, kwargs in stages:
        pipe.add(stage, **kwargs)
    results = []
    start = time.time()
    for item in pipe:
        results.append(item)
        if time.time() - start > timeout:
            raise TimeoutError(f"Pipeline timeout after {timeout}s")
    return results
