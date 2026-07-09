"""Demo: multi-stage parallel pipeline on a realistic audio-like workload.

Simulates fetch -> download -> CPU-bound process -> collect with fixed
per-stage worker counts. (An autoscaling variant of this workload lives in
examples/planned/autoscale_demo.py — that one needs the planned autoscaling
feature; see PLANNED.md.)
"""

import hashlib
import random
import time

from pipe import End, Pipe


class AudioGenerator:
    """Simulates fetching audio files of varying lengths."""

    def __init__(self, n_items: int = 100):
        self.n_items = n_items
        self._idx = 0
        random.seed(42)

    def load(self):
        print(f"Generator ready to produce {self.n_items} items")

    def __call__(self):
        if self._idx >= self.n_items:
            return End

        # Random duration 5s to 120s (biased shorter)
        duration = random.expovariate(1 / 20)
        duration = max(5, min(120, duration))

        item = {
            "id": self._idx,
            "duration": duration,
            "data": bytes(random.getrandbits(8) for _ in range(1000)),
        }
        self._idx += 1
        if self._idx >= self.n_items:
            return [item, End]
        return item


class Downloader:
    """Simulates downloading with variable latency."""

    def load(self):
        print("Downloader ready")

    def __call__(self, item):
        # Simulate network: 50-150ms
        time.sleep(random.uniform(0.05, 0.15))
        item["downloaded"] = True
        return item


class Processor:
    """CPU-bound processing that scales with duration."""

    def load(self):
        print("Processor ready")

    def __call__(self, item):
        # Processing time proportional to duration: ~5ms per second of audio
        process_time = item["duration"] * 0.005
        time.sleep(process_time)
        item["hash"] = hashlib.sha256(item["data"]).hexdigest()[:16]
        item["processed"] = True
        return item


class Collector:
    def load(self):
        pass

    def __call__(self, item):
        return item


if __name__ == "__main__":
    n_items = 100

    print("=" * 60)
    print("PIPE DEMO")
    print("=" * 60)
    print(f"Processing {n_items} simulated audio files (5s-120s each)")
    print()

    pipe = Pipe(
        debug=False,
        stats_interval=0.2,
        stats_mode="rich",
        health_check_interval=0,
    )
    pipe.add(AudioGenerator(n_items), outqn=20)
    pipe.add(Downloader(), workers=4, outqn=30)  # I/O-bound: more workers
    pipe.add(Processor(), workers=2, outqn=30)  # CPU-bound
    pipe.add(Collector(), workers=1, outqn=0)

    start = time.time()
    results = []

    for i, item in enumerate(pipe):
        results.append(item)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(f"  [{i + 1:3d}/{n_items}] {elapsed:5.1f}s")

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Items processed: {len(results)}/{n_items}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Throughput: {len(results) / elapsed:.1f} items/sec")
