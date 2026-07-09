"""Demo script for pipe autoscaling with realistic audio-like workload.

PLANNED FEATURE — this demo uses the autoscale=/max_workers= API, which has been
pulled from the live code path (implementation preserved in
src/pipe/_planned/autoscale.py; see PLANNED.md). It will NOT run as-is until
autoscaling is re-integrated. Kept here as a reference workload."""

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
        # Simulate network: 50-150ms (slower to trigger scaling)
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
    print("PIPE AUTOSCALING DEMO")
    print("=" * 60)
    print(f"Processing {n_items} simulated audio files (5s-120s each)")
    print()

    # Global autoscale=True enables autoscaling for all stages
    # Per-stage max_workers can still override the global max_workers_per_stage
    pipe = Pipe(
        debug=False,
        stats_interval=0.2,
        stats_mode="rich",
        health_check_interval=0,
        autoscale=True,
        max_workers_per_stage=6,
    )
    pipe.add(AudioGenerator(n_items), outqn=20)
    pipe.add(Downloader(), workers=2, outqn=30)  # Will autoscale 1-6 (global settings)
    pipe.add(Processor(), workers=1, outqn=30, max_workers=4)  # Will autoscale 1-4 (custom max)
    pipe.add(Collector(), workers=1, outqn=0)

    start = time.time()
    results = []
    worker_snapshots = []

    for i, item in enumerate(pipe):
        results.append(item)
        dl_workers = pipe.stage_worker_counts[1].value
        proc_workers = pipe.stage_worker_counts[2].value
        worker_snapshots.append((dl_workers, proc_workers))

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(
                f"  [{i + 1:3d}/{n_items}] {elapsed:5.1f}s | downloaders={dl_workers} processors={proc_workers}"
            )

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Items processed: {len(results)}/{n_items}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Throughput: {len(results) / elapsed:.1f} items/sec")
    print()

    # Worker history
    dl_history = [s[0] for s in worker_snapshots]
    proc_history = [s[1] for s in worker_snapshots]
    print(f"Downloader workers: min={min(dl_history)} max={max(dl_history)} final={dl_history[-1]}")
    print(
        f"Processor workers:  min={min(proc_history)} max={max(proc_history)} final={proc_history[-1]}"
    )

    dl_changes = sum(1 for i in range(1, len(dl_history)) if dl_history[i] != dl_history[i - 1])
    proc_changes = sum(
        1 for i in range(1, len(proc_history)) if proc_history[i] != proc_history[i - 1]
    )
    print(f"Scaling events: downloaders={dl_changes} processors={proc_changes}")
