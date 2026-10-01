"""Per-item framework cost: tiny items through N do-nothing stages.

Times first -> last item at the consumer (spawn and shutdown excluded), so the
figure is pure pipe overhead: serialization + queue transport + worker loop.
Compare transports with PIPE_QUEUE=mp vs the default shared-memory ring:

    PIPE_QUEUE=mp python bench/bench_pipe_overhead.py
    python bench/bench_pipe_overhead.py
"""
import os
import sys
import time

os.environ.setdefault("PIPE_DRAIN_GRACE", "0.2")

from pipe import Pipe  # noqa: E402


class Source:
    def __init__(self, n, payload):
        self.n = n
        self.payload = payload

    def __call__(self):
        for i in range(self.n):
            yield {"id": i, "text": self.payload}


class Passthrough:
    def __call__(self, item):
        return item


def run(n, stages, workers, chunk, payload_bytes):
    pipe = Pipe(stats_interval=0)
    pipe.add(Source(n, "x" * payload_bytes), outqn=256, chunk=chunk)
    for _ in range(stages):
        pipe.add(Passthrough(), workers=workers, outqn=256, chunk=chunk)
    got, t0 = 0, None
    for _ in pipe:
        if t0 is None:
            t0 = time.perf_counter()
        got += 1
        if got == n:
            break
    dt = time.perf_counter() - t0
    pipe.stop()
    return got / dt


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    print(f"queue: {os.environ.get('PIPE_QUEUE', 'auto')}")
    print(f"{'stages':>6} {'workers':>7} {'chunk':>5} {'payload':>7} {'items/s':>10} {'us/item':>8}")
    for stages, workers, chunk, payload in [
        (1, 1, 0, 64), (3, 1, 0, 64), (3, 4, 0, 64),
        (1, 1, 32, 64), (3, 1, 32, 64), (3, 4, 32, 64),
        (3, 1, 0, 65536), (3, 1, 32, 65536),
    ]:
        r = run(n, stages, workers, chunk, payload)
        print(f"{stages:>6} {workers:>7} {chunk:>5} {payload:>7} {r:>10,.0f} {1e6 / r:>8.1f}", flush=True)
