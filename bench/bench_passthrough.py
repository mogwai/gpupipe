"""Unchanged payloads through a pipeline: a source emits {"id", "wave"} items
(numpy float32 arrays) and three stages each add a field and pass the array
on untouched — the common shape of audio/ML pipelines.

    PIPE_QUEUE=mp python bench/bench_passthrough.py    # torch.multiprocessing.Queue
    PIPE_STORE=0  python bench/bench_passthrough.py    # shm ring, arrays pickled per hop
    python bench/bench_passthrough.py                  # shm ring + payload store

Times first -> last item at the consumer; reports items/s and GB/s of array
payload delivered end to end.
"""
import os
import time

os.environ.setdefault("PIPE_DRAIN_GRACE", "0.2")

import numpy as np  # noqa: E402

from pipe import Pipe  # noqa: E402


class Source:
    def __init__(self, n, nbytes):
        self.n = n
        self.nbytes = nbytes

    def load(self):
        self.wave = np.random.rand(self.nbytes // 4).astype(np.float32)

    def __call__(self):
        for i in range(self.n):
            yield {"id": i, "wave": self.wave}


class Annotate:
    def __init__(self, key):
        self.key = key

    def __call__(self, item):
        item[self.key] = float(item["wave"][0])
        return item


def run(n, nbytes, workers):
    pipe = Pipe(stats_interval=0)
    pipe.add(Source(n, nbytes), outqn=32)
    for k in ("a", "b", "c"):
        pipe.add(Annotate(k), workers=workers, outqn=32)
    got, t0 = 0, None
    for item in pipe:
        if t0 is None:
            t0 = time.perf_counter()
        got += 1
        if got == n:
            break
    dt = time.perf_counter() - t0
    pipe.stop()
    return got / dt


if __name__ == "__main__":
    mode = "mp.Queue" if os.environ.get("PIPE_QUEUE") == "mp" else (
        "shm ring" if os.environ.get("PIPE_STORE") == "0" else "shm ring + store")
    print(f"transport: {mode}")
    print(f"{'array':>7} {'workers':>7} {'items/s':>10} {'GB/s':>7}")
    for nbytes, workers in [(64 << 10, 1), (1 << 20, 1), (1 << 20, 4), (8 << 20, 1), (8 << 20, 4)]:
        n = max(200, min(20_000, (4 << 30) // nbytes // 4))
        r = run(n, nbytes, workers)
        size = f"{nbytes >> 10}K" if nbytes < 1 << 20 else f"{nbytes >> 20}M"
        print(f"{size:>7} {workers:>7} {r:>10,.0f} {r * nbytes / 1e9:>7.2f}", flush=True)
