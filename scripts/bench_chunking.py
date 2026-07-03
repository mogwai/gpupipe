"""Before/after benchmark for chunked queue transport.

Measures wall time to move N small metadata dicts through a 3-stage pipeline
(root -> batch worker -> consumer), with the root->batch edge chunked (auto,
via downstream batch=) vs forced off (chunk=0). The worker does no real work,
so the difference is almost pure queue-plumbing cost.

Run:  uv run python scripts/bench_chunking.py [N]
"""
import sys
import time

from pipe import End, Pipe


class Gen:
    def __init__(self, n):
        self.n = n
        self._i = 0

    def __call__(self):
        # emit in python-loop chunks of 100 to keep the root from being the bottleneck
        if self._i >= self.n:
            return End
        out = []
        for _ in range(min(100, self.n - self._i)):
            out.append({"id": self._i, "s3_key": "bucket/key/abcdef0123456789.wav",
                        "duration": 12.34, "retries": 0})
            self._i += 1
        return out


class BatchNop:
    def __call__(self, batch):
        return batch


def run_once(n, chunk):
    pipe = Pipe(stats_interval=0, health_check_interval=0)
    pipe.add(Gen(n), outqn=4096, chunk=chunk)   # chunk=None -> auto (batch below); 0 -> off
    # Final edge feeds the consumer (no batch=), so auto stays off — chunk it
    # explicitly in chunked mode.
    pipe.add(BatchNop(), workers=2, batch=64, outqn=4096,
             chunk=(64 if chunk is None else chunk))
    count = 0
    t_first = None
    for _ in pipe:
        if t_first is None:
            t_first = time.time()   # steady state: exclude process spawn/torch import
        count += 1
    dt = time.time() - t_first
    assert count == n, f"lost items: {count}/{n}"
    return dt


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    print(f"N={n:,} small dicts, 3-stage pipeline (root -> batch=64 x2 -> consumer)")
    print("(steady-state: timed from first item out, spawn excluded)")
    for label, chunk in (("chunked (auto=64)", None), ("unchunked (chunk=0)", 0)):
        best = min(run_once(n, chunk) for _ in range(2))
        print(f"  {label:22s} {best:6.2f}s  = {n / best / 1e3:8.1f}k items/s")


if __name__ == "__main__":
    main()
