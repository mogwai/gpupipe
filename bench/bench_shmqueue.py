"""Raw queue throughput: torch.multiprocessing.Queue vs pipe's ShmQueue.

    python bench/bench_shmqueue.py           # items: P producers / C consumers
    python bench/bench_shmqueue.py tensors   # fresh CPU tensors, 1 -> 1

Items are pickled dicts like pipe's metadata items with an inline bytes pad.
ShmQueue gets a payload store, as every pipeline edge does, so messages too
big for the ring travel in recycled store blocks.
Tensor rows send a fresh tensor per item (as a real stage does): mp.Queue
moves each one to shm and passes an fd; ShmQueue inlines tensors up to
PIPE_INLINE_TENSOR_BYTES and uses torch's fd path above that.
"""
import sys
import time

import torch
import torch.multiprocessing as mp

from pipe.shmqueue import ShmQueue
from pipe.shmstore import PayloadStore


# Producers outlive the consumers' reads (`finished`): under torch's
# file_descriptor strategy a consumer fetches each tensor's fd from the
# producer's socket at get() time — the reason pipe workers park after End.
def _producer(q, n, size, go, finished):
    item = {"id": 0, "pad": b"x" * size}
    go.wait()
    for i in range(n):
        item["id"] = i
        q.put(item)
    finished.wait()


def _tensor_producer(q, n, numel, go, finished):
    t = torch.randn(numel)
    go.wait()
    for i in range(n):
        q.put({"id": i, "x": t.clone()})
    finished.wait()


def _consumer(q, n, go, done):
    go.wait()
    for _ in range(n):
        q.get()
    done.release()


def _run(q, producers, consumers):
    go, done, finished = mp.Event(), mp.Semaphore(0), mp.Event()
    procs = [mp.Process(target=f, args=(q, *a, go, finished)) for f, a in producers]
    procs += [mp.Process(target=_consumer, args=(q, n, go, done)) for n in consumers]
    for p in procs:
        p.start()
    time.sleep(1.0)
    t0 = time.perf_counter()
    go.set()
    for _ in consumers:
        done.acquire()
    dt = time.perf_counter() - t0
    finished.set()
    for p in procs:
        p.join()
    return sum(consumers) / dt


def bench_items():
    print(f"{'P':>2} {'C':>2} {'bytes':>8} {'mp.Queue/s':>12} {'ShmQueue/s':>12} {'speedup':>8}")
    store = PayloadStore()
    for P, C, size in [(1, 1, 64), (4, 4, 64), (8, 8, 64), (1, 1, 4096), (1, 1, 65536), (4, 4, 65536), (1, 1, 1 << 20)]:
        n = 200_000 if size <= 4096 else (20_000 if size <= 65536 else 2_000)
        r = [
            _run(make(), [(_producer, (n // P, size))] * P, [n // C] * C)
            for make in (lambda: mp.Queue(maxsize=256), lambda: ShmQueue(maxsize=256, store=store))
        ]
        print(f"{P:>2} {C:>2} {size:>8} {r[0]:>12,.0f} {r[1]:>12,.0f} {r[1] / r[0]:>7.1f}x", flush=True)


def bench_tensors():
    print(f"sharing strategy: {torch.multiprocessing.get_sharing_strategy()}")
    print(f"{'tensor':>8} {'mp.Queue/s':>12} {'ShmQueue/s':>12} {'speedup':>8}")
    store = PayloadStore()
    for numel in [1 << 8, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22]:
        n = 4000 if numel <= 1 << 18 else 400
        r = [
            _run(make(), [(_tensor_producer, (n, numel))], [n])
            for make in (lambda: mp.Queue(maxsize=64), lambda: ShmQueue(maxsize=64, store=store))
        ]
        kb = numel * 4 >> 10
        size = f"{kb}K" if kb < 1024 else f"{kb >> 10}M"
        print(f"{size:>8} {r[0]:>12,.0f} {r[1]:>12,.0f} {r[1] / r[0]:>7.1f}x", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    bench_tensors() if sys.argv[1:] == ["tensors"] else bench_items()
