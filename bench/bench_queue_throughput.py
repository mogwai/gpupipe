"""Benchmark queue put/get throughput with varying worker counts.

Measures raw ops/sec for multiprocessing.Queue with 1, 4, 8, 16 producer/consumer pairs.
Also compares InstrumentedQueue overhead vs plain Queue.
"""
import time
from multiprocessing import Process, Queue, Value, Event


def producer(q, n_items, start_event):
    start_event.wait()
    for i in range(n_items):
        q.put(i)


def consumer(q, n_items, start_event, done_count):
    start_event.wait()
    for _ in range(n_items):
        q.get()
    with done_count.get_lock():
        done_count.value += 1


def bench_plain_queue(n_workers, items_per_worker=5000, queue_size=100):
    q = Queue(maxsize=queue_size)
    start_event = Event()
    done_count = Value("i", 0)
    total_items = n_workers * items_per_worker

    producers = [Process(target=producer, args=(q, items_per_worker, start_event)) for _ in range(n_workers)]
    consumers = [Process(target=consumer, args=(q, items_per_worker, start_event, done_count)) for _ in range(n_workers)]

    for p in producers + consumers:
        p.start()

    time.sleep(0.1)
    t0 = time.monotonic()
    start_event.set()

    for p in producers + consumers:
        p.join(timeout=30)

    elapsed = time.monotonic() - t0
    ops = total_items / elapsed
    return ops, elapsed


class InstrumentedQueue:
    """Copy of pipe's InstrumentedQueue for isolated benchmarking."""
    def __init__(self, queue):
        self._queue = queue
        self.items_put = Value("l", 0)
        self.items_got = Value("l", 0)
        self.total_transit = Value("d", 0.0)

    def put(self, item, **kwargs):
        self._queue.put((item, time.time()), **kwargs)
        with self.items_put.get_lock():
            self.items_put.value += 1

    def get(self, **kwargs):
        item, put_time = self._queue.get(**kwargs)
        transit = time.time() - put_time
        with self.total_transit.get_lock():
            self.total_transit.value += transit
        with self.items_got.get_lock():
            self.items_got.value += 1
        return item

    def full(self):
        return self._queue.full()


def instrumented_producer(q, n_items, start_event):
    start_event.wait()
    for i in range(n_items):
        q.put(i)


def instrumented_consumer(q, n_items, start_event, done_count):
    start_event.wait()
    for _ in range(n_items):
        q.get()
    with done_count.get_lock():
        done_count.value += 1


def bench_instrumented_queue(n_workers, items_per_worker=5000, queue_size=100):
    q = InstrumentedQueue(Queue(maxsize=queue_size))
    start_event = Event()
    done_count = Value("i", 0)
    total_items = n_workers * items_per_worker

    producers = [Process(target=instrumented_producer, args=(q, items_per_worker, start_event)) for _ in range(n_workers)]
    consumers = [Process(target=instrumented_consumer, args=(q, items_per_worker, start_event, done_count)) for _ in range(n_workers)]

    for p in producers + consumers:
        p.start()

    time.sleep(0.1)
    t0 = time.monotonic()
    start_event.set()

    for p in producers + consumers:
        p.join(timeout=30)

    elapsed = time.monotonic() - t0
    ops = total_items / elapsed
    return ops, elapsed


if __name__ == "__main__":
    worker_counts = [1, 4, 8, 16]

    print("=== Queue Throughput Benchmark ===\n")
    print(f"{'workers':>8} {'plain ops/s':>14} {'instr ops/s':>14} {'overhead':>10}")
    print("-" * 50)

    for n in worker_counts:
        plain_ops, plain_t = bench_plain_queue(n)
        instr_ops, instr_t = bench_instrumented_queue(n)
        overhead = (1 - instr_ops / plain_ops) * 100
        print(f"{n:>8} {plain_ops:>14,.0f} {instr_ops:>14,.0f} {overhead:>9.1f}%")

    print()
