"""Benchmark batch accumulation effectiveness.

Compares old approach (get_nowait, breaks on first Empty) vs new approach
(50ms deadline with get(timeout=10ms) retries). Measures actual batch sizes
achieved at different producer rates.
"""
import time
from multiprocessing import Process, Queue, Event
from queue import Empty


def slow_producer(q, n_items, interval, start_event):
    """Produces items with a delay between each."""
    start_event.wait()
    for i in range(n_items):
        q.put(i)
        if interval > 0:
            time.sleep(interval)


def batch_old_style(q, batch_size, timeout=0.5):
    """Old: get first item with timeout, then get_nowait for rest."""
    try:
        item = q.get(timeout=timeout)
    except Empty:
        return None
    batch = [item]
    while len(batch) < batch_size:
        try:
            raw = q.get_nowait()
        except Empty:
            break
        batch.append(raw)
    return batch


def batch_new_style(q, batch_size, timeout=0.5):
    """New: get first item with timeout, then 50ms deadline with get(timeout=10ms)."""
    try:
        item = q.get(timeout=timeout)
    except Empty:
        return None
    batch = [item]
    deadline = time.monotonic() + 0.05
    while len(batch) < batch_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            raw = q.get(timeout=min(remaining, 0.01))
        except Empty:
            if time.monotonic() >= deadline:
                break
            continue
        batch.append(raw)
    return batch


def run_bench(batch_fn, q, batch_size, n_batches):
    sizes = []
    for _ in range(n_batches):
        batch = batch_fn(q, batch_size)
        if batch is None:
            break
        sizes.append(len(batch))
    avg = sum(sizes) / len(sizes) if sizes else 0
    fill = avg / batch_size * 100
    return avg, fill, len(sizes)


if __name__ == "__main__":
    batch_size = 16
    n_items = 500
    # Different producer intervals simulate different upstream speeds
    # 0 = as fast as possible, 1ms = moderate, 5ms = slow
    intervals = [0, 0.001, 0.003, 0.005]

    print("=== Batch Accumulation Benchmark ===\n")
    print(f"batch_size={batch_size}, items={n_items}\n")
    print(f"{'interval':>10} {'method':>10} {'avg_batch':>10} {'fill%':>8} {'batches':>8}")
    print("-" * 50)

    for interval in intervals:
        for label, batch_fn in [("old", batch_old_style), ("new", batch_new_style)]:
            q = Queue(maxsize=200)
            start_event = Event()

            p = Process(target=slow_producer, args=(q, n_items, interval, start_event))
            p.start()
            start_event.set()

            # let some items accumulate
            time.sleep(0.05)

            avg, fill, n_batches = run_bench(batch_fn, q, batch_size, n_items)
            print(f"{interval*1000:>8.1f}ms {label:>10} {avg:>10.1f} {fill:>7.0f}% {n_batches:>8}")

            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        print()
