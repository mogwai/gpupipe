"""Queue wrappers: an instrumented multiprocessing Queue (transit-latency stats)
and the lightweight iterator used to read a pipeline's final output queue."""
import time
from queue import Empty

from torch.multiprocessing import Value

from .shm import _item_from_shm
from .types import End


class InstrumentedQueue:
    def __init__(self, queue):
        self._queue = queue
        self.items_put = Value('l', 0)
        self.items_got = Value('l', 0)
        self.total_transit = Value('d', 0.0)

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

    def get_nowait(self):
        item, put_time = self._queue.get_nowait()
        transit = time.time() - put_time
        with self.total_transit.get_lock():
            self.total_transit.value += transit
        with self.items_got.get_lock():
            self.items_got.value += 1
        return item

    def put_nowait(self, item):
        self._queue.put_nowait((item, time.time()))
        with self.items_put.get_lock():
            self.items_put.value += 1

    def qsize(self):
        return self._queue.qsize()

    def full(self):
        return self._queue.full()

    def empty(self):
        return self._queue.empty()

    def close(self):
        return self._queue.close()

    def cancel_join_thread(self):
        return self._queue.cancel_join_thread()

    @property
    def _maxsize(self):
        return self._queue._maxsize


class PipeIterator:
    """Lightweight iterator that reads from a multiprocessing queue."""

    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self.queue.get(timeout=1.0)
                item = _item_from_shm(item)
                if item is End:
                    raise StopIteration
                return item
            except Empty:
                continue
