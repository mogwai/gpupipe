"""Queue wrappers: an instrumented multiprocessing Queue (transit-latency stats),
the chunked-transport primitives (Chunk / _OutputChannel / _InputChannel), and the
lightweight iterator used to read a pipeline's final output queue."""
import time
from collections import deque
from queue import Empty, Full

from torch.multiprocessing import Value

from .shm import _item_from_shm
from .types import End


class Chunk:
    """Wire marker: N already-serialized payloads bundled into ONE queue message.

    Amortizes the per-message queue cost (lock, pipe write, consumer wakeup) across
    N items; each payload inside is still an independent `_item_to_shm` output, so
    shm-refs and torch fd-sharing behave exactly as unchunked items. Sentinels
    (`End`, "worker_stop") are never placed inside a Chunk. A dedicated class (not
    a bare list) because worker payloads may themselves be lists."""

    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items

    def __getstate__(self):
        return self.items

    def __setstate__(self, state):
        self.items = state


class _OutputChannel:
    """Accumulates serialized payloads and flushes them as Chunks.

    Flush triggers: pending count reaches `chunk_size`, or the OLDEST pending item
    is older than `chunk_ms` (checked on every send and via `maybe_flush()` from
    the worker's idle loop — so a short batch always gets through). With
    `chunk_size=0` acts as a plain passthrough put (today's behavior), letting all
    output paths use the channel unconditionally.

    Single-threaded by design: each worker process/thread owns its own channel.
    The blocking put retries forever — matching the old inline put loops: an
    in-flight item must survive a graceful stop (should_stop=1) and deliver
    once the consumer frees queue space (see test_should_stop_does_not_drop_
    inflight_put). Force stop terminates worker processes outright."""

    def __init__(self, queue, chunk_size=0, chunk_ms=10.0):
        self.queue = queue
        self.chunk_size = chunk_size
        self.chunk_s = chunk_ms / 1000.0
        self._pending = []
        self._oldest = 0.0  # monotonic time of first pending item

    def _blocking_put(self, msg):
        while True:
            try:
                self.queue.put(msg, timeout=0.1)
                return
            except Full:
                time.sleep(0.01)

    def send(self, payload):
        """Queue one serialized payload (chunked or passthrough)."""
        if self.chunk_size <= 0:
            self._blocking_put(payload)
            return
        if not self._pending:
            self._oldest = time.monotonic()
        self._pending.append(payload)
        if (
            len(self._pending) >= self.chunk_size
            or time.monotonic() - self._oldest >= self.chunk_s
        ):
            self.flush()

    def maybe_flush(self):
        """Flush a partial chunk whose oldest item exceeded chunk_ms. Call from
        idle points in the worker loop (top of loop / Empty branch)."""
        if self._pending and time.monotonic() - self._oldest >= self.chunk_s:
            self.flush()

    def flush(self):
        """Emit pending payloads as one (possibly partial) Chunk. Must run before
        any End sentinel is put so a chunk can never be overtaken by End."""
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        self._blocking_put(Chunk(pending))


class _InputChannel:
    """Reads items from a queue that may carry Chunks, bare payloads, or sentinels.

    Chunks are unpacked into a local buffer and returned one payload at a time, so
    every consumer keeps its exact one-item-at-a-time semantics. Bare items (e.g.
    from push() back-edges) pass straight through, as do sentinels — `End` and
    "worker_stop" are never inside a Chunk.

    NOTE: buffered payloads are already OFF the shared queue. A worker that exits
    early (worker_stop) must drain `buffered()` first or those items are lost."""

    __slots__ = ("queue", "_buf")

    def __init__(self, queue):
        self.queue = queue
        self._buf = deque()

    def get(self, timeout=None):
        if self._buf:
            return self._buf.popleft()
        msg = self.queue.get(timeout=timeout)
        if type(msg) is Chunk:
            self._buf.extend(msg.items)
            return self._buf.popleft()
        return msg

    def get_nowait(self):
        if self._buf:
            return self._buf.popleft()
        msg = self.queue.get_nowait()
        if type(msg) is Chunk:
            self._buf.extend(msg.items)
            return self._buf.popleft()
        return msg

    def buffered(self):
        """Number of locally buffered (already-dequeued) payloads."""
        return len(self._buf)

    def requeue_buffered(self, should_stop=None):
        """Return locally buffered payloads to the shared queue (as one Chunk).

        MUST be called before a worker exits early (worker_stop): buffered
        payloads are already off the shared queue and would otherwise be
        silently lost. Retries forever like every put (see _OutputChannel);
        `should_stop` is accepted for call-site symmetry but a graceful stop
        must still deliver these items."""
        if not self._buf:
            return
        pending = list(self._buf)
        self._buf.clear()
        msg = Chunk(pending)
        while True:
            try:
                self.queue.put(msg, timeout=0.1)
                return
            except Full:
                time.sleep(0.01)

    def put(self, msg, **kwargs):
        """Pass-through so requeue paths (e.g. re-putting "worker_stop") work."""
        return self.queue.put(msg, **kwargs)


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
        self.queue = _InputChannel(queue)

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
