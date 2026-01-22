# Pipe Reference (for LLM context)

Multiprocessing streaming pipeline framework. Stages connected by queues, each with N workers processing items in parallel. Handles backpressure, autoscaling, GPU distribution, graceful shutdown, tensor serialization, and worker crash recovery.

Use for IO-heavy scripts that benefit from parallel stages: downloads, GPU inference, DB writes, audio processing.

## Architecture

```
Consumer (for item in pipe)
    ↑ reads from
[Output Queue N]
    ↑ feeds
Stage N workers (processes/threads)
    ↑ reads from
[Queue N-1]
    ↑
Stage 2 workers
    ↑
[Queue 1]
    ↑
Stage 1 (root generator, no input queue)

Background threads: Health Monitor, Stats Monitor, Autoscaler
```

Each stage pulls from its input queue, processes, pushes to output queue. Multiple workers per stage = data parallelism. Backpressure: when output queue full, workers block on put.

## Core Pattern

```python
from pipe import Pipe

pipe = Pipe()
pipe.add(Source(), outqn=50)
pipe.add(Processor(), workers=4, outqn=50)
pipe.add(Writer(), workers=1, outqn=0)

for result in pipe:
    pass  # results from final stage
```

## Worker Lifecycle

1. Process/thread spawned
2. `load()` called (if exists) - heavy init here
3. Worker called repeatedly with items from input queue
4. On upstream completion: worker called with `"end"` to trigger flush
5. `flush()` method called (if exists) for any remaining buffered items
6. Worker exits, increments stage_end_counter
7. Last worker at a stage sets stage_done Event → signals downstream

## Worker Types

### Generator (first stage, no input queue)

Called repeatedly with no arguments. Must eventually return "end".

```python
class Source:
    def load(self):
        self.db = connect()
        self.items = self.db.query()
        self.idx = 0

    def __call__(self):
        if self.idx >= len(self.items):
            return "end"
        item = self.items[self.idx]
        self.idx += 1
        return item
```

Or using yield (called once, framework iterates):
```python
class Source:
    def __call__(self):
        for row in query_db():
            yield row
        return "end"
```

### Processor (receives items from upstream)

```python
class Processor:
    def load(self):
        self.model = load_model()

    def __call__(self, item):
        if item == "end": return item
        return {"result": self.model(item)}
```

### Batcher (accumulate then flush)

Buffer items, emit when batch full. Framework calls both `worker("end")` AND `flush()` on shutdown.

```python
class Batcher:
    def load(self):
        self._buf = []

    def __call__(self, item):
        if item == "end":
            return self._flush()
        self._buf.append(item)
        if len(self._buf) >= 32:
            return self._flush()
        return None  # don't emit yet

    def _flush(self):
        out = self._buf
        self._buf = []
        return out  # list = multiple items downstream

    def flush(self):
        # called by framework after worker("end") - safety net
        if self._buf:
            return self._flush()
```

### Async IO Worker (batched async downloads)

```python
class AsyncDownloader:
    def __init__(self, buffer_size: int = 32, max_concurrent: int = 128):
        self._buffer_size = buffer_size
        self._max_concurrent = max_concurrent

    def load(self):
        self._loop = asyncio.new_event_loop()
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._buffer = []

    async def _fetch_batch(self, items):
        async def fetch_one(item):
            async with self._semaphore:
                return await download(item["url"])
        return await asyncio.gather(*[fetch_one(i) for i in items])

    def _flush_buffer(self):
        if not self._buffer:
            return []
        items = self._buffer
        self._buffer = []
        results = self._loop.run_until_complete(self._fetch_batch(items))
        return [r for r in results if r is not None]

    def __call__(self, item):
        if item == "end":
            return self.flush() or None
        self._buffer.append(item)
        if len(self._buffer) >= self._buffer_size:
            return self._flush_buffer()
        return None

    def flush(self):
        return self._flush_buffer()
```

## Return Value Semantics

| Return | Effect |
|--------|--------|
| Single item (dict, etc.) | Passed downstream as one item |
| List of items | Each element emitted separately downstream |
| `None` | Item consumed/filtered - nothing downstream |
| `"end"` | Only meaningful from root generator (signals done) |

The framework wraps non-list returns in a list, filters out None and "end" from lists before putting to output queue.

## Completion Signaling

Uses Event-based coordination (not sentinel passing):

1. Root generator returns "end" → root worker exits → sets `stage_done_events[0]`
2. Stage 1 workers see `upstream_done` Event set + input queue empty for 1s → exit
3. Last worker at stage N sets `stage_done_events[N]` → signals stage N+1
4. Final stage done → consumer iteration ends

Workers are flushed on exit: framework calls `worker("end")` then `worker.flush()` to drain buffered items before signaling downstream.

## pipe.add() Options

```python
pipe.add(worker,
    workers=4,          # number of worker processes (or threads if thread=True)
    outqn=50,           # output queue max size (0=unbuffered final stage, None=unlimited)
    thread=True,        # use threads instead of processes (IO-bound, no pickle needed)
    pergpu=True,        # spawn one worker per available GPU (sets CUDA_VISIBLE_DEVICES)
    gpu_id=0,           # pin all workers to specific GPU
    autoscale=True,     # enable autoscaling for this stage
    min_workers=1,      # autoscale floor (won't scale below)
    max_workers=8,      # autoscale ceiling (won't scale above)
)
```

### Queue sizing guidance
- `outqn=0`: Final stage only (output goes directly to consumer iterator)
- `outqn=None`: Unlimited (use for GPU stages with variable latency)
- Small (10-50): Tight backpressure, low memory, good for large items (audio tensors)
- Large (200-1024): Smooth throughput, higher memory, good for small items (metadata dicts)

## Pipe() Options

```python
pipe = Pipe(
    debug=False,              # sequential single-process mode (no multiprocessing)
    autoscale=True,           # global autoscale enable
    max_workers_per_stage=8,  # global autoscale cap
    stats_interval=30,        # print queue sizes + timing every N seconds (0=off)
    health_check_interval=30, # check worker liveness every N seconds (0=off)
    share_tensors=False,      # auto-serialize PyTorch tensors through queues
    expected_consumers=1,     # for DDP: multiply end signals for N consumers
    raise_errors=False,       # raise exceptions in workers instead of logging
)
```

## Autoscaling Details

Autoscaler runs as background thread, checks every 1s:

- **Scale UP**: input queue fill >= 80% for 3 consecutive samples
  - Won't scale if CPU > 85% (system saturated)
  - Won't scale if output queue >= 90% full (downstream bottleneck)
  - Spawns new worker process with same config
- **Scale DOWN**: input queue fill <= 20% for 5 consecutive samples
  - Sends "worker_stop" sentinel to input queue
  - Worker exits gracefully after current item
  - Won't scale below min_workers
- **Cooldown**: 3s between scaling actions per stage
- **GPU stages**: Never autoscale (limited by GPU count)

## Health Monitoring

Background thread checks `process.is_alive()` every `health_check_interval`:
- Detects crashed workers (non-zero exitcode)
- Restarts individual workers with same config
- Repeated crashes (3+ in short window) → full pipeline restart via `pipe.restart()`

## Tensor Handling

When `share_tensors=True` or tensors detected in items:
- Tensors serialized to bytes via `torch.save()` before queue.put()
- Deserialized via `torch.load()` after queue.get()
- Works in nested structures (dict, list, tuple)
- Avoids PyTorch's FD-sharing mechanism (breaks when sender exits first)
- Workers auto-increase file descriptor limits

## Common Patterns

### Threaded IO + GPU processing
```python
pipe = Pipe(stats_interval=3)
pipe.add(DBReader(), outqn=200)
pipe.add(S3Downloader(), workers=16, thread=True, outqn=200)
pipe.add(GPUModel(), pergpu=True, outqn=50)
pipe.add(ResultWriter(), workers=1, outqn=0)
```

### Filtering (return None to drop)
```python
class Filter:
    def __call__(self, item):
        if item == "end": return item
        if not valid(item): return None
        return item
```

### Expansion (1 → N items)
```python
class Splitter:
    def __call__(self, item):
        if item == "end": return item
        return [{"chunk": c} for c in split(item)]
```

### Progress tracking in final stage
```python
class Writer:
    def load(self):
        self.count = 0
        self.start = time.time()

    def __call__(self, item):
        if item == "end": return self.flush() or None
        self.count += 1
        if self.count % 100 == 0:
            rate = self.count / (time.time() - self.start)
            print(f"  [{self.count}] {rate:.1f}/s")
        return item
```

## Complete Real-World Example

Audio alignment pipeline: DB → S3 download → GPU alignment → S3 upload + DB write

```python
from pipe import Pipe

class BatchLoader:
    def load(self):
        with connection() as cur:
            cur.execute("SELECT id, s3_key, duration FROM items WHERE processed = false")
            self.items = cur.fetchall()
        self.idx = 0

    def __call__(self):
        if self.idx >= len(self.items):
            return "end"
        item = self.items[self.idx]
        self.idx += 1
        return item

class AudioDownloader:
    def __init__(self, buffer_size=32, max_concurrent=128):
        self._buffer_size = buffer_size
        self._max_concurrent = max_concurrent

    def load(self):
        from obstore.store import S3Store
        self._store = S3Store(bucket="my-bucket", ...)
        self._loop = asyncio.new_event_loop()
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._buffer = []

    def __call__(self, item):
        if item == "end":
            return self.flush() or None
        self._buffer.append(item)
        if len(self._buffer) >= self._buffer_size:
            return self._flush_buffer()
        return None

    def _flush_buffer(self):
        items = self._buffer
        self._buffer = []
        results = self._loop.run_until_complete(self._download_batch(items))
        return [r for r in results if r is not None]

    def flush(self):
        return self._flush_buffer()

class GPUProcessor:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size

    def load(self):
        self.model = load_model("cuda")
        self._buffer = []

    def __call__(self, item):
        if item == "end":
            return self._flush_buffer()
        self._buffer.append(item)
        if len(self._buffer) >= self.batch_size:
            return self._flush_buffer()
        return None

    def _flush_buffer(self):
        if not self._buffer:
            return []
        batch = self._buffer
        self._buffer = []
        return self.model.process_batch(batch)

    def flush(self):
        return self._flush_buffer()

class ResultWriter:
    def load(self):
        self._batch = []

    def __call__(self, item):
        if item == "end":
            return self.flush() or None
        self._batch.append(item)
        if len(self._batch) >= 64:
            return self._flush()
        return None

    def _flush(self):
        batch = self._batch
        self._batch = []
        # batch DB update + S3 upload
        with connection() as cur:
            cur.executemany("UPDATE items SET processed=true WHERE id=%s",
                           [(i["id"],) for i in batch])
        return batch

    def flush(self):
        return self._flush()

# Pipeline assembly
pipe = Pipe(stats_interval=3, health_check_interval=120)
pipe.add(BatchLoader(), workers=1, outqn=1024)
pipe.add(AudioDownloader(buffer_size=256, max_concurrent=256), workers=4, outqn=1024)
pipe.add(GPUProcessor(batch_size=16), pergpu=True, outqn=None)
pipe.add(ResultWriter(), workers=1, outqn=0)

for result in pipe:
    pass
```

## Key Rules

1. **`load()` for heavy init** - models, DB connections, S3 clients, event loops go here (runs after fork in child process)
2. **`__init__()` must be picklable** - no lambdas, CUDA tensors, open files, boto3 clients
3. **Always handle "end"** - flush buffers and return remaining items when receiving "end"
4. **`flush()` method** - framework calls this as safety net after worker("end"), should return list or iterable
5. **thread=True for IO-bound** - downloads, DB queries. Shares memory, no pickle needed, GIL-friendly for IO waits
6. **pergpu=True** - one worker per GPU, sets CUDA_VISIBLE_DEVICES per worker
7. **debug=True** - single process, sequential execution, for debugging/testing
8. **Queue full = backpressure** - workers block on put when downstream slow. This is intentional.
9. **outqn=None for GPU stages** - GPU has variable latency, unlimited queue prevents blocking fast stages
10. **Workers are independent** - no shared state between workers at same stage (use DB/S3 for coordination)

## Common Pitfalls

- **Deadlock**: Usually queue full + worker waiting. Fix: increase outqn or add more downstream workers
- **Pickle errors**: Worker has unpicklable attributes in `__init__`. Fix: move to `load()`
- **OOM**: Queue too large with big items (tensors). Fix: reduce outqn
- **Items lost**: Worker returns list containing None. Fix: filter None before returning
- **flush() not called**: Only called if worker has the method AND there's an output queue
- **Slow shutdown**: Workers blocked on queue.put(). Framework uses 0.1s timeouts to eventually exit

## Helpers (from pipe import ...)

- `Batcher(size, collate_fn=None)` - simple batching, returns None until full then returns batch
- `BufferAndShuffle(size)` - ring buffer with random shuffle on overflow
- `RetrieveSQL(conn_str, query, batch_size)` - paginated DB iteration with randomization
- `SQLConnection(conn_str)` - reusable postgres connection for workers
- `RTF()` - real-time factor tracking (audio_duration / process_duration)
