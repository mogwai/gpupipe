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
4. On upstream completion: framework calls `flush()` if it exists
5. Worker exits, increments stage_end_counter
6. Last worker at a stage sets stage_done Event → signals downstream

**Important:** Middle workers never see the `End` sentinel - the framework handles it internally.

## Worker Types

### Root Worker (first stage, no input queue)

Two patterns for root workers:

**Return-based** - called repeatedly, return `End` when done:
```python
from pipe import End

class Source:
    def load(self):
        self.db = connect()
        self.items = self.db.query()
        self.idx = 0

    def __call__(self):
        if self.idx >= len(self.items):
            return End
        item = self.items[self.idx]
        self.idx += 1
        return item
```

**Generator-based** - yield items, exhaustion signals completion:
```python
class Source:
    def __call__(self):
        for row in query_db():
            yield row
```

### Processor (receives items from upstream)

Workers never see `End` - the framework handles it. Just process items:

```python
class Processor:
    def load(self):
        self.model = load_model()

    def __call__(self, item):
        return {"result": self.model(item)}
```

### Expander using yield (1 input → N outputs)

Processors can use generators to emit multiple items per input:

```python
class Splitter:
    def __call__(self, item):
        for chunk in split_into_chunks(item["data"]):
            yield {"chunk": chunk, "parent_id": item["id"]}
```

### Batcher (accumulate then flush)

Buffer items, emit when batch full. Implement `flush()` to return remaining items at shutdown:

```python
class Batcher:
    def load(self):
        self._buf = []

    def __call__(self, item):
        self._buf.append(item)
        if len(self._buf) >= 32:
            return self._flush()
        return None  # don't emit yet

    def _flush(self):
        out = self._buf
        self._buf = []
        return out  # list = multiple items downstream

    def flush(self):
        # called by framework at shutdown
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
| Generator (yield) | Each yielded item emitted separately downstream |
| `None` | Item consumed/filtered - nothing downstream |
| `End` | Root worker only: signals completion (or use generator exhaustion) |

The framework handles `End` internally - middle workers never see it. Generators are iterated until exhausted.

## Completion Signaling

Uses Event-based coordination (not sentinel passing):

1. Root worker returns `End` (or generator exhausts) → root worker exits → sets `stage_done_events[0]`
2. Stage 1 workers see `upstream_done` Event set + input queue empty for 1s → exit
3. Last worker at stage N sets `stage_done_events[N]` → signals stage N+1
4. Final stage done → consumer iteration ends

Workers are flushed on exit: framework calls `worker.flush()` to drain buffered items before signaling downstream.

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
    stats_interval=30,        # stats collection interval in seconds (0=off)
    stats_mode="rich",        # "rich" (default, standalone progress bar),
                              # "text" (ANSI one-liners to stdout),
                              # "external" (no display, poll with get_stats())
    health_check_interval=30, # check worker liveness every N seconds (0=off)
    expected_consumers=1,     # for DDP: multiply end signals for N consumers
    raise_errors=False,       # raise exceptions in workers instead of logging
)
```

### Stats Modes

| Mode | Display | Thread | Use case |
|------|---------|--------|----------|
| `"rich"` | Rich Progress bar (standalone Live) | Yes | Standalone scripts |
| `"text"` | ANSI one-liners to stdout | Yes | Non-interactive / logging |
| `"external"` | None (caller polls `get_stats()`) | No | Embedded in another UI (e.g. training loop) |

### Polling stats externally

When `stats_mode="external"`, no background display thread runs. The caller polls stats from the main thread:

```python
pipe = Pipe(stats_mode="external")
pipe.add(Source(), outqn=50)
pipe.add(Processor(), workers=4, outqn=50)

for item in pipe:
    stats = pipe.get_stats()  # list of dicts, one per stage
    for s in stats:
        print(f"Stage {s['stage_idx']}: {s['qsize']}/{s['qmax']} queued, "
              f"{s['items']} items, {s['active']}/{s['total_workers']} workers"
              + (f", {s['stage_rtf']:.0f}x RTF" if s['has_audio'] else ""))
    process(item)
```

`get_stats()` returns a list of dicts per stage:
- `stage_idx`, `done`, `qsize`, `qmax` — queue fill level
- `items`, `active`, `total_workers` — worker activity
- `stage_rtf`, `avg_worker_rtf`, `has_audio` — real-time factor (audio pipelines)

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

## Tensor Handling (Shared Memory)

Items containing torch tensors are automatically serialized to `/dev/shm` files. The queue carries only a path reference (~60 bytes), not the tensor data.

**Why not use torch.multiprocessing's built-in tensor sharing?**
PyTorch's default `file_descriptor` strategy passes tensors via a socket server in the producer process. If the producer dies (crash, autoscale-down), the socket is gone and consumers can't retrieve the data. Our approach writes named files to `/dev/shm` which persist regardless of process state.

**How it works:**
1. Producer: `_item_to_shm(item)` → writes tensors + pickled metadata to `/dev/shm/pipe_<pid>_<uuid>`
2. Queue carries: `{"__shm__": "/dev/shm/pipe_12345_abcdef"}` (tiny)
3. Consumer: `_item_from_shm(ref)` → mmaps file, reconstructs tensors, unlinks file

**File format:** `[4B header_len][JSON header][field bytes...]`
- Tensors: raw numpy bytes at offsets (bfloat16 stored as uint16)
- Non-tensor fields: pickled at offsets
- Header maps field names → type, dtype, shape, offset, size

**Performance (round-trip serialize + deserialize):**

| Item size | Overhead | vs typical processing |
|-----------|----------|----------------------|
| 64KB (1s audio @16kHz) | 0.1ms | <0.1% |
| 640KB (10s audio) | 0.4ms | ~0.4% |
| 1.9MB (30s audio) | 2.7ms | ~1% |
| 58MB (10min audio) | 118ms | significant |
| 346MB (1hr audio) | 715ms | bottleneck |

**Current cost breakdown:** each stage boundary copies ALL tensor fields, even if the worker only read metadata. A 5-stage pipeline with 10s audio = 5 × 0.4ms = 2ms total serialization overhead.

**Best practice:** drop tensor fields once no longer needed:
```python
def __call__(self, item):
    result = self.model(item["audio"])
    item["audio"] = None  # stop serializing audio downstream
    item["result"] = result
    return item
```

**Stale file cleanup:** `_cleanup_stale_shm()` runs on `Pipe()` init, removes any `/dev/shm/pipe_*` files from previous crashed runs.

### Future: ShmPool (not yet implemented)

Pre-allocated pool of shared memory slots with cached handles per worker process. Eliminates per-item file create/open/mmap/unlink syscalls.

**Benchmarks vs current file approach (with cached handles):**

| Item size | File (current) | Pool | Speedup |
|-----------|---------------|------|---------|
| 64KB | 0.10ms | 0.10ms | 1x |
| 640KB | 0.42ms | 0.11ms | 3.9x |
| 1.9MB | 2.7ms | 0.19ms | 14x |

**Pass-through optimization:** stages that don't access tensor fields can forward the slot reference without any copy. A 5-stage pipeline where only 1 stage reads audio: 4 stages × 0.001ms + 1 stage × 0.11ms = 0.1ms total (vs 2ms current).

**Tradeoffs:**
- Requires reserving memory upfront: pool_size = sum(all outqn) × slot_size
- Slot size must fit largest expected item (fallback to file-based for oversized)
- Pool grows on demand (new slots allocated if free queue empty)
- Slots released by final consumer, not intermediate stages

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
        if not valid(item): return None
        return item
```

### Expansion (1 → N items)
```python
class Splitter:
    def __call__(self, item):
        return [{"chunk": c} for c in split(item)]
```

### Progress tracking in final stage
```python
class Writer:
    def load(self):
        self.count = 0
        self.start = time.time()

    def __call__(self, item):
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
3. **Workers don't handle "end"** - framework handles End sentinel internally, workers never see it
4. **`flush()` method for batchers** - framework calls this at shutdown to emit remaining buffered items
5. **thread=True for IO-bound** - downloads, DB queries. Shares memory, no pickle needed, GIL-friendly for IO waits
6. **pergpu=True** - one worker per GPU, sets CUDA_VISIBLE_DEVICES per worker
7. **debug=True** - single process, sequential execution, for debugging/testing
8. **Queue full = backpressure** - workers block on put when downstream slow. This is intentional.
9. **outqn=None for GPU stages** - GPU has variable latency, unlimited queue prevents blocking fast stages
10. **Workers are independent** - no shared state between workers at same stage (use DB/S3 for coordination)

## Verbose output

Pipe is quiet by default. Set `PIPE_VERBOSE=1` to enable informational prints (startup messages, worker lifecycle, signal handling). Errors and warnings always print regardless. Stats display is controlled separately via `stats_mode`.

## Common Pitfalls

- **Deadlock**: Usually queue full + worker waiting. Fix: increase outqn or add more downstream workers
- **Pickle errors**: Worker has unpicklable attributes in `__init__`. Fix: move to `load()`
- **OOM**: Queue too large with big items (tensors). Fix: reduce outqn
- **Items lost**: Worker returns list containing None. Fix: filter None before returning
- **flush() not called**: Only called if worker has the method AND there's an output queue
- **Slow shutdown**: Workers blocked on queue.put(). Framework uses 0.1s timeouts to eventually exit

## Web Server (from pipe.web import ...)

Serve pipe output over HTTP. Useful for distributed pipelines where producer and consumer run on separate machines.

Install: `uv pip install -e ".[web]"` (adds fastapi, uvicorn, lz4)

### Quick Start

```python
from pipe import Pipe
from pipe.web import serve_pipe

pipe = Pipe()
pipe.add(DataLoader(), workers=2, outqn=50)
pipe.add(Processor(), workers=4, outqn=50)
serve_pipe(pipe, port=8000, compression="lz4")
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/next` | GET | Returns next item as torch-serialized bytes. Headers: `X-Compression`, `X-Items-Served`, `X-Size-Bytes` |
| `/health` | GET | JSON: status, pipe_alive, items_served, errors, uptime |
| `/stats` | GET | JSON: throughput (items/s, MB/s), queue_depth, total served |
| `/` | GET | API info |

### Pre-serialization (offload from HTTP handler)

Add `SerializerWorker` as final pipe stage to serialize + compress in parallel workers rather than in the single-threaded HTTP handler:

```python
from pipe.web import SerializerWorker, serve_pipe

pipe = Pipe()
pipe.add(DataLoader(), workers=2, outqn=50)
pipe.add(Processor(), workers=4, outqn=50)
pipe.add(SerializerWorker(compression="lz4"), workers=2, outqn=20)
serve_pipe(pipe, port=8000, pre_serialized=True)
```

### Client (consumer side)

```python
import httpx
import torch
import io
import lz4.frame

def fetch_item(url="http://localhost:8000/next"):
    r = httpx.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    if r.headers.get("X-Compression") == "lz4":
        data = lz4.frame.decompress(data)
    return torch.load(io.BytesIO(data), weights_only=False)
```

### PipeServer class (for more control)

```python
from pipe.web import PipeServer

server = PipeServer(
    pipe,
    compression="lz4",   # "none" or "lz4"
    timeout=30.0,         # seconds to wait for next item before 503
    pre_serialized=False, # True if using SerializerWorker
)
server.start(host="0.0.0.0", port=8000, workers=1)  # uvicorn kwargs
```

### Bandwidth considerations

| Scenario | Per-item size | Throughput needed |
|----------|--------------|-------------------|
| Small metadata dicts | ~1 KB | trivial |
| Audio tensors (10s) | ~640 KB | ~50 MB/s for 80/s |
| Mel batches (batch=8) | ~6.5 MB (lz4) | ~100 MB/s for 15/s |

Recommended for localhost or same-datacenter (10+ Gbps). For WAN, run pipeline locally.

### Pipe.stop()

Public method to gracefully stop the pipeline. Sends end signals, joins workers, cleans up queues and manager.

```python
pipe.stop()         # graceful (waits for workers)
pipe.stop(force=True)  # terminate immediately
```

## Helpers (from pipe import ...)

- `Batcher(size, collate_fn=None)` - simple batching, returns None until full then returns batch
- `BufferAndShuffle(size)` - ring buffer with random shuffle on overflow
- `RetrieveSQL(conn_str, query, batch_size)` - paginated DB iteration with randomization
- `SQLConnection(conn_str)` - reusable postgres connection for workers
- `RTF()` - real-time factor tracking (audio_duration / process_duration)
