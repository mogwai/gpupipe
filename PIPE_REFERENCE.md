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

### Framework Batching (batch=N)

Use `batch=N` on `pipe.add()` to have the framework collect items into batches. Worker receives a list:

```python
class GPUInference:
    def load(self):
        self.model = load_model("cuda")

    def __call__(self, batch):
        # batch is a list of up to 16 items
        tensors = torch.stack([item["audio"] for item in batch])
        results = self.model(tensors)
        return [{"result": r, **item} for r, item in zip(results, batch)]

pipe.add(GPUInference(), pergpu=True, batch=16, outqn=50)
```

The framework collects up to N items greedily (drains queue without blocking), then calls `__call__` with whatever's available. Partial batches are normal — no need to wait for a full batch.

The edge feeding a `batch=N` stage is automatically **transport-chunked** at N (see "Chunked transport"): the upstream worker ships N items per queue message, so the collector assembles a full batch from one queue op instead of N.

At shutdown, the framework calls `flush()` if it exists to emit remaining buffered items.

### Async IO Worker (batched async downloads)

Use `batch=N` to let the framework collect items, then download them all concurrently:

```python
class AsyncDownloader:
    def __init__(self, max_concurrent: int = 128):
        self._max_concurrent = max_concurrent

    def load(self):
        self._loop = asyncio.new_event_loop()
        self._sem = asyncio.Semaphore(self._max_concurrent)

    async def _fetch_batch(self, items):
        async def fetch_one(item):
            async with self._sem:
                return await download(item["url"])
        return await asyncio.gather(*[fetch_one(i) for i in items])

    def __call__(self, batch):
        results = self._loop.run_until_complete(self._fetch_batch(batch))
        return [r for r in results if r is not None]

pipe.add(AsyncDownloader(), workers=4, thread=True, batch=64, outqn=200)
```

No manual buffer, no `flush()`, no `None` returns. The framework drains up to 64 items from the queue and passes them as a list. Partial batches just work.

### Worker with run() + pull/put

For workers that need dynamic control over how many items to pull each iteration (e.g. filling variable GPU slots), define a `run()` method. The framework injects `self.pull(n)` and `self.put(item)`:

```python
class StreamingTranscriber:
    def load(self):
        self.model = load_model("cuda")
        self.max_batch = 8

    def run(self):
        while True:
            items = self.pull(self.max_batch)
            if not items:
                break
            results = self.model.transcribe([i["audio"] for i in items])
            for item, result in zip(items, results):
                item["text"] = result
                self.put(item)

pipe.add(StreamingTranscriber(), pergpu=True, batch=8, outqn=50)
```

- `self.pull(n)` returns up to n items (non-blocking, returns what's available)
- `self.put(item)` sends to the output queue (blocks if full, respects backpressure)
- `pull()` returns `[]` when upstream is done and queue is empty — use this to break
- Works in both multiprocessing and sequential mode
- Triggered automatically when a non-root worker defines `run()`

### Sending an item BACK to an earlier stage: `self.push(stage, item)`

Every non-root worker (plain `__call__` or `run()`) gets `self.push(stage, item)`,
which drops `item` onto the INPUT queue of an EARLIER stage so it gets reprocessed.
The canonical use is a downstream quality gate that re-runs an expensive earlier
stage on failure (e.g. re-render audio whose WER check failed):

```python
class WERCheck:
    MAX_RETRIES = 2

    def __call__(self, item):
        if self.wer(item) <= 0.15:
            return item                      # pass -> forward to next stage
        if item.get("retries", 0) < self.MAX_RETRIES:
            item["retries"] = item.get("retries", 0) + 1
            self.push("Renderer", item)      # back to the Renderer stage
            return None                      # nothing forwarded this time
        return item                          # give up: forward the best we have

pipe.add(Renderer(), pergpu=True)            # stage 1
pipe.add(WERCheck(), pergpu=True)            # stage 2 -> pushes back to stage 1
```

- `stage` is an int stage index (0 = root), a stage **name** (str), or the worker
  **class** / a worker **instance** (resolved by its `__name__`). Names/classes
  resolve to the FIRST stage with that name. `push` targets that stage's input
  queue. Pushing to stage 0 is invalid (root has no input). So all of these are
  equivalent: `self.push(1, item)`, `self.push("Renderer", item)`,
  `self.push(Renderer, item)`.
- **Always bound retries on the item** — pipe's completion is signalled
  forward-only, so the retry cycle must terminate on its own.
- **Best-effort at end-of-run:** a stage finishes once its upstream is done and
  its input queue drains. An item pushed back *after* the target stage has
  already drained (only possible once the whole upstream is exhausted) may be
  dropped. Mid-run (the normal case, upstream still producing) every push is
  honoured. For bulk data-gen this is exactly the right trade-off.
- Works in both multiprocessing and sequential mode.

**Avoiding cyclic deadlock (size the back-edge queue).** `push` adds a cycle to
an otherwise linear pipeline, and a cycle of *bounded* queues can deadlock: if
the items simultaneously circulating in the cycle outnumber the cycle's total
queue capacity, the forward stage blocks putting (its output full) so it stops
draining the back-edge queue, so `push` blocks (back-edge full) — nobody moves.
The rule:

> the combined capacity of the queues in the cycle (the back-edge's landing
> queue + the forward queues between the target stage and the pushing stage)
> must exceed the peak number of items in the cycle at once.

For the intended use (a quality gate that re-runs only the small fraction of
items that fail) the cycle holds very few items, so default queue sizes are
safe. If you expect a high retry fraction — or want a hard guarantee — make the
target stage's input queue (i.e. the `outqn` of the stage *before* the push
target) large, or unbounded (`outqn=0`). Always cap retries regardless.

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
    pergpu=True,        # spawn one worker per available GPU (== gpus=range(N))
    gpus=[5, 6],        # pin this stage to a SPECIFIC GPU pool (round-robin per worker);
                        # `workers` is a per-GPU multiplier (workers=2 -> 2 per listed GPU)
    gpu_id=0,           # pin all workers to a single GPU (== gpus=[0])
    cpus=[0, 1, 2, 3],  # pin this stage to a CPU core pool, CHUNKED across its workers
                        # (each worker gets a contiguous slice + threads sized to it).
                        # Orthogonal to gpus= (a GPU feeder can set both). Confines workers
                        # to those cores (locality); disjoint pools partition the box across
                        # YOUR stages — it does NOT reserve cores from other processes (use a
                        # cgroup cpuset for that). Disables autoscale.
    cpu_threads=8,      # per-worker BLAS/torch thread count (torch/OMP/MKL/OpenBLAS).
                        # Lifts the default flat 2-thread cap for a CPU-heavy stage (mel)
                        # WITHOUT pinning. With cpus= too, this wins over the slice size.
    batch=16,           # framework collects up to N items, __call__ receives a list
    chunk=None,         # OUTPUT-edge transport chunking: bundle N serialized items into
                        # ONE queue message. None (default) = auto: adopts the DOWNSTREAM
                        # stage's batch size, so edges feeding batch=B stages chunk at B.
                        # 0 = force off. Explicit N = chunk at N (e.g. set on the final
                        # stage to speed up consumer iteration). Big win on high-rate
                        # small-item edges; see "Chunked transport" below.
    chunk_ms=10.0,      # max age of a partial chunk before it's flushed anyway, so a
                        # short batch / slow trickle never waits for chunk-mates
    autoscale=True,     # enable autoscaling for this stage (disabled for GPU/CPU-pinned stages)
    min_workers=1,      # autoscale floor (won't scale below)
    max_workers=8,      # autoscale ceiling (won't scale above)
    drain=True,         # on Ctrl+C: True=finish in-flight items, False=stop immediately
                        # (drain=False stages must form a contiguous prefix from stage 0)
)
```

### Queue sizing guidance
- `outqn=0`: Final stage only (output goes directly to consumer iterator)
- `outqn=None`: Unlimited (use for GPU stages with variable latency)
- Small (10-50): Tight backpressure, low memory, good for large items (audio tensors)
- Large (200-1024): Smooth throughput, higher memory, good for small items (metadata dicts)

## Chunked transport (chunk= / chunk_ms=)

Every `put`/`get` on an mp.Queue costs a lock acquisition, a pipe write, and a
consumer wakeup — and under multi-worker contention the single queue lock
serializes. `chunk=N` bundles N **already-serialized** items into ONE queue
message (a `Chunk`), amortizing that cost by N. Items inside a chunk are still
independent `_item_to_shm` payloads, so **torch fd-sharing / shm-refs are
unchanged** — only the message count drops.

**Auto mode (default):** an edge feeding a `batch=B` stage chunks at B, so one
queue message = one downstream batch, and the batch collector fills from a
single `get` instead of B. Edges not feeding a batch stage stay unchunked
(preserves fine-grained fan-out). Force with explicit `chunk=N`, disable with
`chunk=0`.

**Flush rule (why short batches still get through):** a chunk is emitted when
it reaches `chunk` items OR its oldest item is `chunk_ms` (default 10ms) old OR
the worker exits (End sentinels always come after the final flush). Under high
load chunks fill instantly (max throughput); under low load the timeout fires
with small chunks (low latency, fine-grained fan-out). Latency bound for a
partial chunk is `max(chunk_ms, current worker call duration)`.

**Semantics preserved:**
- `outqn` stays in ITEMS — the underlying queue maxsize is scaled down by the
  chunk factor, so backpressure triggers at the same item count.
- Stats display scales `qsize`/`qmax` back to items; autoscaler thresholds use
  fill ratios, which are unit-invariant.
- `push()` back-edge items are bare messages and coexist with chunks on the
  same queue.
- A worker holding chunk-buffered input when told to scale down (`worker_stop`)
  re-queues that buffer for its siblings — nothing is dropped.
- A chunk is grabbed whole by ONE downstream worker. That's the deliberate
  trade: on a chunked edge, work is handed out N items at a time. Auto mode
  aligns this with `batch=` (which wants N at a time anyway); don't chunk edges
  where single heavy items need to spread across workers.

**When it helps:** high-rate small/metadata items (DB rows, shm refs, dicts) —
benchmark: ~1.7x end-to-end on a no-op batch pipeline (scripts/bench_chunking.py),
more the more queue-bound the edge. **When it doesn't:** big-tensor edges — the
fd-sharing default already moves payload out-of-band, so message count is not
the bottleneck; chunking a tensor edge also pins N tensors' handles per message.

## Pipe() Options

```python
pipe = Pipe(
    sequential=False,           # sequential single-process mode (no multiprocessing)
    debug=False,                # wrap queues with InstrumentedQueue, track per-queue transit latency
    autoscale=False,            # global autoscale enable (disabled by default)
    max_workers_per_stage=8,    # global autoscale cap
    stats_interval=0.2,         # stats collection interval in seconds (0=off)
    stats_mode="text",          # "text" (ANSI one-liners to stdout, default),
                                # "rich" (Rich Progress bar, standalone Live),
                                # "external" (no display, poll with get_stats())
    health_check_interval=30,   # check worker liveness every N seconds (0=off)
    expected_consumers=1,       # for DDP: multiply end signals for N consumers
    raise_errors=None,          # if None, defaults to sequential mode; True raises exceptions
    allow_full_restart=True,    # allow restarting entire pipeline on repeated crashes
    use_shm=False,              # use shared memory for tensor serialization
    output_shm=False,           # output items use shared memory encoding
)
```

### Stats Modes

| Mode | Display | Thread | Use case |
|------|---------|--------|----------|
| `"text"` | ANSI one-liners to stdout (default) | Yes | Standalone scripts, logging |
| `"rich"` | Rich Progress bar (standalone Live) | Yes | Standalone scripts with nicer display |
| `"external"` | None (caller polls `get_stats()`) | No | Embedded in another UI (e.g. training loop), don't start stats thread |

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

## Tensor Handling

**Default (`use_shm=False`): torch's `file_descriptor` sharing.** Items containing torch tensors are passed via PyTorch's built-in `file_descriptor` strategy — the producer holds the tensor's shared-memory fd open and the queue carries a lightweight handle. This is the fast path and the recommended default. Workers deliberately stay alive until the whole pipeline stops (see the keep-alive loop in `_worker_run`), so a producing process never exits while its tensors are still in flight — which removes the "producer dies, fd is gone" failure that torch's fd strategy is sometimes criticised for. In practice this is faster and simpler than the shm path below.

**Opt-in (`use_shm=True`): `/dev/shm` files.** Tensors are serialized to named `/dev/shm/pipe_<pid>_<uuid>` files and the queue carries only a path reference (~60 bytes). Only reach for this when a producer may genuinely die mid-flight (e.g. crash-prone workers) and you need already-queued tensors to outlive the producer — the files persist regardless of process state. Tradeoffs: per-item file create/mmap/unlink syscalls (slower than fd sharing for small items), and **CPU tensors only** (move tensors to CPU before emitting). Scope it to specific output stages with `output_shm=True`.

**How `use_shm=True` works:**
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
from pipe import Pipe, End

class BatchLoader:
    def load(self):
        with connection() as cur:
            cur.execute("SELECT id, s3_key, duration FROM items WHERE processed = false")
            self.items = cur.fetchall()
        self.idx = 0

    def __call__(self):
        if self.idx >= len(self.items):
            return End
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
3. **Workers never see End** - framework handles End sentinel internally, workers never receive it
4. **`flush()` method for batchers** - framework calls this at shutdown to emit remaining buffered items
5. **thread=True for IO-bound** - downloads, DB queries. Shares memory, no pickle needed, GIL-friendly for IO waits
6. **pergpu=True / gpus=[...]** - GPU pinning. `pergpu` runs one worker on *every*
   GPU; `gpus=[5,6]` pins a stage to a *specific* pool (round-robin per worker, with
   `workers` as a per-GPU multiplier). Each worker's process is `set_device`-pinned —
   no need to hand-roll lock-file GPU pools. Different stages can target disjoint
   pools (e.g. `render` on `gpus=[0,1,2,3]`, `wer` on `gpus=[4]`, `tts` on `gpus=[5,6]`).
6b. **cpus=[...]** - CPU affinity, the core analog of `gpus=`. The pool is *chunked*
   across the stage's workers (worker 0 → first slice, etc.) and each worker's BLAS/torch
   thread count is sized to its slice, so an 8-worker mel stage on `cpus=range(32)` gives
   every worker 4 dedicated cores instead of 8 workers thrashing 32. Use it to (a) keep a
   CPU-heavy stage (mel, decode) from migrating/oversubscribing, and (b) *reserve* cores
   for GPU feeders — pin the feeder to its cores and pin other CPU stages to disjoint
   pools so nothing else lands there. Orthogonal to `gpus=` (a GPU feeder may set both).
   Linux-only (`os.sched_setaffinity`); more workers than cores → round-robin oversubscribe.
   Disables autoscale (a static pin can't track a live worker count).
   *Caveat:* affinity only **confines your worker** to those cores — it does **not** keep
   other processes off them. Reserving cores across your own pipeline means pinning every
   CPU-heavy stage to disjoint pools; walling off *unrelated* jobs needs a cgroup cpuset
   (`systemd AllowedCPUs=`, `docker --cpuset-cpus`, `isolcpus=`), which composes on top.
6c. **cpu_threads=N** - per-worker BLAS/torch thread count, independent of pinning. Every
   worker defaults to a flat 2 threads (this is the main guard against N torch workers each
   fanning BLAS across the whole machine); `cpu_threads=` raises that for a CPU-heavy stage
   (e.g. `mel` wanting 8) without touching affinity or autoscale. Precedence when combined
   with `cpus=`: explicit `cpu_threads` > `cpus=` slice size > the default 2.
7. **sequential=True** - single process, sequential execution, for debugging/testing
8. **debug=True** - wraps queues with InstrumentedQueue, shows per-queue transit latency in stats
9. **Queue full = backpressure** - workers block on put when downstream slow. This is intentional.
10. **outqn=None for GPU stages** - GPU has variable latency, unlimited queue prevents blocking fast stages
11. **Workers are independent** - no shared state between workers at same stage (use DB/S3 for coordination)

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
