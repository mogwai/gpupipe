# gpupipe

Get the most out of your CPUs and GPUs without leaving Python: streaming pipelines of multi-worker stages — parallel downloads feeding GPU inference feeding DB writes — with backpressure, batching, and graceful shutdown handled for you.

## Installation

```bash
# With uv
uv add gpupipe

# Or pip
pip install gpupipe
```

The distribution is `gpupipe`; the module you import is `pipe` (sklearn-style):

```python
from pipe import Pipe
```

## Quick Start

```python
from pipe import Pipe

class Generator:
    def __call__(self):
        # Generator pattern - yield items, exhaustion signals completion
        for i in range(100):
            yield {"id": i}

class Worker:
    def load(self):
        # Called after process spawn - initialize heavy resources here
        pass

    def __call__(self, item):
        item["processed"] = True
        return item

pipe = Pipe()
pipe.add(Generator(), outqn=20)
pipe.add(Worker(), workers=4, outqn=20)

for result in pipe:
    print(result)
```

> **Autoscaling** (queue-pressure-based worker scaling) is a **planned feature** —
> designed and implemented, but not currently wired into `Pipe`. See
> [`PLANNED.md`](PLANNED.md) for the design and re-integration steps.

## Capability Tour

### GPU pools

```python
pipe.add(GPUWorker(), pergpu=True, outqn=None)      # one worker per GPU
pipe.add(GPUWorker(), gpus=[5, 6], workers=2)       # 2 workers on each of GPUs 5,6
pipe.add(GPUWorker(), gpu_id=0)                     # pin every worker to GPU 0
```

Each process worker is isolated to its GPU via `CUDA_VISIBLE_DEVICES`; requesting a
GPU that doesn't exist fails loudly instead of silently running on CPU.

### CPU affinity

```python
# 16 cores chunked across 4 workers (4 dedicated cores each), BLAS threads sized to match
pipe.add(MelSpectrogram(), workers=4, cpus=range(16))
pipe.add(HeavyCPU(), workers=2, cpu_threads=8)      # just lift the 2-thread BLAS cap
```

### Threaded workers (I/O-bound)

```python
pipe.add(IOWorker(), workers=8, thread=True, outqn=20)   # one process, 8 threads
```

### Streaming async IO (downloads)

```python
from pipe import AsyncPoolWorker

class Downloader(AsyncPoolWorker):
    async def process(self, item):                  # N in flight continuously,
        item["data"] = await fetch(item["url"])     # each result emitted the
        return item                                 # moment it lands - no batch barrier

pipe.add(Downloader(max_concurrent=256), workers=1, outqn=200)
```

### Framework batching

```python
class GPUInference:
    def __call__(self, batch):                      # batch = list of up to 16 items,
        return run_model(batch)                     # partial batches included

pipe.add(GPUInference(), pergpu=True, batch=16)
```

### Send items back for retry: `push()`

```python
class QualityGate:
    def __call__(self, item):
        if ok(item):
            return item
        if item.setdefault("tries", 0) < 2:
            item["tries"] += 1
            self.push("Renderer", item, timeout=5)  # back to the Renderer stage
            return None
        return item                                 # give up, pass through
```

### One pipeline feeding N DDP ranks

```python
pipe = Pipe(expected_consumers=world_size)          # rank 0 runs the pipe
pipe.start()
# every rank:
for batch in PipeIterator(shared_queue):            # items distributed across ranks,
    train_step(batch)                               # each rank gets its own End
```

### Live stats without log clobbering

```python
pipe = Pipe(stats_interval=3)                       # pinned one-line display on a TTY
pipe.print("checkpoint saved")                      # prints ABOVE the stats line
```

### Profiling

```python
pipe = Pipe(profile=True)   # per-worker cProfile + peak RSS, summary on stop
```

## Testing

```bash
# Run all tests
pytest tests/

# Run a specific test
pytest tests/test_basic.py -v
```

## Key Classes

- `Pipe` - Main pipeline orchestrator
- `End` - Sentinel a root worker returns to signal completion
- `AsyncPoolWorker` - Streaming async IO stage: subclass, implement `async def process(item)`
- `Batcher` - Batch items together
- `BufferAndShuffle` - Buffer and shuffle items
- `PipeIterator` - Read a pipe's output queue from another process (DDP shared mode)

## Features

- Multi-worker stages connected by queues, with backpressure
- Threaded workers (`thread=True`) for I/O-bound stages
- Streaming async IO stages (`AsyncPoolWorker`: N requests in flight, no batch barrier)
- Framework batching (`batch=N`: worker receives a list, partial batches included)
- GPU pools: `pergpu=True`, `gpu_id=N`, or explicit `gpus=[...]` with per-worker isolation
- CPU affinity: `cpus=[...]` chunked across workers, `cpu_threads=` for BLAS sizing
- Chunked queue transport (`chunk=`/`chunk_ms=`) for small-item edges
- `worker.push(stage, item)` - send an item back to an earlier stage (retry/quality gates)
- DDP shared mode: one pipeline feeds N training ranks (`expected_consumers` + `PipeIterator`)
- Live stats: pinned one-line display on a TTY (`pipe.print()`/`print_above()` write above it),
  rich mode, or poll `get_stats()` externally
- Health monitoring with worker restart; graceful shutdown with flush semantics
- Per-worker profiling (`profile=True`: cProfile + peak RSS summary)
- Sequential mode (`sequential=True`) for single-process debugging
- HTTP serving (`pipe.web`) for cross-machine pipelines

## Documentation

- [`PIPE_REFERENCE.md`](PIPE_REFERENCE.md) - the full reference: worker types, return-value
  semantics, completion signaling, tensor handling, DDP, env vars, pitfalls
- [`docs/examples/`](docs/examples/) - runnable examples and job templates
- [`PLANNED.md`](PLANNED.md) - designed-but-unwired features (autoscaling)
