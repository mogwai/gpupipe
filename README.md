# Pipe - Multiprocessing Pipeline Framework

A robust multiprocessing pipeline framework for streaming data processing with multi-worker stages, threaded workers, and graceful shutdown.

## Installation

```bash
# From git
pip install git+https://github.com/USERNAME/pipe.git

# Or with uv
uv add git+https://github.com/USERNAME/pipe.git
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

## GPU Workers

```python
# One worker per GPU
pipe.add(GPUWorker(), workers=1, pergpu=True, outqn=20)

# Pin to specific GPU
pipe.add(GPUWorker(), workers=1, gpu_id=0, outqn=20)
```

## Threaded Workers

For I/O-bound work:

```python
pipe.add(IOWorker(), workers=8, thread=True, outqn=20)
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
- `Batcher` - Batch items together
- `BufferAndShuffle` - Buffer and shuffle items
- `PipeIterator` - Read a pipe's output queue from another process (DDP shared mode)

## Features

- Multi-worker stages with automatic load balancing
- Threaded workers for I/O-bound tasks
- GPU pinning with `pergpu` or `gpu_id`
- Autoscaling with CPU awareness
- Health monitoring and worker restart
- Graceful shutdown with no data loss
- Sequential mode for single-process execution
- Debug mode for queue transit latency stats
