# Pipe - Multiprocessing Pipeline Framework

A robust multiprocessing pipeline framework for streaming data processing with autoscaling, multi-worker stages, threaded workers, and graceful shutdown.

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

## Autoscaling

Enable automatic worker scaling based on queue pressure:

```python
# Global autoscale - all stages scale automatically
pipe = Pipe(autoscale=True, max_workers_per_stage=8)
pipe.add(Generator(), outqn=20)
pipe.add(Worker(), workers=1, outqn=20)  # Will scale 1-8 based on load
pipe.add(Worker(), workers=1, outqn=20, max_workers=4)  # Custom max

# Per-stage autoscale
pipe = Pipe()
pipe.add(Generator(), outqn=20)
pipe.add(Worker(), workers=1, outqn=20, autoscale=True, min_workers=1, max_workers=6)
```

Autoscaling features:
- **Queue pressure based**: Scales up when input queue is full, down when empty
- **CPU aware**: Won't scale up if CPU usage exceeds 85%
- **GPU stages disabled**: GPU workers (`pergpu=True` or `gpu_id`) never autoscale
- **Cooldown**: 3 second minimum between scaling actions

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

# Run specific tests
pytest tests/test_pipe.py::test_autoscale_global -v
```

## Key Classes

- `Pipe` - Main pipeline orchestrator
- `Batcher` - Batch items together
- `BufferAndShuffle` - Buffer and shuffle items
- `PipeIterator` - Iterate over pipeline output

## Features

- Multi-worker stages with automatic load balancing
- Threaded workers for I/O-bound tasks
- GPU pinning with `pergpu` or `gpu_id`
- Autoscaling with CPU awareness
- Health monitoring and worker restart
- Graceful shutdown with no data loss
- Sequential mode for single-process execution
- Debug mode for queue transit latency stats
