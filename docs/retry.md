# Item-level retry via re-queue

## Problem

When a worker throws an exception, the item is silently dropped. All upstream processing is lost.

## Design: Re-queue failed items

Failed items go back on the **input** queue. Any worker picks them up later. Works with stateful workers, batchers, anything — the worker never sees retry metadata.

```python
pipe.add(S3Downloader(), workers=4, retries=3, outqn=200)
```

### How it works

1. Worker gets item from input queue (wrapper stripped if retry)
2. Calls `worker(item)`
3. On success: put result on output queue (normal path)
4. On failure:
   - Sleep with backoff + jitter
   - Wrap item: `{"__pipe_retry": count, "__pipe_item": item}`
   - Put back on input queue
   - Worker moves on to next item immediately

### Error classification

```python
TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError, BrokenPipeError)
```

- **Transient errors**: Re-queue forever. Backoff: `min(2**count, 30) * (0.5 + random.random())`
- **Other errors**: Re-queue up to `retries` times. After that, drop + log.
- **Skip errors** (optional): `skip_errors=(CudaOOMError,)` — drop immediately, no retry.

Users can extend transient list:
```python
pipe.add(Worker(), retries=3, transient_errors=(ConnectionError, MyRetryableError))
```

### Worker loop changes (in _worker_run)

```python
# Unwrap retry metadata
if isinstance(raw_item, dict) and "__pipe_retry" in raw_item:
    item = raw_item["__pipe_item"]
    retry_count = raw_item["__pipe_retry"]
else:
    item = raw_item
    retry_count = 0

try:
    result = worker(item)
except skip_errors:
    print(f"Worker {name} skipping item (non-retryable): {e}")
    continue
except transient_errors as e:
    delay = min(2 ** retry_count, 30) * (0.5 + random.random())
    print(f"Worker {name} transient error, re-queue (attempt {retry_count+1}), backoff {delay:.1f}s: {e}")
    time.sleep(delay)
    wrapped = {"__pipe_retry": retry_count + 1, "__pipe_item": item}
    in_queue.put(wrapped, timeout=1.0)
    continue
except Exception as e:
    if retry_count < retries:
        delay = min(2 ** retry_count, 5) * (0.5 + random.random())
        print(f"Worker {name} error (attempt {retry_count+1}/{retries}), re-queue: {e}")
        time.sleep(delay)
        wrapped = {"__pipe_retry": retry_count + 1, "__pipe_item": item}
        in_queue.put(wrapped, timeout=1.0)
    else:
        print(f"Worker {name} dropping item after {retries} retries: {e}")
        if on_error:
            on_error(item, e, retry_count)
    continue
```

### Interaction with shm

The re-queued item is the deserialized Python object (already unpacked from shm). When another worker gets it, the wrapper dict doesn't contain `__shm__`, so `_item_from_shm` passes it through. The `__pipe_retry` wrapper is stripped before the worker sees it.

If `share_tensors=True` and the item contains tensors, re-queueing means the tensors stay in RAM of the current process. The item gets pickled through the mp.Queue normally (torch handles tensor sharing). This is fine — retries are rare, and the item is small enough to fit in the queue.

### Interaction with autoscaler

When workers are sleeping on transient retries, input queue stays full — autoscaler wants to scale up. But more workers won't help (they'll all hit the same error).

Fix: track a per-stage `transient_retry_active` flag (mp.Value). When any worker is in transient retry sleep, set it. Autoscaler skips scale-up for that stage while flag is set.

### Interaction with completion detection

Workers check `upstream_done + queue empty` to know when to exit. Re-queued items prevent the queue from being empty, so workers won't prematurely exit.

But: if the last remaining items in the queue are all permanently-failing retries that keep cycling, the pipeline never finishes. The `retries` cap prevents this for non-transient errors. For transient errors, the pipeline correctly waits until the environment recovers.

### Ordering

Already non-guaranteed with multiple workers. Re-queued items go to back of queue, so they're processed after newer items. This is fine for the target use cases (downloads, DB writes, GPU inference).

## pipe.add() parameter changes

```python
pipe.add(worker,
    retries=0,              # max retry count for non-transient errors (0=no retry, drop on error)
    transient_errors=None,  # tuple of exception types to retry forever (added to defaults)
    skip_errors=None,       # tuple of exception types to drop immediately (no retry)
    on_error=None,          # callback(item, error, retry_count) for dropped items
)
```

## Files to modify

1. `src/pipe/_workers.py` — Add retry/re-queue logic to `_worker_run` and `_threaded_worker_run`
2. `src/pipe/pipe.py` — Accept new params in `add()`, pass them through to worker args

## Scenarios

| Situation | Classification | Behavior |
|-----------|---------------|----------|
| S3 down 2 min | Transient | Workers backoff + re-queue, resume when network returns |
| One bad audio file | Non-transient | Re-queued N times, then dropped. Other items unaffected. |
| DB pool exhausted | Transient (user-added) | Backoff until pool frees up |
| GPU OOM on large item | skip_error | Dropped immediately, no retry |
| API rate limit (429) | Transient | Backoff with jitter, no thundering herd |
| DB row not found | Non-transient | Re-queued N times (wasteful). Better as skip_error. |
| Network flap | Transient | Items cycle through queue, processed during "up" windows |
| Batcher with bad batch | Non-transient | Item re-queued, picked up by same/different batcher instance. Works. |

## Design gaps (deferred)

- **Autoscaler suppression**: Track `transient_retry_active` per stage. Implement in v2.
- **Metrics**: Count retries/drops per stage. Expose via timing_dict. Implement in v2.
- **Dead-letter persistence**: `on_error` can write to disk, but no built-in DLQ. User handles.

## Verification

```python
from pipe import Pipe

class Source:
    def __init__(self):
        self.i = 0
    def __call__(self):
        if self.i >= 20:
            return "end"
        self.i += 1
        return {"n": self.i}

class FlakyWorker:
    def load(self):
        self.seen = {}
    def __call__(self, item):
        n = item["n"]
        self.seen[n] = self.seen.get(n, 0) + 1
        if n == 5 and self.seen[n] <= 2:
            raise ValueError("fails twice then succeeds")
        if n == 10:
            raise ValueError("always fails")
        return item

pipe = Pipe()
pipe.add(Source(), outqn=50)
pipe.add(FlakyWorker(), workers=1, retries=3, outqn=0)
count = pipe.run()
# n=5 succeeds on 3rd try, n=10 dropped after 3 retries
assert count == 19, f"Expected 19, got {count}"
```
