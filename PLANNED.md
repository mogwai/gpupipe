# Planned features

Features that are designed and implemented but **not wired into the live code
path**. The implementation is preserved so it can be re-integrated without
re-deriving it. Preserved code lives under `src/pipe/_planned/` — never
imported by the live package, though it does ship in wheels as inert reference
code (a deliberate choice, see the note in `pyproject.toml`). Its tests live
under `tests/planned/`, excluded from collection via `collect_ignore` in
`tests/conftest.py` plus module-level skip marks. A live drift guard,
`tests/test_planned_preserved.py`, always runs and fails if the preserved code
falls out of sync with the live worker signatures.

---

## Autoscaling

**Status:** removed from the live path on 2026-07-09; implementation preserved.

- Implementation: [`src/pipe/_planned/autoscale.py`](src/pipe/_planned/autoscale.py)
- Tests: [`tests/planned/test_autoscale.py`](tests/planned/test_autoscale.py) (skipped)
- Demo: [`examples/planned/autoscale_demo.py`](examples/planned/autoscale_demo.py)

### Idea

Scale the number of worker processes per stage up and down at runtime based on
queue pressure, so a pipeline adapts to uneven per-stage throughput without the
user hand-tuning `workers=` for every stage.

A background thread samples each autoscale-enabled stage once per second:

- **Scale UP** — input-queue fill `>= 80%` for 3 consecutive samples → spawn one
  worker process (same config). Suppressed when system CPU is `> 85%`
  (saturated) or the stage's output queue is `>= 90%` full (downstream
  bottleneck — more workers wouldn't drain).
- **Scale DOWN** — input-queue fill `<= 20%` for 5 consecutive samples → put a
  `WorkerStop` sentinel (`pipe.types.WorkerStop`) on the input queue; the
  worker finishes its current item, re-queues any chunk-buffered items for its
  siblings, and exits. Never below `min_workers`.
- **Cooldown** — 3s minimum between scaling actions per stage.
- **Never autoscales** GPU stages (capped by device count), threaded stages (one
  process, N threads — the process-spawn model doesn't apply), or `cpus=`-pinned
  stages (a static core pin can't track a live worker count).

### Intended public API (removed)

```python
# Global — every eligible stage scales
pipe = Pipe(autoscale=True, max_workers_per_stage=8)
pipe.add(Generator(), outqn=20)
pipe.add(Worker(), workers=1, outqn=20)                 # scales 1..8
pipe.add(Worker(), workers=1, outqn=20, max_workers=4)  # custom ceiling

# Per-stage
pipe = Pipe()
pipe.add(Generator(), outqn=20)
pipe.add(Worker(), workers=1, outqn=20,
         autoscale=True, min_workers=1, max_workers=6)
```

- `Pipe(autoscale=False, max_workers_per_stage=8)` — global default + cap.
- `add(..., autoscale=None, min_workers=None, max_workers=None)` — per-stage
  override (`autoscale=None` inherits the global flag).
- Live worker count is observable at `pipe.stage_worker_counts[i].value`.

### Why it was pulled

Removed for now to shrink the live surface area; the mechanism is sound but not
needed yet. The queue/worker primitives it relies on remain in the core:
`stage_worker_counts`, `stage_end_counters`, and the inert `WorkerStop`
graceful-exit handling in `_worker_run` / `_threaded_worker_run` (see
`src/pipe/workers.py`) are all still present, so re-enabling is additive.

### Re-integration checklist

The single authoritative checklist is the module docstring at the top of
[`src/pipe/_planned/autoscale.py`](src/pipe/_planned/autoscale.py) — it is
kept next to the code so refactors update both together. Follow it rather
than any summary. When re-enabling, also move `tests/planned/test_autoscale.py`
back to `tests/` and drop its module-level `pytest.mark.skip` and the
`collect_ignore` entry in `tests/conftest.py`.

### Known follow-ups (from `docs/retry.md`)

When workers sleep on transient retries the input queue stays full, so the
autoscaler wants to scale up — but more workers just hit the same error. Fix:
track a per-stage `transient_retry_active` flag (`mp.Value`); skip scale-up for a
stage while that flag is set.
