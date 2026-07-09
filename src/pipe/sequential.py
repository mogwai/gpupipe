"""Sequential (single-process) execution of a pipeline, for debugging/testing.

Mixed into `Pipe`; runs all stages in-process with no multiprocessing, preserving
generator/list/None/batch/run()/push() semantics so a `sequential=True` run behaves
like the parallel one."""
import inspect

from .utils import _log
from .workers import _is_end


class SequentialMixin:
    def _sequential_run(self):
        _log("Running in sequential mode")

        if not self.jobs:
            return

        workers = [job["func"] for job in self.jobs]

        run_stages = set()
        for i, (w, job) in enumerate(zip(workers, self.jobs)):
            if hasattr(w, "run") and i > 0:
                run_stages.add(i)

        for worker in workers:
            if hasattr(worker, "load"):
                worker.load()

        pending_items = []
        root_ended = False

        # Sequential-mode push(stage, item): re-enqueue the item to run at the
        # target stage (same semantics as the multiprocessing worker.push — send
        # an item back to an earlier stage). Resolves a stage name to its index.
        seq_stage_names = [job["name"] for job in self.jobs]

        def _make_seq_push():
            def _push(stage, item, block=True):
                if item is None:
                    return
                if isinstance(stage, bool):
                    raise TypeError("push: stage must be an index, name, or class")
                if isinstance(stage, int):
                    idx = stage
                else:
                    name = (
                        stage if isinstance(stage, str)
                        else (stage if isinstance(stage, type) else type(stage)).__name__
                    )
                    idx = seq_stage_names.index(name)
                if idx <= 0 or idx >= len(workers):
                    raise ValueError(
                        f"push: stage {stage!r} (idx {idx}) has no input to push to"
                    )
                pending_items.append((idx, item))
            return _push

        for i, worker in enumerate(workers):
            if i > 0:
                worker.push = _make_seq_push()

        def _add_result(result, next_stage):
            if result is None:
                return
            if inspect.isgenerator(result):
                for res_item in result:
                    if res_item is not None and not _is_end(res_item):
                        pending_items.append((next_stage, res_item))
            elif _is_end(result):
                return
            elif isinstance(result, list):
                for res_item in result:
                    if res_item is not None and not _is_end(res_item):
                        pending_items.append((next_stage, res_item))
            else:
                pending_items.append((next_stage, result))

        def _process_callable(worker, stage_idx, item):
            job = self.jobs[stage_idx]
            batch_sz = job.get("batch", 0)

            if batch_sz > 0:
                batch = [item]
                while len(batch) < batch_sz and pending_items and pending_items[0][0] == stage_idx:
                    _, next_item = pending_items.pop(0)
                    batch.append(next_item)
                result = worker(batch)
            else:
                result = worker(item)
            _add_result(result, stage_idx + 1)

        while not root_ended or pending_items:
            if not root_ended:
                root_result = workers[0]()

                if inspect.isgenerator(root_result):
                    for root_item in root_result:
                        if root_item is not None and not _is_end(root_item):
                            pending_items.append((1, root_item))
                    root_ended = True
                elif root_result is None:
                    pass
                elif _is_end(root_result):
                    root_ended = True
                elif isinstance(root_result, list):
                    for root_item in root_result:
                        if _is_end(root_item):
                            root_ended = True
                            break
                        if root_item is not None:
                            pending_items.append((1, root_item))
                else:
                    pending_items.append((1, root_result))

            if pending_items:
                stage_idx, item = pending_items.pop(0)

                if stage_idx >= len(workers):
                    yield item
                    continue

                worker = workers[stage_idx]

                if stage_idx in run_stages:
                    # Collect all currently pending items for this stage
                    buffer = [item]
                    remaining = []
                    for si, it in pending_items:
                        if si == stage_idx:
                            buffer.append(it)
                        else:
                            remaining.append((si, it))
                    pending_items[:] = remaining

                    batch_sz = self.jobs[stage_idx].get("batch", 1) or 1

                    def _make_pull(buf, bs):
                        def _pull(n=bs):
                            items = buf[:n]
                            del buf[:n]
                            return items
                        return _pull

                    def _make_put(pi, next_stage):
                        def _put(item):
                            if item is not None:
                                pi.append((next_stage, item))
                        return _put

                    worker.pull = _make_pull(buffer, batch_sz)
                    worker.put = _make_put(pending_items, stage_idx + 1)
                    try:
                        worker.run()
                    except Exception as e:
                        if self.raise_errors:
                            raise
                        print(f"Error in run() worker at stage {stage_idx}: {e}")
                    continue

                try:
                    _process_callable(worker, stage_idx, item)
                except Exception as e:
                    if self.raise_errors:
                        raise
                    print(f"Error in worker at stage {stage_idx}: {e}, continuing...")

        # Flush callable workers (run() workers manage their own flushing)
        for stage_idx, worker in enumerate(workers[1:], 1):
            if stage_idx in run_stages:
                continue
            if hasattr(worker, "flush"):
                for flushed_item in worker.flush():
                    if flushed_item is not None:
                        pending_items.append((stage_idx + 1, flushed_item))

        # Process remaining items from flush
        while pending_items:
            stage_idx, item = pending_items.pop(0)

            if stage_idx >= len(workers):
                yield item
                continue

            worker = workers[stage_idx]
            try:
                _process_callable(worker, stage_idx, item)
            except Exception as e:
                if self.raise_errors:
                    raise
                continue
