import contextlib
import os
import signal
import threading
from queue import Empty

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Event, Queue, Value

from .lifecycle import LifecycleMixin
from .monitors import _collect_stats
from .queues import InstrumentedQueue, PipeIterator, _InputChannel  # noqa: F401 (kept importable from pipe.pipe)
from .sequential import SequentialMixin
from .shm import _cleanup_stale_shm, _item_from_shm
from .utils import _log, print_above
from .workers import _check_picklable, _is_end

if not mp.get_start_method(allow_none=True):
    mp.set_start_method("spawn", force=True)


class Pipe(LifecycleMixin, SequentialMixin):
    """Multi-stage parallel pipeline.

    Public surface lives here (construction, `add`, iteration); process lifecycle
    (spawn/restart/stop) is in `LifecycleMixin` and single-process execution in
    `SequentialMixin`. Queue helpers are in `queues.py`, worker run-loops in
    `workers.py`.
    """

    def __init__(
        self,
        sequential=False,
        debug=False,
        raise_errors=None,
        health_check_interval=30,
        expected_consumers=1,
        stats_interval=0.2,
        stats_mode="text",
        allow_full_restart=True,
        use_shm=False,
        output_shm=False,
        profile=False,
        scavenge_poll=1.0,
        scavenge_free_secs=30.0,
    ):
        # Set env vars for shm control (inherited by spawned workers)
        if not use_shm:
            os.environ["PIPE_NO_SHM"] = "1"
        elif "PIPE_NO_SHM" in os.environ:
            del os.environ["PIPE_NO_SHM"]
        if not output_shm:
            os.environ["PIPE_NO_SHM_OUTPUT"] = "1"
        elif "PIPE_NO_SHM_OUTPUT" in os.environ:
            del os.environ["PIPE_NO_SHM_OUTPUT"]
        _cleanup_stale_shm()
        self.sequential = sequential
        self.debug = debug
        self.use_shm = use_shm
        self.raise_errors = raise_errors if raise_errors is not None else sequential
        self.health_check_interval = health_check_interval
        self.allow_full_restart = allow_full_restart
        self.jobs = []
        self.queues: list[Queue] = []
        self.processes = []
        self.worker_info = []
        self.worker_configs = {}
        self.started = False
        self.should_stop = Value("i", 0)
        self.drain_event = Event()
        self.health_monitor_thread = None
        self.health_monitor_stop_event = threading.Event()
        self.stats_monitor_thread = None
        self.stats_monitor_stop_event = threading.Event()
        # GPU scavenging (see scavenge.py): per-worker slots built in start(),
        # monitor thread drives hold/freeze/thaw of workers on claimed GPUs.
        self.scavenge_poll = scavenge_poll
        self.scavenge_free_secs = scavenge_free_secs
        self.scavenge_slots = []
        self.scavenger_thread = None
        self.scavenger_stop_event = threading.Event()
        self.stats_interval = stats_interval
        self.stats_mode = stats_mode
        self.profile = profile
        self.profile_dir = None
        self.profile_rss = {}
        self.gpus = self._get_gpu_count()
        self.expected_consumers = expected_consumers

        # Manager (for timing_dict) is created lazily in start(): only for
        # multiprocessing runs with stats enabled. Sequential runs and
        # stats_interval=0 runs never pay for the manager process.
        self.manager = None
        self.timing_dict = None

        self.stage_end_counters = []
        self.stage_worker_counts = []
        self.stage_done_events = []

    def _get_gpu_count(self):
        try:
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            _log("CUDA not available, pergpu flag will be ignored")
            return 0
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
            return 0

    def add(
        self,
        func: callable,
        workers=1,
        outqn=None,
        pergpu=False,
        thread=False,
        gpu_id=None,
        gpus=None,
        cpus=None,
        cpu_threads=None,
        batch=0,
        drain=True,
        chunk=None,
        chunk_ms=10.0,
        scavenge=False,
    ):
        if hasattr(func, "__name__"):
            stage_name = func.__name__
        elif hasattr(func, "__class__"):
            stage_name = func.__class__.__name__
        else:
            stage_name = str(type(func).__name__)

        _check_picklable(func, stage_name, is_thread=thread)

        is_root_stage = len(self.jobs) == 0
        if is_root_stage and (workers > 1 or pergpu):
            print(
                f"WARNING: root stage '{stage_name}' requested multiple workers "
                f"(workers={workers}, pergpu={pergpu}); clamping to 1. Each root worker "
                f"runs the source independently and would duplicate the entire stream. "
                f"Parallelize a downstream stage instead, or shard the source externally."
            )
            workers = 1
            pergpu = False

        # Scavenge mode (scavenge.py): only run this stage's workers on GPUs no
        # other process is using, and freeze them in place (cuda-checkpoint) the
        # moment a foreign process claims their GPU. Checked before the GPU pool
        # is resolved so a misuse reports itself, not a device-id error.
        # Cooperative hold/freeze can't reach into threads sharing a process,
        # and a root stage can't pause without stalling the whole source.
        if scavenge:
            if thread:
                raise ValueError(
                    f"{stage_name}: scavenge=True is not supported with "
                    f"thread=True (freeze targets whole processes)"
                )
            if is_root_stage:
                raise ValueError(
                    f"{stage_name}: scavenge=True is not supported on the "
                    f"root stage"
                )

        gpu_count = self.gpus

        # Resolve the stage's GPU pool into a single canonical list `gpu_list`:
        #   gpus=[5,6]   -> pin this stage's workers to GPUs 5 and 6 (round-robin)
        #   pergpu=True  -> one worker per GPU across ALL GPUs (== gpus=range(N))
        #   gpu_id=N     -> pin every worker to GPU N (== gpus=[N])
        # `workers` is a per-GPU multiplier in every case, so total workers =
        # workers * len(gpu_list). None => CPU stage. This replaces hand-rolled
        # lock-file GPU pinning in user code.
        # `gpus=`/`gpu_id=` are EXPLICIT device requests: if the box can't satisfy
        # them we raise (below) rather than silently degrade — running a GPU model
        # on CPU is almost never what you want. `pergpu=True` is the only adaptive
        # mode: it means "every GPU there is", so 0 GPUs -> a loud CPU fallback.
        if gpus is not None:
            gpu_list = [int(g) for g in gpus]
            if not gpu_list:
                raise ValueError(
                    f"{stage_name}: gpus=[] is empty — pass at least one GPU id, "
                    f"or omit gpus for a CPU stage"
                )
        elif pergpu:
            if gpu_count > 0:
                gpu_list = list(range(gpu_count))
            else:
                print(
                    f"WARNING: '{stage_name}' requested pergpu=True but no CUDA GPUs "
                    f"were found; running on CPU."
                )
                gpu_list = None
        elif gpu_id is not None:
            gpu_list = [int(gpu_id)]
        else:
            gpu_list = None

        # Fail fast with a clear message if an explicitly requested GPU doesn't
        # exist (incl. a 0-GPU box), rather than crashing in a worker's set_device()
        # or, worse, silently running on CPU.
        if gpu_list is not None:
            bad = sorted({g for g in gpu_list if g < 0 or g >= gpu_count})
            if bad:
                avail = (
                    f"valid ids 0..{gpu_count - 1}" if gpu_count
                    else "no CUDA GPUs are available"
                )
                raise ValueError(
                    f"{stage_name}: gpus/gpu_id reference GPU(s) {bad}, but the box "
                    f"has {gpu_count} CUDA GPU(s) ({avail})"
                )

        # Scavenge needs a real GPU pool to have anything to yield.
        if scavenge and gpu_list is None:
            print(
                f"WARNING: '{stage_name}' requested scavenge=True but has "
                f"no GPU pool; ignoring."
            )
            scavenge = False

        is_gpu_stage = gpu_list is not None
        if is_gpu_stage:
            actual_workers = workers * len(gpu_list)
            _log(
                f"GPU pool {gpu_list} for '{stage_name}': {workers} worker(s)/GPU "
                f"({actual_workers} total)"
            )
        else:
            actual_workers = workers

        # CPU affinity pool (orthogonal to GPU pinning — a stage may set both, e.g. a
        # GPU feeder pinned to reserved cores). `cpus=[0..15]` is chunked across the
        # stage's workers (each gets a contiguous slice + threads sized to it); disjoint
        # pools on different stages reserve cores (keep mel off the GPU-feeder cores).
        # Validated here so a bad core id fails at add(), not in a worker's setaffinity.
        if cpus is not None:
            cpu_list = [int(c) for c in cpus]
            if not cpu_list:
                raise ValueError(
                    f"{stage_name}: cpus=[] is empty — pass at least one CPU id, "
                    f"or omit cpus for an unpinned stage"
                )
            ncpu = os.cpu_count() or 0
            bad = sorted({c for c in cpu_list if c < 0 or c >= ncpu})
            if bad:
                raise ValueError(
                    f"{stage_name}: cpus reference CPU(s) {bad}, but the box has "
                    f"{ncpu} CPU(s) (valid ids 0..{ncpu - 1})"
                )
        else:
            cpu_list = None

        # Per-worker BLAS/torch thread count. Independent of pinning: lifts the flat
        # 2-thread cap for a CPU-heavy stage (e.g. mel) without reserving cores. When
        # both are set, cpu_threads wins over the cpus= slice size (see _setup_cpu).
        if cpu_threads is not None and (
            isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1
        ):
            raise ValueError(
                f"{stage_name}: cpu_threads must be a positive int, got {cpu_threads!r}"
            )

        self.jobs.append(
            {
                "func": func,
                "name": stage_name,
                "outqn": outqn,
                "num_workers": actual_workers,
                "pergpu": pergpu,
                "thread": thread,
                "gpu_id": gpu_id,
                "gpus": gpu_list,  # canonical GPU pool (round-robin per worker)
                "cpus": cpu_list,  # canonical CPU pool (chunked per worker), or None
                "cpu_threads": cpu_threads,  # explicit per-worker thread count, or None
                "is_gpu_stage": is_gpu_stage,
                # Opportunistic GPU use: workers hold/freeze when a foreign
                # process claims their GPU (see scavenge.py).
                "scavenge": scavenge,
                "batch": batch,
                "drain": drain,
                # Output-edge chunking: bundle N serialized items into one queue
                # message (huge win for small/metadata items; the per-item torch
                # fd-sharing is unchanged). None = auto: adopt the downstream
                # stage's batch size (resolved in start()). 0 = force off.
                "chunk": chunk,
                "chunk_ms": chunk_ms,
            }
        )

    def get_stats(self):
        return _collect_stats(self)

    def print(self, msg):
        """Print a message without clobbering the stats display.

        In text stats mode the message lands on its own line above the pinned
        stats line; in rich mode it routes through the Live-aware console; with
        stats off it is a plain print.
        """
        console = getattr(self, "_rich_console", None)
        if console is not None:
            console.print(msg)
        else:
            print_above(msg)

    def __iter__(self):
        if not self.started:
            self.start()

        if self.sequential:
            yield from self._sequential_run()
            return

        # Catch the first Ctrl+C at the OS level so it drains regardless of
        # whether Python is currently in pipe code or in the user's loop body.
        # The default handler is restored after one hit, so a second Ctrl+C
        # raises KeyboardInterrupt as usual and the iterator force-stops below.
        prev_handler = None
        is_main_thread = threading.current_thread() is threading.main_thread()
        if is_main_thread:
            def _drain_on_sigint(signum, frame):
                signal.signal(signal.SIGINT, prev_handler if prev_handler is not None else signal.SIG_DFL)
                self.drain_event.set()
                stopped = [j["name"] for j in self.jobs if not j.get("drain", True)]
                stopped_str = ", ".join(stopped) if stopped else "root"
                in_flight = sum(
                    q.qsize() * max(1, j.get("chunk_eff", 0))
                    for q, j in zip(self.queues, self.jobs)
                )
                print(
                    f"Starting to drain... {stopped_str} stopped (their inputs are not drained); "
                    f"finishing ~{in_flight} items already in the pipeline. "
                    "(Ctrl+C again to force stop)",
                    flush=True,
                )
            prev_handler = signal.signal(signal.SIGINT, _drain_on_sigint)

        try:
            final_done = self.stage_done_events[-1]
            consecutive_empty = 0
            EMPTY_THRESHOLD = 5
            # Chunk-aware reader over the final queue. Rebuilt after restart()
            # because restart may recreate the underlying queue objects.
            out_ch = _InputChannel(self.queues[-1])

            while True:
                if self.should_stop.value == 2:
                    _log("Worker encountered connection error - restarting pipeline...")
                    self.restart(reason="Worker connection error")
                    out_ch = _InputChannel(self.queues[-1])
                    consecutive_empty = 0
                    continue

                try:
                    item = out_ch.get(timeout=0.1)
                    consecutive_empty = 0
                    item = _item_from_shm(item)

                    if _is_end(item):
                        continue

                    yield item
                except Empty:
                    consecutive_empty += 1
                    if final_done.is_set() and consecutive_empty >= EMPTY_THRESHOLD:
                        _log("Iterator: pipeline complete")
                        self._stop()
                        return
                except (ConnectionError, FileNotFoundError):
                    self.restart()
                    out_ch = _InputChannel(self.queues[-1])
                    consecutive_empty = 0
                except KeyboardInterrupt:
                    # Second Ctrl+C: default handler restored, KeyboardInterrupt fired.
                    _log("Force stopping pipeline")
                    self._stop(force=True)
                    return
        finally:
            if is_main_thread and prev_handler is not None:
                with contextlib.suppress(Exception):
                    signal.signal(signal.SIGINT, prev_handler)

    def run(self):
        count = 0
        for _ in self:
            count += 1
        return count

    def __del__(self):
        try:
            if self.started and (self.processes or self.queues):
                self._stop(force=True)
        except Exception:
            pass
