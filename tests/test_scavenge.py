"""Scavenge-mode tests: GPU probing, add() validation, the worker-side hold
gate, and the scavenger thread's freeze/thaw decisions.

No GPU and no cuda-checkpoint binary needed — nvidia-smi and every
cuda-checkpoint action are stubbed at the subprocess boundary, so these run on
any machine (including a mac laptop).
"""
import os
import threading
import time

import pytest
import torch.multiprocessing as mp

from pipe import Pipe
from pipe import scavenge as sc
from pipe.types import End
from pipe.workers import _scavenge_hold_wait

# The hold gate inspects raw queue payloads; keep the shm layer off so a real
# item is recognisable (mirrors Pipe(use_shm=False), the default).
os.environ.setdefault("PIPE_NO_SHM", "1")


# === GPU PROBING ===

SMI_GPUS = "0, GPU-aaaa\n1, GPU-bbbb\n2, GPU-cccc"


def _fake_run(smi_apps, gpu_free=None, log=None):
    """Stub for scavenge._run covering the three commands it issues."""

    def run(cmd, timeout=sc.ACTION_TIMEOUT_S):
        if log is not None:
            log.append(list(cmd))
        joined = " ".join(cmd)
        if "--query-gpu=index,uuid" in joined:
            return 0, SMI_GPUS
        if "--query-compute-apps" in joined:
            return 0, smi_apps
        if "--query-gpu=index,memory.free" in joined:
            return 0, gpu_free or "0, 32000\n1, 32000\n2, 32000"
        return 0, ""  # cuda-checkpoint actions succeed

    return run


@pytest.fixture(autouse=True)
def _clear_uuid_cache():
    sc._uuid_to_index_cache = None
    yield
    sc._uuid_to_index_cache = None


def test_busy_gpus_ignores_our_own_pids(monkeypatch):
    apps = "GPU-aaaa, 111, 4000\nGPU-cccc, 222, 8000"
    monkeypatch.setattr(sc, "_run", _fake_run(apps))
    # 111 is ours -> only GPU 2 (uuid cccc, foreign pid 222) counts as busy.
    assert sc.busy_physical_gpus({111}) == {2}
    assert sc.busy_physical_gpus(set()) == {0, 2}


def test_busy_gpus_empty_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(sc, "_run", lambda cmd, timeout=None: (1, "not found"))
    assert sc.busy_physical_gpus(set()) == set()


def test_physical_gpu_honours_cuda_visible_devices(monkeypatch):
    monkeypatch.setattr(sc, "_run", _fake_run(""))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert sc.physical_gpu(0) == 4
    assert sc.physical_gpu(3) == 7
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert sc.physical_gpu(3) == 3


def test_physical_gpu_resolves_uuid_cvd_entries(monkeypatch):
    monkeypatch.setattr(sc, "_run", _fake_run(""))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-cccc,GPU-aaaa")
    assert sc.physical_gpu(0) == 2
    assert sc.physical_gpu(1) == 0


# === add() VALIDATION ===


def _passthrough(x):
    return x


def test_scavenge_rejects_thread_and_root_stages():
    p = Pipe(stats_interval=0)
    with pytest.raises(ValueError, match="root stage"):
        p.add(_passthrough, scavenge=True, gpu_id=0)

    p2 = Pipe(stats_interval=0)
    p2.add(_passthrough)
    with pytest.raises(ValueError, match="thread=True"):
        p2.add(_passthrough, scavenge=True, thread=True, gpu_id=0)


def test_scavenge_ignored_without_gpu_pool(capsys):
    p = Pipe(stats_interval=0)
    p.add(_passthrough)
    p.add(_passthrough, scavenge=True)
    assert p.jobs[1]["scavenge"] is False
    assert "no GPU pool" in capsys.readouterr().out


# === WORKER-SIDE HOLD GATE ===


def _hold_args(hold, should_stop, upstream_done, queue, end_counter, worker_count):
    return dict(
        hold=hold,
        should_stop=should_stop,
        upstream_done=upstream_done,
        in_queue=queue,
        stage_end_counter=end_counter,
        stage_worker_count=worker_count,
        worker_desc="test_worker",
    )


def test_hold_releases_when_scavenger_clears_it():
    hold = mp.Value("i", 1)
    result = {}

    def run():
        result["released"] = _scavenge_hold_wait(
            **_hold_args(hold, mp.Value("i", 0), None, None, None, None)
        )

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.4)
    assert "released" not in result, "worker proceeded while still held"
    hold.value = 0
    t.join(timeout=5)
    assert result["released"] is True


def test_held_worker_exits_when_stream_ends_with_empty_queue():
    """Upstream done + nothing left to do => the held worker stops waiting so
    the stage can finish (End sentinels and counters stay correct)."""
    hold = mp.Value("i", 1)
    done = mp.Event()
    done.set()
    q = mp.Queue()
    result = {}

    def run():
        result["released"] = _scavenge_hold_wait(
            **_hold_args(hold, mp.Value("i", 0), done, q,
                         mp.Value("i", 0), mp.Value("i", 1))
        )

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=5)
    assert result["released"] is False
    assert hold.value == 1, "hold must not be cleared by the worker itself"


def test_held_worker_keeps_waiting_while_real_work_remains():
    """A queued item means work needs this GPU: keep holding (and put the item
    back) rather than exiting and dropping it."""
    hold = mp.Value("i", 1)
    done = mp.Event()
    done.set()
    q = mp.Queue()
    q.put({"payload": "work"})
    time.sleep(0.2)
    result = {}

    def run():
        result["released"] = _scavenge_hold_wait(
            **_hold_args(hold, mp.Value("i", 0), done, q,
                         mp.Value("i", 0), mp.Value("i", 1))
        )

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.8)
    assert "released" not in result, "worker exited while work was queued"
    hold.value = 0
    t.join(timeout=5)
    assert result["released"] is True
    assert q.get(timeout=2) == {"payload": "work"}, "item was not preserved"


def test_held_worker_drains_end_sentinels_then_exits():
    hold = mp.Value("i", 1)
    done = mp.Event()
    done.set()
    q = mp.Queue()
    q.put(End)
    time.sleep(0.2)
    result = {}

    def run():
        result["released"] = _scavenge_hold_wait(
            **_hold_args(hold, mp.Value("i", 0), done, q,
                         mp.Value("i", 0), mp.Value("i", 1))
        )

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout=5)
    assert result["released"] is False


def test_held_worker_waits_for_siblings_before_exiting():
    """With 3 workers at the stage and none finished, an item on the queue may
    be another worker's — this one must not race ahead and exit."""
    hold = mp.Value("i", 1)
    done = mp.Event()
    done.set()
    q = mp.Queue()
    result = {}

    def run():
        result["released"] = _scavenge_hold_wait(
            **_hold_args(hold, mp.Value("i", 0), done, q,
                         mp.Value("i", 0), mp.Value("i", 3))
        )

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.6)
    assert "released" not in result
    hold.value = 0
    t.join(timeout=5)


def test_hold_aborts_on_should_stop():
    hold = mp.Value("i", 1)
    stop = mp.Value("i", 0)
    result = {}

    def run():
        result["released"] = _scavenge_hold_wait(
            **_hold_args(hold, stop, None, None, None, None)
        )

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.3)
    stop.value = 1
    t.join(timeout=5)
    assert result["released"] is False


# === SCAVENGER THREAD ===


class _FakeProc:
    def __init__(self, pid):
        self.pid = pid

    def is_alive(self):
        return True


class _FakePipe:
    """Minimal stand-in for the Pipe attributes the scavenger thread touches."""

    def __init__(self, slots, pids):
        self.should_stop = mp.Value("i", 0)
        self.scavenge_slots = slots
        self.worker_info = [
            (_FakeProc(pid), wid, "Stage") for wid, pid in pids.items()
        ]
        self.messages = []

    def print(self, msg):
        self.messages.append(msg)


def _slot(worker_id="w0", physical=0, hold_value=0):
    return {
        "stage_name": "Stage",
        "worker_id": worker_id,
        "gpu": physical,
        "physical": physical,
        "hold": mp.Value("i", hold_value),
        "frozen": False,
        "frozen_mib": 0,
    }


def _run_scavenger(fake, monkeypatch, apps, calls, gpu_free=None, polls=3,
                   free_secs=0.0):
    monkeypatch.setattr(sc, "_run", _fake_run(apps, gpu_free, log=calls))
    monkeypatch.setattr(sc, "_find_cuda_checkpoint", lambda: "/fake/cuda-checkpoint")
    monkeypatch.setattr(sc, "_host_available_mib", lambda: 999_999)
    stop = threading.Event()
    t = threading.Thread(
        target=sc._scavenger_thread, args=(fake, stop, 0.05, free_secs), daemon=True
    )
    t.start()
    time.sleep(0.05 * polls + 0.3)
    stop.set()
    t.join(timeout=5)


def _ckpt_actions(calls):
    out = []
    for c in calls:
        if c and c[0] == "/fake/cuda-checkpoint":
            out.append(c[c.index("--action") + 1])
    return out


def test_foreign_process_triggers_freeze(monkeypatch):
    slot = _slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    # Our worker (500) is on GPU 0 holding 6 GB; foreign pid 900 claims it too.
    apps = "GPU-aaaa, 500, 6000\nGPU-aaaa, 900, 1000"
    calls = []
    _run_scavenger(fake, monkeypatch, apps, calls)

    assert _ckpt_actions(calls)[:2] == ["lock", "checkpoint"]
    assert slot["frozen"] is True
    assert slot["frozen_mib"] == 6000
    assert slot["hold"].value == 1
    assert any("claimed by pid(s)" in m for m in fake.messages)


def test_idle_gpu_is_not_touched(monkeypatch):
    slot = _slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    _run_scavenger(fake, monkeypatch, apps="GPU-aaaa, 500, 6000", calls=calls)

    assert _ckpt_actions(calls) == []
    assert slot["frozen"] is False
    assert slot["hold"].value == 0


def test_freed_gpu_thaws_frozen_worker(monkeypatch):
    slot = _slot(physical=0, hold_value=1)
    slot["frozen"] = True
    slot["frozen_mib"] = 6000
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    # No compute apps at all: GPU 0 is idle again.
    _run_scavenger(fake, monkeypatch, apps="", calls=calls)

    assert _ckpt_actions(calls)[:2] == ["restore", "unlock"]
    assert slot["frozen"] is False
    assert slot["hold"].value == 0


def test_thaw_refused_when_vram_is_gone(monkeypatch):
    """A restore that OOMs is unrecoverable, so a worker whose VRAM has been
    taken stays parked instead."""
    slot = _slot(physical=1, hold_value=1)
    slot["frozen"] = True
    slot["frozen_mib"] = 20000
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    # GPU 1 has no compute apps visible, but only 3 GB free (another job's
    # allocation has not shown up as a compute PID yet).
    _run_scavenger(
        fake, monkeypatch, apps="", calls=calls,
        gpu_free="0, 32000\n1, 3000\n2, 32000",
    )

    assert "restore" not in _ckpt_actions(calls)
    assert slot["frozen"] is True
    assert any("staying parked" in m for m in fake.messages)


def test_freeze_refused_when_host_ram_is_short(monkeypatch):
    slot = _slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    monkeypatch.setattr(sc, "_run", _fake_run(
        "GPU-aaaa, 500, 20000\nGPU-aaaa, 900, 1000", log=calls))
    monkeypatch.setattr(sc, "_find_cuda_checkpoint", lambda: "/fake/cuda-checkpoint")
    monkeypatch.setattr(sc, "_host_available_mib", lambda: 1000)
    stop = threading.Event()
    t = threading.Thread(
        target=sc._scavenger_thread, args=(fake, stop, 0.05, 0.0), daemon=True
    )
    t.start()
    time.sleep(0.4)
    stop.set()
    t.join(timeout=5)

    assert _ckpt_actions(calls) == []
    assert slot["frozen"] is False
    assert any("host RAM" in m for m in fake.messages)


def test_missing_checkpoint_binary_warns_once(monkeypatch):
    slot = _slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    monkeypatch.setattr(
        sc, "_run", _fake_run("GPU-aaaa, 500, 6000\nGPU-aaaa, 900, 1000"))
    monkeypatch.setattr(sc, "_find_cuda_checkpoint", lambda: None)
    stop = threading.Event()
    t = threading.Thread(
        target=sc._scavenger_thread, args=(fake, stop, 0.05, 0.0), daemon=True
    )
    t.start()
    time.sleep(0.4)
    stop.set()
    t.join(timeout=5)

    warnings = [m for m in fake.messages if "no cuda-checkpoint" in m]
    assert len(warnings) == 1
    # The hold still goes up so the worker parks itself at its next chance.
    assert slot["hold"].value == 1


# === CHECKPOINT SAFETY GUARD ===
#
# Checkpointing a process whose device memory is shared with another does not
# fail safely — the driver leaves a half-done checkpoint and the job can die.
# Skipping costs one GPU we wanted to hand back; getting it wrong kills work.


def test_cpu_output_is_not_flagged():
    import torch

    from pipe.workers import _has_cuda_tensor

    cpu = torch.zeros(4)
    assert _has_cuda_tensor({"mel": cpu}) is False
    assert _has_cuda_tensor({"text": "hi", "n": 3}) is False
    assert _has_cuda_tensor([cpu, cpu]) is False
    assert _has_cuda_tensor({"a": {"b": cpu}}) is False


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs a GPU"
)
def test_cuda_output_is_flagged():
    import torch

    from pipe.workers import _has_cuda_tensor

    gpu = torch.zeros(4, device="cuda")
    assert _has_cuda_tensor(gpu) is True
    assert _has_cuda_tensor({"mel": gpu}) is True
    assert _has_cuda_tensor([{"mel": gpu}]) is True
    # Depth-limited: deeper than 2 levels is not worth the per-item cost.
    assert _has_cuda_tensor({"a": {"b": {"c": gpu}}}) is False


def test_hazard_none_for_a_lone_worker(monkeypatch):
    monkeypatch.setattr(sc, "_has_checkpoint_job_file", lambda pid: False)
    slot = _slot(physical=0)
    apps = [(0, 500, 6000), (0, 900, 1000)]  # 900 is foreign, not ours
    assert sc._checkpoint_hazard(slot, 500, apps, {500}) is None


def test_hazard_when_our_workers_share_a_gpu(monkeypatch):
    """Two of our own processes on one GPU may share device memory."""
    monkeypatch.setattr(sc, "_has_checkpoint_job_file", lambda pid: False)
    slot = _slot(physical=0)
    apps = [(0, 500, 6000), (0, 501, 6000)]
    hazard = sc._checkpoint_hazard(slot, 500, apps, {500, 501})
    assert hazard and "shares GPU 0" in hazard and "[501]" in hazard


def test_job_file_makes_a_shared_gpu_safe(monkeypatch):
    """cuda-checkpoint --launch-job makes a multi-process job checkpointable
    as a unit, so the shared-GPU objection no longer applies."""
    monkeypatch.setattr(sc, "_has_checkpoint_job_file", lambda pid: True)
    slot = _slot(physical=0)
    apps = [(0, 500, 6000), (0, 501, 6000)]
    assert sc._checkpoint_hazard(slot, 500, apps, {500, 501}) is None


def test_hazard_when_worker_exports_cuda_tensors(monkeypatch):
    monkeypatch.setattr(sc, "_has_checkpoint_job_file", lambda pid: False)
    slot = _slot(physical=0)
    slot["ipc"] = mp.Value("i", 1)
    hazard = sc._checkpoint_hazard(slot, 500, [(0, 500, 6000)], {500})
    assert hazard and "CUDA IPC" in hazard


def test_scavenger_skips_an_unsafe_worker_instead_of_freezing(monkeypatch):
    slot = _slot(physical=0)
    slot["ipc"] = mp.Value("i", 1)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    monkeypatch.setattr(sc, "_has_checkpoint_job_file", lambda pid: False)
    _run_scavenger(
        fake, monkeypatch,
        apps="GPU-aaaa, 500, 25000\nGPU-aaaa, 900, 1000", calls=calls, polls=6)

    assert _ckpt_actions(calls) == [], "checkpointed a worker that was unsafe"
    assert slot["frozen"] is False
    warnings = [m for m in fake.messages if "cannot be checkpointed" in m]
    assert len(warnings) == 1, "should warn exactly once, not every poll"


def test_unsafe_worker_can_still_park(monkeypatch):
    """The guard gates only the checkpoint fallback — a stage releasing its
    own memory never touches the driver's checkpoint path."""
    slot = _park_slot(physical=0)
    slot["ipc"] = mp.Value("i", 1)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    state = {"apps": "GPU-aaaa, 500, 25000\nGPU-aaaa, 900, 1000"}

    def acker():
        while slot["park"].value != 1:
            time.sleep(0.01)
        state["apps"] = "GPU-aaaa, 900, 1000"
        slot["park"].value = 2

    threading.Thread(target=acker, daemon=True).start()
    monkeypatch.setattr(
        sc, "_run", lambda cmd, timeout=sc.ACTION_TIMEOUT_S: (
            _fake_run(state["apps"], log=calls)(cmd, timeout)))
    monkeypatch.setattr(sc, "_find_cuda_checkpoint", lambda: "/fake/cuda-checkpoint")
    monkeypatch.setattr(sc, "_host_available_mib", lambda: 999_999)
    stop = threading.Event()
    t = threading.Thread(
        target=sc._scavenger_thread, args=(fake, stop, 0.05, 0.0), daemon=True)
    t.start()
    time.sleep(0.6)
    stop.set()
    t.join(timeout=5)

    assert slot["parked"] is True
    assert _ckpt_actions(calls) == []


# === COOPERATIVE PARK ===


def _park_slot(worker_id="w0", physical=0):
    s = _slot(worker_id, physical)
    s["park"] = mp.Value("i", 0)
    s["supports_park"] = True
    s["parked"] = False
    return s


def test_park_preferred_over_checkpoint(monkeypatch):
    """A stage that can release its own GPU memory should never pay for a
    VRAM->host checkpoint copy."""
    slot = _park_slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []

    # The 'worker' acks the park request and its memory goes away.
    state = {"apps": "GPU-aaaa, 500, 25000\nGPU-aaaa, 900, 1000"}

    def acker():
        while slot["park"].value != 1:
            time.sleep(0.01)
        state["apps"] = "GPU-aaaa, 900, 1000"  # our pid released everything
        slot["park"].value = 2

    threading.Thread(target=acker, daemon=True).start()
    monkeypatch.setattr(
        sc, "_run", lambda cmd, timeout=sc.ACTION_TIMEOUT_S: (
            _fake_run(state["apps"], log=calls)(cmd, timeout)))
    monkeypatch.setattr(sc, "_find_cuda_checkpoint", lambda: "/fake/cuda-checkpoint")
    monkeypatch.setattr(sc, "_host_available_mib", lambda: 999_999)
    stop = threading.Event()
    t = threading.Thread(
        target=sc._scavenger_thread, args=(fake, stop, 0.05, 0.0), daemon=True)
    t.start()
    time.sleep(0.6)
    stop.set()
    t.join(timeout=5)

    assert slot["parked"] is True
    assert slot["frozen"] is False
    assert _ckpt_actions(calls) == [], "should not have checkpointed"
    assert any("without a checkpoint copy" in m for m in fake.messages)


def test_park_falls_back_to_checkpoint_when_memory_stays(monkeypatch):
    """on_park() acked but the memory is still there — checkpoint it."""
    slot = _park_slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []

    def acker():
        while slot["park"].value != 1:
            time.sleep(0.01)
        slot["park"].value = 2  # ack, but memory (below) never drops

    threading.Thread(target=acker, daemon=True).start()
    _run_scavenger(
        fake, monkeypatch,
        apps="GPU-aaaa, 500, 25000\nGPU-aaaa, 900, 1000", calls=calls)

    assert slot["parked"] is False
    assert _ckpt_actions(calls)[:2] == ["lock", "checkpoint"]
    assert any("checkpointing the remainder" in m for m in fake.messages)


def test_park_emits_everything_on_park_returns():
    """Whatever on_park() drains must reach the output queue before the GPU is
    given up — a dropped item can strand its whole parent work unit
    downstream (e.g. a raw file that never re-assembles)."""
    from pipe.workers import _maybe_park

    class Stage:
        def on_park(self):
            return [{"id": i} for i in range(5)]

    park = mp.Value("i", 1)
    emitted = []
    stop = mp.Value("i", 0)

    def unpark_later():
        while park.value != 2:
            time.sleep(0.01)
        park.value = 0

    threading.Thread(target=unpark_later, daemon=True).start()
    _maybe_park(park, Stage(), emitted.append, stop, "w")

    assert emitted == [{"id": i} for i in range(5)], "drained work was lost"


def test_park_refused_when_stage_cannot_release_safely():
    """on_park() raising must NOT free the GPU — anything it was holding would
    be dropped. Refuse, and let the checkpoint path preserve it instead."""
    from pipe.workers import _maybe_park

    class BadStage:
        def on_park(self):
            raise RuntimeError("could not drain in-flight work")

    park = mp.Value("i", 1)
    emitted = []
    _maybe_park(park, BadStage(), emitted.append, mp.Value("i", 0), "w")

    assert park.value == 3, "must signal refusal, not a successful park"
    assert emitted == []


def test_scavenger_checkpoints_a_refused_park(monkeypatch):
    slot = _park_slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []

    def refuser():
        while slot["park"].value != 1:
            time.sleep(0.01)
        slot["park"].value = 3  # on_park() failed

    threading.Thread(target=refuser, daemon=True).start()
    _run_scavenger(
        fake, monkeypatch,
        apps="GPU-aaaa, 500, 25000\nGPU-aaaa, 900, 1000", calls=calls)

    assert slot["parked"] is False
    assert _ckpt_actions(calls)[:2] == ["lock", "checkpoint"]
    assert any("refused to park" in m for m in fake.messages)
    assert slot["park"].value == 0


def test_park_unparks_via_on_unpark_then_resumes():
    from pipe.workers import _maybe_park

    events = []

    class Stage:
        def on_park(self):
            events.append("park")
            return []

        def on_unpark(self):
            events.append("unpark")

    park = mp.Value("i", 1)

    def release():
        while park.value != 2:
            time.sleep(0.01)
        time.sleep(0.1)
        park.value = 0

    threading.Thread(target=release, daemon=True).start()
    _maybe_park(park, Stage(), lambda x: None, mp.Value("i", 0), "w")

    assert events == ["park", "unpark"]


def test_partial_park_then_thaw_releases_the_park_too(monkeypatch):
    """Regression: a worker that parked (acked) and then had its residual
    checkpointed waits on BOTH the restore and the park flag. Thawing without
    clearing `park` strands it — it never rebuilds and its GPU sits idle."""
    slot = _park_slot(physical=0)
    slot["frozen"] = True
    slot["frozen_mib"] = 1150
    slot["park"].value = 2  # acked the park, then got checkpointed
    slot["hold"].value = 1
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    _run_scavenger(fake, monkeypatch, apps="", calls=calls)

    assert _ckpt_actions(calls)[:2] == ["restore", "unlock"]
    assert slot["frozen"] is False
    assert slot["park"].value == 0, "worker left parked forever after thaw"
    assert slot["hold"].value == 0


def test_partial_park_checkpoints_the_slim_footprint(monkeypatch):
    """After a partial park, frozen_mib must reflect what's actually left —
    a stale pre-park figure would make the thaw preflight refuse to restore."""
    slot = _park_slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    state = {"apps": "GPU-aaaa, 500, 29710\nGPU-aaaa, 900, 1000"}

    def acker():
        while slot["park"].value != 1:
            time.sleep(0.01)
        state["apps"] = "GPU-aaaa, 500, 1150\nGPU-aaaa, 900, 1000"
        slot["park"].value = 2

    threading.Thread(target=acker, daemon=True).start()
    monkeypatch.setattr(
        sc, "_run", lambda cmd, timeout=sc.ACTION_TIMEOUT_S: (
            _fake_run(state["apps"], log=calls)(cmd, timeout)))
    monkeypatch.setattr(sc, "_find_cuda_checkpoint", lambda: "/fake/cuda-checkpoint")
    monkeypatch.setattr(sc, "_host_available_mib", lambda: 999_999)
    stop = threading.Event()
    t = threading.Thread(
        target=sc._scavenger_thread, args=(fake, stop, 0.05, 0.0), daemon=True)
    t.start()
    time.sleep(0.7)
    stop.set()
    t.join(timeout=5)

    assert slot["frozen"] is True
    assert slot["frozen_mib"] == 1150, (
        f"checkpointed the stale pre-park footprint ({slot['frozen_mib']})")


def test_park_ack_timeout_falls_back_to_checkpoint(monkeypatch):
    """A worker stuck in a long item never acks — don't wait forever."""
    monkeypatch.setattr(sc, "PARK_ACK_TIMEOUT_S", 0.2)
    slot = _park_slot(physical=0)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    _run_scavenger(
        fake, monkeypatch,
        apps="GPU-aaaa, 500, 25000\nGPU-aaaa, 900, 1000", calls=calls, polls=8)

    assert slot["parked"] is False
    assert _ckpt_actions(calls)[:2] == ["lock", "checkpoint"]
    assert any("did not park within" in m for m in fake.messages)
    assert slot["park"].value == 0, "park request must be withdrawn on fallback"


def test_parked_worker_resumes_without_restore(monkeypatch):
    slot = _park_slot(physical=0)
    slot["parked"] = True
    slot["park"].value = 2
    slot["hold"].value = 1
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    _run_scavenger(fake, monkeypatch, apps="", calls=calls)

    assert slot["parked"] is False
    assert slot["park"].value == 0, "worker must be told to rebuild"
    assert slot["hold"].value == 0
    assert _ckpt_actions(calls) == [], "cooperative resume needs no restore"


def test_free_streak_required_before_release(monkeypatch):
    """A GPU must be foreign-free for the full free_secs window, not one poll."""
    slot = _slot(physical=0, hold_value=1)
    fake = _FakePipe([slot], {"w0": 500})
    calls = []
    # free_secs=10s at a 0.05s poll = 200 consecutive free polls needed.
    _run_scavenger(fake, monkeypatch, apps="", calls=calls, polls=3, free_secs=10.0)

    assert slot["hold"].value == 1, "released before the free window elapsed"
