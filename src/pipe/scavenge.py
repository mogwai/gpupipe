"""GPU scavenging — run GPU stages opportunistically on idle GPUs and yield
them, fast, when another process claims one.

Opt in per stage: `pipe.add(stage, pergpu=True, scavenge=True)`.

Mechanism (workers are never killed; nothing in flight is lost):

- start() spawns every GPU worker as usual, but a worker whose GPU already has
  a foreign compute process starts HELD: it skips load(), holds no VRAM, and
  waits for the scavenger to release it.
- A daemon thread in the parent polls `nvidia-smi` every `scavenge_poll`s.
  A foreign compute PID appearing on a GPU we occupy => the worker process is
  frozen in place with `cuda-checkpoint`: lock (bounded wait for in-flight GPU
  work to drain) + checkpoint (all device state, including the CUDA context,
  is evicted to host RAM). ~10s for a 20 GB worker on a 5090. The process is
  deliberately NOT SIGSTOPped — it simply blocks at its next CUDA call, so
  queue locks can never be wedged and the health monitor still sees it alive.
- When a claimed GPU has had no foreign compute PIDs for `scavenge_free_secs`,
  the worker is thawed (restore + unlock) or, if it never loaded, released to
  run load(). A restore that runs out of device memory is UNRECOVERABLE
  (strands a partial allocation, wedges the process — measured on r610), so
  free VRAM is checked against the recorded footprint before every restore;
  on refusal the worker stays parked and is retried next cycle.

End-of-stream: a held (never-loaded) worker exits once upstream is done and
the stage's input queue is drained, so End sentinels and stage counters behave
exactly as for normal workers. A FROZEN worker holding in-flight items keeps
the pipeline waiting until its GPU frees — a scavenger has no way to finish
that work without a GPU, and killing it would lose the items.

Requires the `cuda-checkpoint` binary matching the installed driver (r570+).
See `cryo doctor` (dotfiles) for install/version checks. Host RAM must fit the
frozen workers' GPU footprint (checked, with a loud warning, before freezing).
"""

import os
import subprocess
import time

from .utils import _log

# Wall-clock cap per cuda-checkpoint action. Checkpoint moves VRAM->host RAM at
# a few GB/s, so a full 32 GB worker is well under this.
ACTION_TIMEOUT_S = float(os.environ.get("PIPE_SCAVENGE_ACTION_TIMEOUT", "180"))
# How long `--action lock` waits for already-submitted GPU work to drain.
LOCK_TIMEOUT_MS = int(os.environ.get("PIPE_SCAVENGE_LOCK_TIMEOUT_MS", "30000"))
# Safety margin required on top of the recorded footprint before a restore.
RESTORE_MARGIN_MIB = 256
# How long to wait for a stage's on_park() to release the GPU before falling
# back to a checkpoint. Generous: the worker only parks between items, so this
# covers finishing whatever it is holding (a full decode batch, say).
PARK_ACK_TIMEOUT_S = float(os.environ.get("PIPE_SCAVENGE_PARK_TIMEOUT", "60"))
# VRAM a parked worker may still hold and still count as released. Below this,
# scavenging stops at the (free) cooperative park; above it, the slim remainder
# is also checkpointed so the GPU is handed over completely. That last step
# costs a few seconds, so raise this to trade a fully-clean GPU for a faster
# handover (a bare CUDA context is ~500 MB, torch's is ~1.2 GB).
PARK_RESIDUAL_MIB = int(os.environ.get("PIPE_SCAVENGE_PARK_RESIDUAL_MIB", "600"))


def _find_cuda_checkpoint():
    """Locate a cuda-checkpoint binary (must match the driver version — see
    `cryo doctor`). Returns None when unavailable (scavenging degrades to
    hold-at-start only, with a loud warning on the first freeze attempt)."""
    env = os.environ.get("PIPE_CUDA_CHECKPOINT")
    if env and os.access(env, os.X_OK):
        return env
    for cand in (
        "/usr/local/bin/cuda-checkpoint",
        os.path.expanduser("~/.local/bin/cuda-checkpoint"),
    ):
        if os.access(cand, os.X_OK):
            return cand
    from shutil import which

    return which("cuda-checkpoint")


def _run(cmd, timeout=ACTION_TIMEOUT_S):
    """Run a command, return (returncode, combined output). Never raises."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001 - monitor thread must never die
        return -1, str(e)


def _nvidia_smi_csv(query_flag, fields):
    rc, out = _run(
        ["nvidia-smi", query_flag + "=" + fields, "--format=csv,noheader,nounits"],
        timeout=20,
    )
    if rc != 0:
        return None
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0]:
            rows.append(parts)
    return rows


_uuid_to_index_cache = None


def _uuid_to_index():
    """Physical GPU uuid -> nvidia-smi index. Static per boot; cached."""
    global _uuid_to_index_cache
    if _uuid_to_index_cache is None:
        rows = _nvidia_smi_csv("--query-gpu", "index,uuid")
        if rows is None:
            return None
        _uuid_to_index_cache = {uuid: int(idx) for idx, uuid in rows}
    return _uuid_to_index_cache


def _gpu_compute_apps():
    """[(physical_gpu_index, pid, used_mib)] for every compute process, or
    None when nvidia-smi is unavailable (callers must treat that as 'unknown',
    not 'idle')."""
    uuid_map = _uuid_to_index()
    if uuid_map is None:
        return None
    rows = _nvidia_smi_csv("--query-compute-apps", "gpu_uuid,pid,used_memory")
    if rows is None:
        return None
    apps = []
    for row in rows:
        if len(row) < 3:
            continue
        uuid, pid, mem = row[0], row[1], row[2]
        idx = uuid_map.get(uuid)
        if idx is None:
            continue
        try:
            apps.append((idx, int(pid), int(mem)))
        except ValueError:
            continue
    return apps


def busy_physical_gpus(our_pids):
    """Physical GPU indices with at least one foreign compute process. Returns
    an empty set when nvidia-smi is unavailable (assume idle — matches the
    non-scavenge behaviour of just taking the GPU)."""
    apps = _gpu_compute_apps()
    if apps is None:
        return set()
    return {idx for idx, pid, _ in apps if pid not in our_pids}


def physical_gpu(logical_id):
    """Map a stage's LOGICAL gpu id to the PHYSICAL nvidia-smi index, honouring
    any CUDA_VISIBLE_DEVICES the parent was launched with (same convention as
    workers._resolve_physical_gpu). CVD entries may be indices or uuids."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return logical_id
    visible = [v.strip() for v in cvd.split(",") if v.strip()]
    if not (0 <= logical_id < len(visible)):
        return logical_id
    entry = visible[logical_id]
    try:
        return int(entry)
    except ValueError:
        uuid_map = _uuid_to_index() or {}
        for uuid, idx in uuid_map.items():
            if uuid.startswith(entry) or entry.startswith(uuid):
                return idx
        return logical_id


def _host_available_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _gpu_free_mib(physical_idx):
    rows = _nvidia_smi_csv("--query-gpu", "index,memory.free")
    if rows is None:
        return None
    for idx, free in rows:
        if int(idx) == physical_idx:
            return int(free)
    return None


def _pid_gpu_mib(pid):
    """Current GPU memory held by a pid, right now. Returns None if unknown."""
    apps = _gpu_compute_apps()
    if apps is None:
        return None
    return sum(m for _, p, m in apps if p == pid)


def _park_worker(slot, pid, say):
    """Ask the stage to release its GPU memory itself (no checkpoint copy).

    Returns True once the worker has acked AND nvidia-smi confirms the memory
    is actually gone. False means fall back to checkpointing."""
    park = slot["park"]
    park.value = 1
    t0 = time.time()
    while park.value != 2:
        if park.value == 3:
            # The stage could not let go without dropping work — checkpoint it
            # instead (preserves everything, just slower).
            say(
                f"scavenge: {slot['worker_id']} refused to park "
                f"(would have lost work) — checkpointing instead"
            )
            park.value = 0
            return False
        if time.time() - t0 > PARK_ACK_TIMEOUT_S:
            say(
                f"scavenge: {slot['worker_id']} did not park within "
                f"{PARK_ACK_TIMEOUT_S:.0f}s — falling back to checkpoint"
            )
            park.value = 0
            return False
        time.sleep(0.1)

    still = _pid_gpu_mib(pid) or 0
    if still > PARK_RESIDUAL_MIB:
        # The stage let go of what it could; a bare CUDA context can only be
        # reclaimed by checkpointing. Stay parked (park stays acked) so the
        # checkpoint is of the SLIM process — a few hundred MB rather than the
        # full footprint. The thaw path must clear `park` as well as restoring.
        say(
            f"scavenge: {slot['worker_id']} released down to {still} MiB "
            f"— checkpointing the remainder"
        )
        return False
    say(
        f"scavenge: parked {slot['worker_id']} in {time.time() - t0:.1f}s "
        f"(released without a checkpoint copy; {still} MiB residual)"
    )
    return True


def _freeze_worker(ckpt_bin, pid, used_mib, say):
    """lock + checkpoint (NO SIGSTOP — the process blocks at its next CUDA
    call instead, which cannot wedge a queue lock). Returns True on success;
    on failure rolls the lock back and leaves the worker running."""
    avail = _host_available_mib()
    if avail is not None and used_mib and avail < used_mib:
        say(
            f"scavenge: NOT freezing pid {pid} — needs {used_mib} MiB host RAM "
            f"for its GPU state but only {avail} MiB available"
        )
        return False
    rc, out = _run(
        [ckpt_bin, "--action", "lock", "--pid", str(pid),
         "--timeout", str(LOCK_TIMEOUT_MS)]
    )
    if rc != 0:
        say(f"scavenge: lock failed on pid {pid} ({out}) — will retry")
        _run([ckpt_bin, "--action", "unlock", "--pid", str(pid)])
        return False
    rc, out = _run([ckpt_bin, "--action", "checkpoint", "--pid", str(pid)])
    if rc != 0:
        say(f"scavenge: checkpoint failed on pid {pid} ({out}) — unlocking")
        _run([ckpt_bin, "--action", "unlock", "--pid", str(pid)])
        return False
    return True


def _thaw_worker(ckpt_bin, pid, physical_idx, frozen_mib, say):
    """restore + unlock, with the mandatory free-VRAM preflight (a restore
    that hits OOM strands its partial allocation and wedges the process
    permanently). Returns True when the worker is running again."""
    if frozen_mib:
        free = _gpu_free_mib(physical_idx)
        if free is not None and free < frozen_mib + RESTORE_MARGIN_MIB:
            say(
                f"scavenge: GPU {physical_idx} has {free} MiB free but pid {pid} "
                f"needs {frozen_mib} MiB — staying parked"
            )
            return False
    rc, out = _run([ckpt_bin, "--action", "restore", "--pid", str(pid)])
    if rc != 0:
        say(f"scavenge: RESTORE FAILED on pid {pid} ({out}) — worker stays parked")
        return False
    rc, out = _run([ckpt_bin, "--action", "unlock", "--pid", str(pid)])
    if rc != 0:
        say(f"scavenge: unlock failed on pid {pid} ({out})")
        return False
    return True


def _scavenger_thread(pipe_instance, stop_event, poll, free_secs):
    """Parent-side monitor driving hold/freeze/thaw for every scavenge slot.

    Slot dicts (built in lifecycle.start()):
      {stage_name, worker_id, gpu (logical), physical (int),
       hold (mp.Value — gates the worker's initial load()),
       frozen (bool), frozen_mib (int)}
    """
    say = pipe_instance.print
    need_free_polls = max(1, int(round(free_secs / poll)))
    free_streak = {}  # physical gpu -> consecutive foreign-free polls
    ckpt_bin = _find_cuda_checkpoint()
    warned_no_ckpt = False
    warned_no_smi = False

    slots = pipe_instance.scavenge_slots
    while not pipe_instance.should_stop.value and not stop_event.is_set():
        apps = _gpu_compute_apps()
        if apps is None:
            if not warned_no_smi:
                say("scavenge: nvidia-smi unavailable — scavenging is idle")
                warned_no_smi = True
            stop_event.wait(poll)
            continue

        # Live pid per worker_id (restart may have respawned a worker).
        pid_by_wid = {
            wid: proc.pid
            for proc, wid, _ in pipe_instance.worker_info
            if proc.is_alive()
        }
        our_pids = {os.getpid()} | set(pid_by_wid.values())
        foreign = {}  # physical -> pid list
        used_by_pid = {}
        for idx, pid, mib in apps:
            used_by_pid[pid] = used_by_pid.get(pid, 0) + mib
            if pid not in our_pids:
                foreign.setdefault(idx, []).append(pid)

        for g in {s["physical"] for s in slots}:
            free_streak[g] = 0 if g in foreign else free_streak.get(g, 0) + 1

        for slot in slots:
            g = slot["physical"]
            pid = pid_by_wid.get(slot["worker_id"])
            if pid is None:
                # Worker died; a health-monitor respawn shows up next poll.
                slot["frozen"] = False
                continue

            if g in foreign:
                if slot["frozen"] or slot.get("parked"):
                    continue
                # Cooperative release first: far cheaper than copying the
                # whole GPU footprint to host RAM.
                if slot.get("supports_park") and pid in used_by_pid:
                    say(
                        f"scavenge: GPU {g} claimed by pid(s) {foreign[g]} — "
                        f"asking {slot['worker_id']} to release "
                        f"({used_by_pid[pid]} MiB)"
                    )
                    slot["hold"].value = 1
                    if _park_worker(slot, pid, say):
                        slot["parked"] = True
                        continue
                    # Partial park: re-measure, because what's left to
                    # checkpoint is now far smaller than the pre-park figure
                    # (and frozen_mib gates the restore's free-VRAM preflight,
                    # so a stale value would refuse valid thaws).
                    fresh = _pid_gpu_mib(pid)
                    if fresh is not None:
                        used_by_pid[pid] = fresh
                if not slot["hold"].value:
                    # Hold first: if the worker hasn't created a CUDA context
                    # yet this alone parks it (it waits before load()).
                    slot["hold"].value = 1
                if pid in used_by_pid:
                    if ckpt_bin is None:
                        if not warned_no_ckpt:
                            say(
                                "scavenge: GPU claimed but no cuda-checkpoint "
                                "binary found — cannot free VRAM of running "
                                "workers (see `cryo doctor`)"
                            )
                            warned_no_ckpt = True
                        continue
                    mib = used_by_pid[pid]
                    say(
                        f"scavenge: GPU {g} claimed by pid(s) "
                        f"{foreign[g]} — freezing {slot['worker_id']} "
                        f"({mib} MiB)"
                    )
                    t0 = time.time()
                    if _freeze_worker(ckpt_bin, pid, mib, say):
                        slot["frozen"] = True
                        slot["frozen_mib"] = mib
                        say(
                            f"scavenge: froze {slot['worker_id']} in "
                            f"{time.time() - t0:.1f}s, {mib} MiB released"
                        )
            elif free_streak.get(g, 0) >= need_free_polls:
                if slot.get("parked"):
                    say(
                        f"scavenge: GPU {g} free — resuming "
                        f"{slot['worker_id']} (rebuilding GPU state)"
                    )
                    slot["park"].value = 0
                    slot["parked"] = False
                    slot["hold"].value = 0
                elif slot["frozen"]:
                    say(f"scavenge: GPU {g} free — thawing {slot['worker_id']}")
                    if _thaw_worker(
                        ckpt_bin, pid, g, slot.get("frozen_mib", 0), say
                    ):
                        slot["frozen"] = False
                        # A worker that parked AND then had its residual
                        # checkpointed is waiting on `park`, not just on the
                        # restore — release it too, or it never rebuilds.
                        park = slot.get("park")
                        if park is not None and park.value == 2:
                            park.value = 0
                        slot["hold"].value = 0
                        say(f"scavenge: thawed {slot['worker_id']}")
                elif slot["hold"].value:
                    say(
                        f"scavenge: GPU {g} free for {free_secs:.0f}s — "
                        f"releasing {slot['worker_id']} to load"
                    )
                    slot["hold"].value = 0

        stop_event.wait(poll)
    _log("Scavenger thread shutting down")
