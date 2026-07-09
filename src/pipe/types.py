"""Type definitions for the pipe module."""
import enum


class Sentinel(enum.Enum):
    """Framework control signals passed through the item queues.

    A dedicated type (not a magic string) so user data can never collide with
    a control signal. Enum members pickle by name and unpickle to the same
    object, so identity checks (`item is End`) hold across process boundaries.
    """

    END = "end"
    """Signals pipeline completion. Root workers return End when done
    producing items; middle workers never see it — the framework handles it
    internally."""

    WORKER_STOP = "worker_stop"
    """Tells exactly one worker at a stage to exit its pool after its current
    item. Only scale-down logic puts this on a queue; today that is the
    planned autoscaler (src/pipe/_planned/autoscale.py), so the handling in
    the worker run-loops is inert but safe."""

    def __repr__(self):
        # "End" / "WorkerStop", matching the exported alias names in logs
        return "".join(part.title() for part in self.name.split("_"))


End = Sentinel.END
WorkerStop = Sentinel.WORKER_STOP
