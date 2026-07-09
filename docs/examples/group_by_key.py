"""Group streaming items by key before batch processing — runnable as-is.

The basics of the speaker-grouping stage in the real training pipelines:
items arrive interleaved (many speakers mixed together), but the model wants
all of one speaker's clips together. A stateful worker buffers per key and
emits a group once it's complete.

The two rules for stateful stages:
- workers=1 — state lives in the worker instance; two workers would each see
  half the stream and neither would ever complete a group
- flush() — the framework calls it at shutdown; yield the incomplete groups
  so the tail of the stream isn't silently dropped

Run: python group_by_key.py
"""
import random

from pipe import Pipe


class ClipSource:
    """Interleaved clips from 8 speakers, 6 clips each."""

    def __call__(self):
        clips = [
            {"speaker": f"spk{s}", "clip": c}
            for s in range(8)
            for c in range(6)
        ]
        random.shuffle(clips)
        yield from clips


class GroupBySpeaker:
    """Buffer clips per speaker; emit the group when it reaches group_size."""

    def __init__(self, group_size: int = 6):
        self.group_size = group_size
        self.groups = {}

    def __call__(self, item):
        key = item["speaker"]
        self.groups.setdefault(key, []).append(item)
        if len(self.groups[key]) >= self.group_size:
            return {"speaker": key, "clips": self.groups.pop(key)}
        return None                       # group not complete yet -> emit nothing

    def flush(self):
        # End of stream: emit whatever is still buffered, even if incomplete.
        for key, clips in self.groups.items():
            yield {"speaker": key, "clips": clips, "partial": True}
        self.groups = {}


class ProcessGroup:
    def __call__(self, group):
        return {
            "speaker": group["speaker"],
            "n_clips": len(group["clips"]),
            "partial": group.get("partial", False),
        }


if __name__ == "__main__":
    pipe = Pipe(stats_interval=0)
    pipe.add(ClipSource(), outqn=100)
    pipe.add(GroupBySpeaker(group_size=6), workers=1, outqn=50)  # stateful -> workers=1
    pipe.add(ProcessGroup(), workers=2, outqn=0)

    for result in pipe:
        print(result)
