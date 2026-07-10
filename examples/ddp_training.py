"""DDP training with ONE shared data pipeline (the fluac/vui/akro pattern).

Instead of every rank running its own copy of the data pipeline (N× DB reads,
N× S3 downloads, N× GPU preprocessing), rank 0 runs it once and all ranks
consume the SAME output queue:

    rank 0:  pipe = Pipe(expected_consumers=world_size); pipe.start()
             shared_queue = pipe.queues[-1]      # pass via mp.spawn args
    rank i:  for batch in PipeIterator(shared_queue): ...

`expected_consumers=world_size` makes the final stage emit one End sentinel per
rank, so every rank's PipeIterator terminates when the data is exhausted.
Items are distributed (each batch goes to exactly ONE rank), not broadcast.
"""
import torch
import torch.multiprocessing as mp

from pipe import Pipe, PipeIterator


class BatchSource:
    """Stand-in for the real loader (DB query -> download -> collate)."""

    def __call__(self):
        for step in range(200):
            yield {"step": step, "x": torch.randn(8, 128)}


def make_pipe(world_size: int) -> Pipe:
    pipe = Pipe(expected_consumers=world_size, stats_interval=3)
    pipe.add(BatchSource(), outqn=200)
    # real pipelines add downloader / augmentation / GPU-encode stages here
    return pipe


def train_rank(rank: int, world_size: int, shared_queue):
    # dist.init_process_group(...) etc. would go here
    steps = 0
    for batch in PipeIterator(shared_queue):
        # each batch is consumed by exactly one rank
        _ = batch["x"].sum()
        steps += 1
    print(f"rank {rank}: {steps} steps")


if __name__ == "__main__":
    world_size = 2

    pipe = make_pipe(world_size)
    pipe.start()
    shared_queue = pipe.queues[-1]

    mp.spawn(train_rank, args=(world_size, shared_queue), nprocs=world_size, join=True)
    pipe.stop()
