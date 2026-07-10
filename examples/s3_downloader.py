"""Streaming S3 downloader (obstore + AsyncPoolWorker).

The canonical download stage, distilled from the real data/fluac/akro jobs.
One worker keeps up to `max_concurrent` GETs in flight continuously and emits
each item the instant its download lands — no batch barrier, so one slow or
retrying object never stalls the rest and the downstream GPU stage stays fed.

(The alternative for simpler cases: a plain `__call__(self, batch)` worker with
`batch=64` + asyncio.gather — fine when per-item latency is uniform.)

Env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL (e.g. R2).
"""
import os

from pipe import AsyncPoolWorker, Pipe


def s3_key(path: str) -> str:
    """s3://bucket/some/key -> some/key"""
    return path.split("/", 3)[3]


class S3Downloader(AsyncPoolWorker):
    def __init__(self, bucket: str = "my-bucket", max_concurrent: int = 256):
        super().__init__(max_concurrent=max_concurrent)
        self.bucket = bucket  # __init__ must stay picklable: config only

    def load(self):
        # Heavy/unpicklable state goes here (runs inside the worker process)
        from obstore.store import S3Store

        self._store = S3Store(
            bucket=self.bucket,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_endpoint=os.environ.get("AWS_ENDPOINT_URL"),
            aws_region="auto",
        )

    async def process(self, item):
        import obstore as obs

        try:
            result = await obs.get_async(self._store, s3_key(item["s3_path"]))
            item["data"] = bytes(result.bytes())
            return item
        except Exception as e:
            print(f"download failed for {item['s3_path']}: {e}")
            return None  # drop; return the item into a retry field if you need retries


class KeySource:
    """Toy root worker; in real jobs this is a DB query (see RetrieveSQL)."""

    def __call__(self):
        for i in range(1000):
            yield {"id": i, "s3_path": f"s3://my-bucket/audio/{i}.flac"}


class Sink:
    def __call__(self, item):
        return {"id": item["id"], "size": len(item["data"])}


if __name__ == "__main__":
    pipe = Pipe(stats_interval=3)
    pipe.add(KeySource(), outqn=500)
    pipe.add(S3Downloader(bucket="my-bucket", max_concurrent=256), workers=1, outqn=200)
    pipe.add(Sink(), workers=1, outqn=0)

    for result in pipe:
        pass
