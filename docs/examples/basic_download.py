"""Basic IO pipeline: query DB -> download files -> process -> update DB.

Pattern: RetrieveSQL source, threaded downloaders, single processor.
From: data/jobs/download_to_ogg.py, data/jobs/convert_opus.py
"""

from pipe import Pipe, RetrieveSQL

QUERY = """
SELECT id, audio_url, duration FROM items
WHERE processed IS NULL
ORDER BY id
"""


class Downloader:
    def __init__(self):
        self._store = None

    def load(self):
        from obstore.store import S3Store
        self._store = S3Store(bucket="my-bucket", aws_region="auto")

    def __call__(self, item):
        import obstore as obs
        data = obs.get(self._store, item["audio_url"])
        item["audio_bytes"] = bytes(data.bytes())
        return item


def process(item):
    audio_bytes = item.pop("audio_bytes")
    # do something with the audio
    item["size"] = len(audio_bytes)
    return item


class DBUpdater:
    def __init__(self):
        self.conn = None

    def load(self):
        import psycopg
        self.conn = psycopg.connect(autocommit=True)

    def __call__(self, item):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE items SET processed = true WHERE id = %s",
                (item["id"],),
            )
        return item


if __name__ == "__main__":
    pipe = Pipe(debug=False, stats_interval=10)
    pipe.add(RetrieveSQL(QUERY, chunk_size=50), outqn=100)
    pipe.add(Downloader(), workers=8, thread=True, outqn=200)
    pipe.add(process, workers=2, outqn=50)
    pipe.add(DBUpdater(), workers=1, outqn=0)

    for item in pipe:
        pass
