import random
import time

from .types import End


class Batcher:
    def __init__(self, size, collate_fn=None):
        self.size = size
        self.collate = collate_fn
        self.buffer = []

    def __call__(self, item):
        self.buffer.append(item)
        if len(self.buffer) >= self.size:
            return self._emit()
        return None

    def _emit(self):
        out = self.buffer
        self.buffer = []
        if self.collate is not None:
            return self.collate(out)
        return out

    def flush(self):
        if not self.buffer:
            return
        out = self._emit()
        if self.collate is not None:
            yield out
        else:
            yield from out


class BufferAndShuffle:
    def __init__(self, size, batch_size):
        self.size = size
        self.buffer = []
        self.first = True
        self.batch_size = batch_size * 4

    def __call__(self, it):
        self.buffer.append(it)
        size = self.batch_size if self.first else self.size

        if len(self.buffer) > size:
            if self.first:
                self.first = False
                print("Buffered First")
            random.shuffle(self.buffer)
            out = self.buffer
            self.buffer = []
            return out
        return None

    def flush(self):
        if self.buffer:
            random.shuffle(self.buffer)
            out = self.buffer
            self.buffer = []
            yield from out


class RetrieveSQL:
    def __init__(self, query, chunk_size=20, skip_count=False) -> None:
        self.offset = 0
        self.chunk_size = chunk_size
        self.base_query = query
        self.query = query
        self.skip_count = skip_count
        self.count = 0 if skip_count else self.get_total_count()

    def __call__(self):
        from data.db import query

        if self.skip_count:
            items = query(self.query)
            if len(items) == 0:
                return End
            return items

        if "ORDER BY" in self.query.upper():
            items = query(self.query + f" LIMIT {self.chunk_size} OFFSET {self.offset}")
        else:
            items = query(
                self.query
                + f" ORDER BY RANDOM() LIMIT {self.chunk_size} OFFSET {self.offset}"
            )

        self.offset += len(items)

        if len(items) == 0 or self.offset >= self.count:
            updated_count = self.get_total_count()
            if updated_count > self.count:
                self.count = updated_count
                self.offset = 0
                return items if len(items) > 0 else []
            return End

        return items

    def get_total_count(self):
        from data.db import query

        count_query = f"SELECT COUNT(*) as count FROM ({self.base_query}) as subquery"
        result = query(count_query)
        return result[0]["count"] if result else 0


class SQLConnection:
    def __init__(self, query=None):
        self.connection = None
        self.cursor = None
        self.query = query

    def _get_connection(self):
        import psycopg
        from data.constants import CONNECTION_STRING
        from psycopg.rows import dict_row

        connection_string = CONNECTION_STRING
        if "sslmode" not in connection_string:
            connection_string += "?sslmode=require"
        connection_string = connection_string.replace("+asyncpg", "")

        return psycopg.connect(
            connection_string, row_factory=dict_row, connect_timeout=10
        )

    def _ensure_connection(self):
        try:
            if self.connection is None or self.connection.closed:
                if self.cursor:
                    self.cursor.close()
                    self.cursor = None
                self.connection = self._get_connection()
                self.cursor = self.connection.cursor()
            elif self.cursor is None:
                self.cursor = self.connection.cursor()
        except Exception as e:
            print(f"Database connection error: {e}")
            self.connection = None
            self.cursor = None
            raise

    def execute(self, *params, query=None):
        self._ensure_connection()
        try:
            query_to_run = query or self.query
            if not query_to_run:
                raise ValueError("No query provided and no predefined query set")

            self.cursor.execute(query_to_run, params)
            self.connection.commit()
        except Exception as e:
            print(f"Database update error: {e}")
            self.connection = None
            self.cursor = None
            raise

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection and not self.connection.closed:
            self.connection.close()

    def __del__(self):
        self.close()


class RTF:
    def __init__(self) -> None:
        self.start = None
        self.total = 0

    def load(self):
        if self.start is None:
            self.start = time.perf_counter()

    def __call__(self, item):
        if self.start is None:
            self.start = time.perf_counter()
        self.total += item["duration"]
        rtf = self.total / (time.perf_counter() - self.start)
        print(f"{rtf:.2f}x")
        return item
