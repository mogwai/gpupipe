"""Type definitions for the pipe module."""


class _EndSentinel:
    """Sentinel value for signaling pipeline completion.

    Root workers return End to signal they're done producing items.
    Middle workers never see End - the framework handles it internally.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "End"

    def __reduce__(self):
        return (self.__class__, ())


End = _EndSentinel()
