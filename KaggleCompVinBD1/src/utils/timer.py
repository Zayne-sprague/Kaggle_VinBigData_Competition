import time
from typing import Optional


class Timer:

    def __init__(self):
        self.start_time: Optional[float] = None

    def start(self) -> None:
        assert self.start_time is None, "timer is already running"
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        assert self.start_time is not None, "timer isn't running"
        elapsed: float = time.perf_counter() - self.start_time

        self.start_time = None
        return elapsed
