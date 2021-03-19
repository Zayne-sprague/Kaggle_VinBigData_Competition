# inspired by detectron2, with modifications and simplifications
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/events.py
from typing import List, Optional

_EVENT_STORAGE_CONTEXTS = []

class EventStorage:

    def __init__(self):

        self.events: dict = {}
        self.ln_schema: dict = {}

        self.max_len: int = 1000

    def put_item(self, name: str, value: any) -> None:
        if name not in self.events:
            self.events[name] = []

        mx_len = self.ln_schema.get(name) or self.max_len

        self.events[name].append(value)

        if len(self.events[name]) > mx_len:
            self.events[name] = self.events[name][1:]

    def find_item(self, name: str, k=None) -> List:
        items = self.events[name]

        if k:
            return items[-k:]
        else:
            return items[::-1]

    def __enter__(self):
        _EVENT_STORAGE_CONTEXTS.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _EVENT_STORAGE_CONTEXTS[-1] == self, "Event Storage Stack out of order!"
        _EVENT_STORAGE_CONTEXTS.pop()


def get_event_storage_context() -> EventStorage:
    return _EVENT_STORAGE_CONTEXTS[-1]
