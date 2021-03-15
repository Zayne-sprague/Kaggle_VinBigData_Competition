# inspired by detectron2, with modifications and simplifications
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/events.py
from typing import List, Optional

_EVENT_STORAGE_CONTEXTS = []

class EventStorage:

    def __init__(self):

        self.events = []

    def put_scalar(self, name: str, value: any) -> None:
        self.events.append([name, value])

    def find_scalar(self, name: str) -> List:
        items = [x for x in self.events if x[0] == name]
        return items

    def __enter__(self):
        _EVENT_STORAGE_CONTEXTS.append(self)
        return self

    def __exit__(self):
        assert _EVENT_STORAGE_CONTEXTS[-1] == self, "Event Storage Stack out of order!"
        _EVENT_STORAGE_CONTEXTS.pop()