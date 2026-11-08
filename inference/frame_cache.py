from __future__ import annotations
from enum import IntEnum
from pprint import pprint
from queue import Queue
import time

import torch
from torch import Tensor
from hytils import blue, blue, red

from denc import (
    BaseThread,
    NnFrame,
    wait_frame_ready,
    mark_frame_ready,
)
from nnarchs import (
    nnlib,
    Hdtype,
    PyTorchModel,
    PyTorchSession,
)
from gpu_algos import (
    gpu_resize_to_,
    gpu_resize_,
)



class _CACHE_STATE(IntEnum):
    INIT = 0x00
    WAIT_FRAMES = 0x01
    PROCESSING = 0x02
    EMPTYING = 0x03
    EMPTY = 0x04




class TemporalFrameCache():

    def __init__(self, window_size: int = 7) -> None:
        self.window_size: int = window_size
        self.frames: list[NnFrame] = []
        self.tensors: list[Tensor] = []
        self.state: _CACHE_STATE = _CACHE_STATE.INIT


    def reset(self) -> None:
        self.frames.clear()
        self.tensors.clear()
        self.state = _CACHE_STATE.INIT


    def emptying(self) -> bool:
        return bool(self.state == _CACHE_STATE.EMPTYING)


    def is_ready(self) -> bool:
        return bool(
            self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING)
        )


    def is_empty(self) -> bool:
        return bool(
            self.state in (_CACHE_STATE.EMPTY, _CACHE_STATE.INIT)
        )



    def append(self, frame: NnFrame | None) -> None:
        if frame is None:
            return

        if self.state == _CACHE_STATE.INIT:
            self.tensors = [frame.tensor] * (self.window_size // 2 + 1)
            self.frames = [frame]
            self.state = _CACHE_STATE.WAIT_FRAMES

        elif self.state == _CACHE_STATE.WAIT_FRAMES:
            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            if len(self.tensors) >= self.window_size:
                self.state = _CACHE_STATE.PROCESSING

        elif self.state == _CACHE_STATE.PROCESSING:
            if len(self.frames) >= self.window_size:
                raise ValueError("Too many frames in cache")
            if len(self.tensors) >= self.window_size:
                raise ValueError("too many tensors")

            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            if frame.last:
                self.state = _CACHE_STATE.EMPTYING

        else:
            raise ValueError(red("cache: append while not ready to receive"))


    def get_window(self) -> list[Tensor] | None:
        if self.state == _CACHE_STATE.PROCESSING:
            return self.tensors

        elif self.state == _CACHE_STATE.EMPTYING:
            to_add: int = (self.window_size - len(self.tensors))
            for _ in range(to_add):
                self.tensors.append(self.tensors[-1])
            return self.tensors

        return None


    def current_frame(self) -> NnFrame | None:
        if _CACHE_STATE.PROCESSING <= self.state < _CACHE_STATE.EMPTY:
            frame: NnFrame = self.frames.pop(0)
            frame.tensor = None
            self.tensors.pop(0)
            if len(self.frames) == 0:
                self.state = _CACHE_STATE.EMPTY
            return frame
        return None
