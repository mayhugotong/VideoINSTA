from abc import ABC
from video_reasoning.operation.base import Operation
from video_reasoning.operation.state.flag import Wait
from video_reasoning.operation.structure.split import Split
from video_reasoning.operation.structure.merge import Merge


class Decision(Operation, ABC):
    def __init__(self, split: Split, merge: Merge, wait: Wait):
        super().__init__()

        # for now, a decision can be made between a split and a wait operation
        self.split = split
        self.merge = merge
        self.wait = wait
