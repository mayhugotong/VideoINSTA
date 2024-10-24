from __future__ import annotations

import logging
import torch

from abc import ABC, abstractmethod

from video_reasoning.state.task import Task
from video_reasoning.state.video import VideoClip

logger = logging.getLogger("root")


class BaseState(ABC):

    def __init__(self, video_clip: VideoClip, task: Task):
        self.video_clip: VideoClip = video_clip
        self.task: Task = task
        self.derived = False

    @abstractmethod
    def get_lexical_representation(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_json_representation(self) -> dict:
        raise NotImplementedError
