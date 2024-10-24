from __future__ import annotations

import logging
import torch

from video_reasoning.state.base import BaseState
from video_reasoning.state.task import Task
from video_reasoning.state.spatial import SpatialClipState
from video_reasoning.state.temporal import TemporalClipState
from video_reasoning.state.video import VideoClip

logger = logging.getLogger("root")


class ClipState(BaseState):

    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            spatial_clip_state: SpatialClipState,
            temporal_clip_state: TemporalClipState,
            lexical_representation: str
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.spatial_clip_state = spatial_clip_state
        self.temporal_clip_state = temporal_clip_state
        self.lexical_representation = lexical_representation

        self.ranking = []
        self.ranking_confidence = None
        self.clip_state_summary = None
        self.waiting = False
        self.answerable = False

        logger.info(f"Initialized clip state")

    def get_lexical_representation(self) -> str:
        # choose the prefix based on the lexical representation
        if self.lexical_representation == "list":
            prefix = "        -"
            level_indentation = "            - "
            delimiter = ":"
        elif self.lexical_representation == "sections":
            prefix = "\n####"
            level_indentation = "\n##### "
            delimiter = ""
        elif self.lexical_representation == "unformatted":
            prefix = ""
            level_indentation = ""
            delimiter = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        start = round(self.video_clip.sampled_indices[0] / self.video_clip.original_fps)
        end = round(self.video_clip.sampled_indices[-1] / self.video_clip.original_fps)
        heading = f"{prefix} Information about the clip in the interval [{start}, {end}]{delimiter}"

        # collect all information
        all_information = [
            self.spatial_clip_state.get_lexical_representation(),
            self.temporal_clip_state.get_lexical_representation()
        ]

        # filter out empty strings
        all_information = [x for x in all_information if x != ""]

        # add level indentation
        all_information = [level_indentation + x for x in all_information]

        # add heading if not unformatted
        if self.lexical_representation != "unformatted":
            all_information.insert(0, heading)

        return "\n".join(all_information)

    def get_json_representation(self) -> dict:
        return {
            "video_clip": self.video_clip.get_json_representation(),
            "task": self.task.get_json_representation(),
            "spatial_clip_state": self.spatial_clip_state.get_json_representation(),
            "temporal_clip_state": self.temporal_clip_state.get_json_representation(),
            "lexical_representation": self.lexical_representation,
            "ranking": self.ranking,
            "ranking_confidence": self.ranking_confidence,
            "clip_state_summary": self.clip_state_summary,
            "waiting": self.waiting,
            "answerable": self.answerable
        }

    def merge(self, other: ClipState) -> ClipState:
        new_video_clip = self.video_clip.get_merged_video_clip(other.video_clip)
        logger.info("Merged clip states")

        return ClipState(
            video_clip=new_video_clip,
            task=self.task,
            spatial_clip_state=self.spatial_clip_state,
            temporal_clip_state=self.temporal_clip_state,
            lexical_representation=self.lexical_representation
        )

    def __str__(self):
        return f"ClipState: {self.get_lexical_representation()}"
