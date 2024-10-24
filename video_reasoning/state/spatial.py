from __future__ import annotations

import logging
import torch

from video_reasoning.state.base import BaseState
from video_reasoning.state.task import Task
from video_reasoning.state.video import VideoClip

logger = logging.getLogger("root")


class SpatialClipState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_action_captions: bool,
            use_object_detections: bool,
            use_action_captions_summary: bool,
            use_object_detections_summary: bool
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation = lexical_representation

        self.action_captions: list[str] = []
        self.action_captions_summary: str = ""
        self.use_action_captions = use_action_captions
        self.use_action_captions_summary = use_action_captions_summary

        self.object_detections: list[list[str]] = []
        self.object_detections_summary: str = ""
        self.use_object_detections = use_object_detections
        self.use_object_detections_summary = use_object_detections_summary

        logger.info("Initialized spatial clip state")

    def get_lexical_representation(self) -> str:
        if not self.use_action_captions and not self.use_action_captions_summary and \
                not self.use_object_detections and not self.use_object_detections_summary:
            return ""

        # choose the delimiter based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = " "
            double_point = ":"
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
            double_point = ""
        elif self.lexical_representation == "unformatted":
            delimiter = ""
            double_point = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = f"Spatial Information{double_point}"

        action_captions = None
        if self.use_action_captions:
            action_captions = [caption[:-1] if caption[-1] == "." else caption for caption in self.action_captions]
            action_captions = '. '.join(action_captions)
            if self.lexical_representation != "unformatted":
                action_captions = f"Action Captions{double_point}{delimiter}{action_captions}." if action_captions != "" else None

        action_captions_summary = None
        if self.use_action_captions_summary:
            # replace "video" with "clip"
            action_captions_summary = self.action_captions_summary.replace("video", "clip")
            if self.lexical_representation != "unformatted":
                action_captions_summary = f"Action Caption Summary{double_point}{delimiter}{action_captions_summary}" if action_captions_summary != "" else None

        object_detections = None
        if self.use_object_detections:
            # delimit the intervals by dots
            unspecific_object_detections_text = '. '.join(self.get_textual_object_list())
            if self.lexical_representation != "unformatted":
                object_detections = f"Object Detections{double_point}{delimiter}{unspecific_object_detections_text}." if unspecific_object_detections_text != "" else None

        object_detections_summary = None
        if self.use_object_detections_summary:
            # replace "video" with "clip"
            object_detections_summary = self.object_detections_summary.replace("video", "clip")
            if self.lexical_representation != "unformatted":
                object_detections_summary = f"Object Detections Summary{double_point}{delimiter}{object_detections_summary}" if object_detections_summary != "" else None

        # concatenate all information
        all_information = [
            action_captions,
            action_captions_summary,
            object_detections,
            object_detections_summary
        ]

        # filter out Nones and empty strings
        all_information = [info for info in all_information if info is not None]
        all_information = [info for info in all_information if info != ""]

        if not all_information:
            return ""

        # choose the level indentation based on the lexical representation
        if self.lexical_representation == "list":
            level_indentation = "                - "
        elif self.lexical_representation == "sections":
            level_indentation = "\n###### "
        elif self.lexical_representation == "unformatted":
            level_indentation = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        # add level indentation
        all_information = [level_indentation + info for info in all_information]

        # add heading
        if self.lexical_representation != "unformatted":
            all_information.insert(0, heading)

        # join by linebreaks
        lexical_representation = "\n".join(all_information)

        return lexical_representation

    def get_json_representation(self) -> dict:
        return {
            "action_captions": self.action_captions,
            "object_detections": self.object_detections,
            "action_captions_summary": self.action_captions_summary,
            "object_detections_summary": self.object_detections_summary,
            "use_action_captions": self.use_action_captions,
            "use_object_detections": self.use_object_detections,
            "use_action_caption_summary": self.use_action_captions_summary,
            "use_object_detections_summary": self.use_object_detections_summary
        }

    def __str__(self):
        return f"SpatialClipState: {self.get_lexical_representation()}"

    def get_textual_object_list(self) -> list[str]:
        # make first character of each object uppercase (looks better, have not ablated on that)
        unspecific_object_detections = [[obj[0].upper() + obj[1:] for obj in objects] for objects in
                                        self.object_detections]

        # delimit the objects of an interval by semicolons
        unspecific_object_detections_interval_texts = ['; '.join(objects) for objects in
                                                       unspecific_object_detections]

        # remove dots
        unspecific_object_detections_interval_texts = [text.replace(".", "") for text in
                                                       unspecific_object_detections_interval_texts]

        return unspecific_object_detections_interval_texts
