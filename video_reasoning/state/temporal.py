from __future__ import annotations

import logging
import torch

from video_reasoning.state.base import BaseState
from video_reasoning.state.task import Task
from video_reasoning.state.video import VideoClip

logger = logging.getLogger("root")


class TemporalClipState(BaseState):
    def __init__(
            self,
            video_clip: VideoClip,
            task: Task,
            lexical_representation: str,
            use_foreground: bool,
            use_relevance: bool,
            use_salience: bool,
            use_temporal_grounding_summary: bool,
    ):
        super().__init__(video_clip=video_clip, task=task)

        self.lexical_representation = lexical_representation

        self.foreground_indicators: list[float] = []
        self.use_foreground = use_foreground

        self.relevance_indicators: list[int] = []
        self.use_relevance = use_relevance

        self.salience_indicators: list[int] = []
        self.use_salience = use_salience

        self.temporal_grounding_summary = None
        self.use_temporal_grounding_summary = use_temporal_grounding_summary

        if self.use_temporal_grounding_summary and not self.use_foreground and not self.use_relevance and not self.use_salience:
            logger.warning("Temporal grounding summary is enabled but no temporal "
                           "grounding is enabled. Enabling all temporal groundings.")
            self.use_foreground = True
            self.use_relevance = True
            self.use_salience = True

        logger.info("Initialized temporal clip state")

    def get_lexical_representation(self) -> str:
        if not self.use_foreground and not self.use_relevance and not self.use_salience and not self.use_temporal_grounding_summary:
            return ""

        temporal_grounding_text = None
        if self.use_foreground or self.use_relevance or self.use_salience:
            logger.debug("Using temporal grounding in lexical representation.")
            temporal_grounding_text = self.get_textual_temporal_grounding()
            if temporal_grounding_text == "":
                temporal_grounding_text = None

        temporal_grounding_summary = None
        if self.use_temporal_grounding_summary:
            temporal_grounding_summary = self.temporal_grounding_summary

        # combine all information
        all_information = [temporal_grounding_text, temporal_grounding_summary]
        logger.debug(f"Collected all temporal information: {all_information}")

        # filter out None values
        all_information = [x for x in all_information if x is not None]
        logger.debug(f"Filtered out None values: {all_information}")

        if not all_information:
            return ""

        # add level indentation
        # choose the prefix based on the lexical representation
        if self.lexical_representation == "list":
            level_indentation = "                - "
            double_point = ":"
        elif self.lexical_representation == "sections" and self.use_temporal_grounding_summary:
            level_indentation = "\n"
            double_point = ""
        elif self.lexical_representation == "sections" and not self.use_temporal_grounding_summary:
            level_indentation = "\n###### "
            double_point = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")
        all_information = [level_indentation + x for x in all_information]

        # add heading
        heading = "Temporal Information"
        heading = f"{heading}{double_point}"

        # add sub heading
        if self.use_temporal_grounding_summary:
            sub_heading = f"\n###### Temporal Grounding Summary"
        else:
            sub_heading = ""
        sub_heading = f"{sub_heading}{double_point}"
        all_information = [heading] + [sub_heading] + all_information
        logger.debug(f"Added heading: {all_information}")

        all_information = [x for x in all_information if x != ""]

        return "\n".join(all_information)

    def get_json_representation(self) -> dict:
        return {
            "foreground_indicators": self.foreground_indicators,
            "foreground_ratio": self.get_foreground_ratio(),
            "relevance_indicators": self.relevance_indicators,
            "relevance_ratio": self.get_relevance_ratio(),
            "salience_indicators": self.salience_indicators,
            "salience_ratio": self.get_salience_ratio(),
            "temporal_grounding_text": self.get_lexical_representation(),
            "temporal_grounding_summary": self.temporal_grounding_summary,
            "use_foreground": self.use_foreground,
            "use_relevance": self.use_relevance,
            "use_salience": self.use_salience,
            "use_temporal_grounding_summary": self.use_temporal_grounding_summary
        }

    def get_textual_temporal_grounding(self) -> str:
        if not self.use_foreground and not self.use_relevance and not self.use_salience:
            return ""

        # choose the delimiter based on the lexical representation
        if self.lexical_representation == "list":
            delimiter = ": "
        elif self.lexical_representation == "sections":
            delimiter = "\n\n"
        elif self.lexical_representation == "unformatted":
            delimiter = ""
        else:
            raise ValueError(f"Unknown lexical representation: {self.lexical_representation}")

        heading = "Temporal Grounding of the Question within the Clip"
        foreground_ratio_text = (f"{round(self.get_foreground_ratio() * 100)}% of the frames within the clip are "
                                 f"foreground regarding the question.") if self.use_foreground else None
        relevance_ratio_text = (f"{round(self.get_relevance_ratio() * 100)}% of the frames within the clip are within "
                                f"the most relevant time interval regarding the question.") if self.use_relevance else None
        salience_ratio_text = (f"The mean salience of the question among all frames within the clip is "
                               f"{round(self.get_salience_ratio() * 100)}%.") if self.use_salience else None

        temporal_grounding = [foreground_ratio_text,
                              relevance_ratio_text,
                              salience_ratio_text]

        temporal_grounding = [x for x in temporal_grounding if x is not None]

        lexical_representation = "\n".join(temporal_grounding)

        if self.lexical_representation == "unformatted":
            return lexical_representation
        else:
            return f"{heading}{delimiter}{lexical_representation}"

    def get_foreground_ratio(self) -> float:
        # get ratio of all foreground indicators >= 0.5
        foreground_ratio = sum([1 for indicator in self.foreground_indicators if indicator >= 0.5]) / len(
            self.foreground_indicators) if len(self.foreground_indicators) > 0 else 0.0
        return foreground_ratio

    def get_relevance_ratio(self) -> float:
        relevance_ratio = sum(self.relevance_indicators) / len(self.relevance_indicators) if len(
            self.relevance_indicators) > 0 else 0.0
        return relevance_ratio

    def get_salience_ratio(self) -> float:
        salience_ratio = sum(self.salience_indicators) / len(self.salience_indicators) if len(
            self.salience_indicators) > 0 else 0.0
        return salience_ratio

    def __str__(self):
        return f"TemporalClipState: {self.get_lexical_representation()}"
