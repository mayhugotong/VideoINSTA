import logging

from typing import Optional
from api.api import API
from api.utils import parse_answer_json, get_single_number_candidates_from_text, replace_c_with_camera_wearer
from video_reasoning.state.task import Task
from video_reasoning.structure.videoinsta import Clip, VideoINSTA
from video_reasoning.operation.base import Operation

logger = logging.getLogger("root")


class NoRatings(Operation):
    def __init__(self):
        super().__init__()

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target: Optional[Clip]) -> None:
        unranked_clips = structure.get_unranked_clips()
        for unranked_clips in unranked_clips:
            logger.info(f"Did not rate state for unranked clip {unranked_clips}.")

        logger.info(f"Executed state rating operation: NoRating")


class AnswerabilityRating(Operation):
    def __init__(
            self,
            prompt: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ):
        super().__init__()

        self.prompt = prompt
        self.completion_start = completion_start
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.replace_c = replace_c

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target: Optional[Clip]) -> None:
        unranked_clips = structure.get_unranked_clips()
        task = structure.root.state.task

        for unranked_clip in unranked_clips:
            logger.info(f"Rating answerability of state for unrated clip {unranked_clip}.")

            # using floats since this leads to a performance increase of 2% in LLoVi, it's a blackbox xD
            video_length = round(
                unranked_clip.state.video_clip.original_num_frames / unranked_clip.state.video_clip.original_fps)
            clip_length = float(unranked_clip.state.video_clip.sampled_num_frames)

            # get the final answerability confidence, like in Video Agent, but assessing the information sufficiency
            answerability_confidence, completion = AnswerabilityRating.derive_answerability_from_clip_state(
                lexical_clip_state_representation=unranked_clip.state.get_lexical_representation(),
                video_length=video_length,
                clip_length=clip_length,
                task=task,
                api=api,
                prompt_template=self.prompt,
                completion_start=self.completion_start,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                replace_c=self.replace_c
            )
            logger.debug(f"Derived completion: {completion}")
            logger.debug(f"Derived answerability: {answerability_confidence}")

            # update ranking confidence of the clip
            unranked_clip.state.ranking_confidence = answerability_confidence

        logger.info(f"Executed state rating operation: AnswerabilityRating")

    @staticmethod
    def derive_answerability_from_clip_state(
            lexical_clip_state_representation: str,
            video_length: float,
            clip_length: float,
            task: Task,
            api: API,
            prompt_template: str,
            completion_start: str,
            max_new_tokens: int,
            temperature: float,
            replace_c: bool
    ) -> (list[str], float, str):
        logger.info("Deriving answerability from lexical clip state representation using LLM...")

        # get the task information
        question = task.question
        options = task.options

        # replace "c" with "the camera wearer" if specified
        # see https://ego4d-data.org/docs/data/annotation-guidelines/#pre-annotations-narrations
        # like https://arxiv.org/pdf/2404.04346, they also replace "C" with "the camera wearer" in the dataset
        if replace_c:
            # removed on 24.05.2024, because we already replace C beforehand
            # re-added on 29.05.2024, because we need to replace C in the data if it will not be replaced before summarization
            lexical_clip_state_representation = replace_c_with_camera_wearer(lexical_clip_state_representation)
            question = replace_c_with_camera_wearer(question)
            options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

        prompt = prompt_template.format(
            lexical_clip_state_representation=lexical_clip_state_representation,
            question=question,
            option_0=options["option 0"],
            option_1=options["option 1"],
            option_2=options["option 2"],
            option_3=options["option 3"],
            option_4=options["option 4"],
            video_length=video_length,
            clip_length=clip_length
        )

        logger.debug(f"Concatenated Prompt: {prompt}")
        logger.debug(f"Num chars of concatenated prompt: {len(prompt)}")
        logger.debug(f"Num words of concatenated prompt: {len(prompt.split())}")

        # get the final answer using the LLM
        completion = api.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        logger.debug(f"Derived llm completion of answerability: {completion}")

        # parse the answerability from the completion
        # define the priority of certain keywords
        keywords_in_priority_order = [
            'answerability',
            'answerability_confidence',
            'answerability_level',
            'confidence',
            'answerability_conf',
            'answerability confidence',
            'answerability level',
            'confidence_level',
            'confidence level'
        ]
        candidate = parse_answer_json(
            text=completion,
            keywords_in_priority_order=keywords_in_priority_order,
            candidate_fun=get_single_number_candidates_from_text
        )

        # inspiration from the Video Agent, compare https://wxh1996.github.io/VideoAgent-Website/
        # but: we do not evaluate the reasoning process, but the information sufficiency
        # if the confidence is not found, the default value will be 1 (i.e. the lowest)
        confidence = 1 if not candidate else candidate
        logger.debug(f"Parsed answerability confidence from completion: {confidence}")

        # clip the confidence to the range [1, 3]
        confidence = max(1, min(3, confidence))

        return confidence, completion
