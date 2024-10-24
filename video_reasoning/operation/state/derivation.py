import logging
from typing import Optional

from api.api import API
from video_reasoning.structure.videoinsta import Clip, VideoINSTA
from video_reasoning.operation.base import Operation

logger = logging.getLogger("root")


class DeriveClipStates(Operation):
    def __init__(
            self,
            derive_action_captions: bool,
            derive_action_captions_summary: bool,
            total_num_words_action_captions_summary: int,
            min_num_words_action_captions_summary: int,
            derive_object_detections: bool,
            derive_object_detections_summary: bool,
            total_num_words_object_detections_summary: int,
            min_num_words_object_detections_summary: int,
            derive_temporal_grounding: bool,
            derive_temporal_grounding_summary: bool,
            total_num_words_temporal_grounding_summary: int,
            min_num_words_temporal_grounding_summary: int,
            normalization_video_length: int = 180
    ):
        super().__init__()

        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.total_num_words_action_captions_summary = total_num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.total_num_words_object_detections_summary = total_num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.total_num_words_temporal_grounding_summary = total_num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        self.normalization_video_length = normalization_video_length

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target=Optional[Clip]) -> None:
        # get the root clip state from the structure, because we can inherit the perceptive information
        root_state = structure.root.state

        if not root_state.derived:
            err_msg = "Root clip state has not been derived yet."
            logger.error(err_msg)
            raise ValueError(err_msg)

        derivable_clips = structure.get_derivable_clips()
        for derivable_clip in derivable_clips:

            # find start and end indices of the video clip
            total_start_frame_index = derivable_clip.state.video_clip.sampled_indices[0]
            total_end_frame_index = derivable_clip.state.video_clip.sampled_indices[-1]
            logger.debug(f"Total start frame index: {total_start_frame_index}")
            logger.debug(f"Total end frame index: {total_end_frame_index}")

            # get the list index in the sampled indices of the root clip state video
            start_list_index = root_state.video_clip.sampled_indices.index(total_start_frame_index)
            end_list_index = root_state.video_clip.sampled_indices.index(total_end_frame_index)
            logger.debug(f"Start list index: {start_list_index}")
            logger.debug(f"End list index: {end_list_index}")

            # get the length of the clip and the whole video in seconds respectively
            len_clip_sec = len(derivable_clip.state.video_clip) / derivable_clip.state.video_clip.sampled_fps
            len_video_sec = len(root_state.video_clip) / root_state.video_clip.sampled_fps
            logger.debug(f"Length of the clip in seconds: {len_clip_sec}")
            logger.debug(f"Length of the video in seconds: {len_video_sec}")

            if self.derive_action_captions:
                # inherit the action captions from the root clip
                clip_action_captions = root_state.spatial_clip_state.action_captions[
                                       start_list_index:end_list_index + 1]

                # set the action captions of the derivable clip
                derivable_clip.state.spatial_clip_state.action_captions = clip_action_captions

                logger.info(f"Inherited action captions from root clip for clip {derivable_clip}.")
                logger.debug(f"Action Captions:\n{derivable_clip.state.spatial_clip_state.action_captions}")

            if self.derive_action_captions_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )
                logger.debug(f"Number of words for action caption summary: {num_words}")

                # use action caption summary API function to summarize all action captions to get a clip summary
                derivable_clip.state.spatial_clip_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=derivable_clip.state.spatial_clip_state.action_captions,
                    object_detections=[],
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_clip.state.video_clip,
                    question=derivable_clip.state.task.question,  # will be used if it is given in the prompt template
                    words=num_words  # will be used if it is given in the prompt template
                )

                logger.info(f"Derived action caption summary for clip {derivable_clip}.")
                logger.debug(
                    f"Action Caption Summary:\n{derivable_clip.state.spatial_clip_state.action_captions_summary}")

            if self.derive_object_detections:
                # inherit the object detections from the root clip
                clip_object_detections = root_state.spatial_clip_state.object_detections[
                                         start_list_index:end_list_index + 1]

                # set the object detections of the derivable clip
                derivable_clip.state.spatial_clip_state.object_detections = clip_object_detections

                logger.info(f"Inherited object detections from root clip for clip {derivable_clip}.")
                logger.debug(f"Object Detections:\n"
                             f"{derivable_clip.state.spatial_clip_state.object_detections}")

            if self.derive_object_detections_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )
                logger.debug(f"Number of words for object detections summary: {num_words}")

                # use object detection summary API function to summarize all object detections to get a universal video summary
                derivable_clip.state.spatial_clip_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=derivable_clip.state.spatial_clip_state.get_textual_object_list(),
                    temporal_grounding=[],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_clip.state.video_clip,
                    question=derivable_clip.state.task.question,  # will be used if it is given in the prompt template
                    words=num_words  # will be used if it is given in the prompt template
                )

                logger.info(f"Derived object detection summary for clip {derivable_clip}.")
                logger.debug(f"Object Detection Summary:\n"
                             f"{derivable_clip.state.spatial_clip_state.object_detections_summary}")

            if self.derive_temporal_grounding:
                # inherit the temporal grounding from the root clip
                derivable_clip.state.temporal_clip_state.foreground_indicators = root_state.temporal_clip_state.foreground_indicators[
                                                                                 start_list_index:end_list_index + 1]
                derivable_clip.state.temporal_clip_state.relevance_indicators = root_state.temporal_clip_state.relevance_indicators[
                                                                                start_list_index:end_list_index + 1]
                derivable_clip.state.temporal_clip_state.salience_indicators = root_state.temporal_clip_state.salience_indicators[
                                                                               start_list_index:end_list_index + 1]

                logger.info(f"Inherited temporal grounding from root clip for clip {derivable_clip}.")
                logger.debug(f"Temporal Grounding:\n"
                             f"{derivable_clip.state.temporal_clip_state.get_textual_temporal_grounding()}")

            if self.derive_temporal_grounding_summary:
                num_words = self.get_num_words_for_summary(
                    len_clip_sec=len_clip_sec,
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.total_num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )
                logger.debug(f"Number of words for temporal grounding summary: {num_words}")

                # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
                self.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                    action_captions=[],
                    object_detections=[],
                    temporal_grounding=[derivable_clip.state.temporal_clip_state.get_textual_temporal_grounding()],
                    interleaved_action_captions_and_object_detections=[],
                    video_clip=derivable_clip.state.video_clip,
                    question=derivable_clip.state.task.question,  # will be used if it is given in the prompt template
                    options=derivable_clip.state.task.options,  # will be used if they are given in the prompt template
                    words=num_words  # will be used if it is given in the prompt template
                )

                logger.info(f"Derived temporal grounding summary for clip {derivable_clip}.")
                logger.debug(f"Temporal Grounding Summary:\n"
                             f"{derivable_clip.state.temporal_clip_state.temporal_grounding_summary}")

            derivable_clip.state.derived = True

            logger.info(f"Derived clip state for clip {derivable_clip}.")
            logger.debug(f"The following state has been derived:\n{derivable_clip.state}")

        logger.info(f"Executed state derivation operation: DeriveClipStates")

    def get_num_words_for_summary(
            self,
            len_clip_sec,
            len_video_sec,
            total_num_words_summary,
            min_num_words_summary
    ):
        # do not normalize the number of words for the summary if the normalization video length is -1
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
        else:
            # calculate the number of words for the summary (proportionate to the video length in seconds)
            # calculate the word contingent for the whole video
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
        # calculate the exact number of words for the clip
        num_words_exact = int(round(whole_video_word_contingent * (len_clip_sec / len_video_sec)))
        # make multiple of 10 for better LLM readability
        num_words_mul_ten = (int(round(num_words_exact / 10))) * 10
        # clip the number of words to the minimum number of words
        num_words = max(num_words_mul_ten, min_num_words_summary)
        return num_words


class DeriveRootClipState(Operation):
    def __init__(
            self,
            derive_action_captions: bool,
            derive_action_captions_summary: bool,
            num_words_action_captions_summary: int,
            min_num_words_action_captions_summary: int,
            derive_object_detections: bool,
            derive_object_detections_summary: bool,
            num_words_object_detections_summary: int,
            min_num_words_object_detections_summary: int,
            derive_temporal_grounding: bool,
            derive_temporal_grounding_summary: bool,
            num_words_temporal_grounding_summary: int,
            min_num_words_temporal_grounding_summary: int,
            normalization_video_length: int = 180
    ):
        super().__init__()

        self.derive_action_captions = derive_action_captions or derive_action_captions_summary
        self.derive_action_captions_summary = derive_action_captions_summary
        self.num_words_action_captions_summary = num_words_action_captions_summary
        self.min_num_words_action_captions_summary = min_num_words_action_captions_summary

        self.derive_object_detections = derive_object_detections or derive_object_detections_summary
        self.derive_object_detections_summary = derive_object_detections_summary
        self.num_words_object_detections_summary = num_words_object_detections_summary
        self.min_num_words_object_detections_summary = min_num_words_object_detections_summary

        self.derive_temporal_grounding = derive_temporal_grounding or derive_temporal_grounding_summary
        self.derive_temporal_grounding_summary = derive_temporal_grounding_summary
        self.num_words_temporal_grounding_summary = num_words_temporal_grounding_summary
        self.min_num_words_temporal_grounding_summary = min_num_words_temporal_grounding_summary

        self.normalization_video_length = normalization_video_length

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target=Optional[Clip]) -> None:
        root_clip = structure.root
        len_video_sec = len(root_clip.state.video_clip) / root_clip.state.video_clip.sampled_fps

        if self.derive_action_captions:
            # use action caption summary API function to summarize all action captions to get a universal video summary
            root_clip.state.spatial_clip_state.action_captions = api.get_action_captions_from_video_clip(
                video_clip=root_clip.state.video_clip
            )

            logger.info(f"Derived action captions for root clip {root_clip}.")
            logger.debug(f"Action Captions:\n{root_clip.state.spatial_clip_state.action_captions}")

        if self.derive_action_captions_summary:
            # use action caption summary API function to summarize all action captions to get a universal video summary
            root_clip.state.spatial_clip_state.action_captions_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=root_clip.state.spatial_clip_state.action_captions,
                object_detections=[],
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_clip.state.video_clip,
                question=root_clip.state.task.question,  # will be used if it is given in the prompt template
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_action_captions_summary,
                    min_num_words_summary=self.min_num_words_action_captions_summary
                )  # will be used if it is given in the prompt template
            )

            logger.info(f"Derived action caption summary for root clip {root_clip}.")
            logger.debug(f"Action Caption Summary:\n{root_clip.state.spatial_clip_state.action_captions_summary}")

        if self.derive_object_detections:
            # use object detection API function to get object detections from the video clip
            root_clip.state.spatial_clip_state.object_detections = api.get_unspecific_objects_from_video_clip(
                video_clip=root_clip.state.video_clip)

            logger.info(f"Derived object detections for root clip {root_clip}.")
            logger.debug(f"Object Detections:\n"
                         f"{root_clip.state.spatial_clip_state.object_detections}")

        if self.derive_object_detections_summary:
            # use object detection summary API function to summarize all object detections to get a universal video summary
            root_clip.state.spatial_clip_state.object_detections_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=root_clip.state.spatial_clip_state.get_textual_object_list(),
                temporal_grounding=[],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_clip.state.video_clip,
                question=root_clip.state.task.question,  # will be used if it is given in the prompt template
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_object_detections_summary,
                    min_num_words_summary=self.min_num_words_object_detections_summary
                )  # will be used if it is given in the prompt template
            )

            logger.info(f"Derived object detection summary for root clip {root_clip}.")
            logger.debug(f"Object Detection Summary:\n"
                         f"{root_clip.state.spatial_clip_state.object_detections_summary}")

        if self.derive_temporal_grounding:
            # use temporal grounding API function to get temporal grounding from the video clip and the question
            temporal_grounding = api.get_temporal_grounding_from_video_clip_and_text(
                video_clip=root_clip.state.video_clip,
                text=root_clip.state.task.question
            )

            root_clip.state.temporal_clip_state.foreground_indicators = temporal_grounding["foreground_indicators"]
            root_clip.state.temporal_clip_state.relevance_indicators = temporal_grounding["relevance_indicators"]
            root_clip.state.temporal_clip_state.salience_indicators = temporal_grounding["salience_indicators"]

            logger.info(f"Derived temporal grounding for root clip {root_clip}.")
            logger.debug(f"Temporal Grounding:\n"
                         f"{root_clip.state.temporal_clip_state.get_textual_temporal_grounding()}")

        if self.derive_temporal_grounding_summary:
            # use temporal grounding summary API function to summarize all temporal grounding to get a universal video summary
            root_clip.state.temporal_clip_state.temporal_grounding_summary = api.get_summary_from_noisy_perceptive_data(
                action_captions=[],
                object_detections=[],
                temporal_grounding=[root_clip.state.temporal_clip_state.get_textual_temporal_grounding()],
                interleaved_action_captions_and_object_detections=[],
                video_clip=root_clip.state.video_clip,
                question=root_clip.state.task.question,  # will be used if it is given in the prompt template
                options=root_clip.state.task.options,  # will be used if they are given in the prompt template
                words=self.get_num_words_for_summary(
                    len_video_sec=len_video_sec,
                    total_num_words_summary=self.num_words_temporal_grounding_summary,
                    min_num_words_summary=self.min_num_words_temporal_grounding_summary
                )  # will be used if it is given in the prompt template
            )

            logger.info(f"Derived temporal grounding summary for root clip {root_clip}.")
            logger.debug(f"Temporal Grounding Summary:\n"
                         f"{root_clip.state.temporal_clip_state.temporal_grounding_summary}")

        root_clip.state.derived = True

        logger.info(f"Derived universal clip state for root clip {root_clip}.")
        logger.debug(f"The following universal clip state has been derived:\n{root_clip.state}")
        logger.info(f"Executed root state derivation operation: DeriveUniversalState")

    def get_num_words_for_summary(
            self,
            len_video_sec,
            total_num_words_summary,
            min_num_words_summary
    ):
        # do not normalize the number of words for the summary if the normalization video length is -1
        if self.normalization_video_length == -1:
            whole_video_word_contingent = total_num_words_summary
        else:
            # calculate the number of words for the summary (proportionate to the video length in seconds)
            # calculate the word contingent for the whole video
            whole_video_word_contingent = (total_num_words_summary / self.normalization_video_length) * len_video_sec
        # make multiple of 10 for better LLM readability
        num_words_mul_ten = (int(round(whole_video_word_contingent / 10))) * 10
        # clip the number of words to the minimum number of words
        num_words = max(num_words_mul_ten, min_num_words_summary)
        return num_words
