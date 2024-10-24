import importlib
import logging
import math

import numpy as np
import random
import torch

from api.utils import parse_list, filter_list_of_objects, get_clip_data_from_video_data, replace_c_with_camera_wearer
from datasets.utils import read_json_file
from toolbox.llm.local import HuggingFaceLLM
from toolbox.llm.remote import OpenAILLM
from toolbox.CogAgent.video import infer_transcript_from_video_clip_using_frame_captions
from toolbox.GroundingDINO.object_detector_inference import infer_bounding_boxes_from_video
from toolbox.lavila_video_captioner.narrator_inference import infer_transcript_from_video_clip_using_action_captions
from toolbox.UniVTG.temporal_grounding_inference import infer_temporal_grounding_score_from_video_and_text
from typing import Optional
from video_reasoning.state.video import VideoClip

logger = logging.getLogger("root")


class API:
    def __init__(
            self,
            get_object_detections_from_video_clip_and_text_config: dict[str, any],
            get_action_captions_from_video_clip_config: dict[str, any],
            get_temporal_grounding_from_video_clip_and_text_config: dict[str, any],
            get_summary_from_noisy_perceptive_data_config: dict[str, any],
            get_unspecific_objects_from_video_clip_config: dict[str, any],
            get_completion_from_text_config: dict[str, any],
            get_specific_objects_from_video_clip_config: dict[str, any],
            random_seed: int,
            reset_seed_for_each_function: bool = True,
            load_models_only_when_needed: bool = False
    ):
        self.get_object_detections_from_video_clip_and_text_config = get_object_detections_from_video_clip_and_text_config
        self.get_action_captions_from_video_clip_config = get_action_captions_from_video_clip_config
        self.get_temporal_grounding_from_video_clip_and_text_config = get_temporal_grounding_from_video_clip_and_text_config
        self.get_summary_from_noisy_perceptive_data_config = get_summary_from_noisy_perceptive_data_config
        self.get_completion_from_text_config = get_completion_from_text_config
        self.get_unspecific_objects_from_video_clip_config = get_unspecific_objects_from_video_clip_config
        self.get_specific_objects_from_video_clip_config = get_specific_objects_from_video_clip_config
        self.load_models_only_when_needed = load_models_only_when_needed
        self.random_seed = random_seed
        self.reset_seed_for_each_function = reset_seed_for_each_function

        self._initialize_llm()

    def _initialize_llm(self):

        available_llm_classes = {
            "HuggingFaceLLM": HuggingFaceLLM.initialize_huggingface_llm_from_config,
            "OpenAILLM": OpenAILLM.initialize_openai_llm_from_config
        }

        self.llm = available_llm_classes[
            self.get_completion_from_text_config["llm_class"]](
            self.get_completion_from_text_config
        )

        # if the models should only be loaded when needed, do not build them now
        if not self.load_models_only_when_needed:
            self.llm.build_model()

    def reset_seed(self):
        # clear caches
        torch.cuda.empty_cache()
        importlib.invalidate_caches()
        logger.info("Caches have been cleared to free up memory and ensure reproducibility and comparability.")

        if not self.reset_seed_for_each_function:
            logger.info("Random seed is not reset for each function call.")
            return

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        logger.info(f"Random seed has been reset to {self.random_seed} to ensure reproducibility and comparability.")

    def get_object_detections_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        return infer_bounding_boxes_from_video(
            video_tensor=video_clip.data,
            obj=text,
            config_file=self.get_object_detections_from_video_clip_and_text_config["config_file"],
            checkpoint_path=self.get_object_detections_from_video_clip_and_text_config["checkpoint"],
            box_threshold=self.get_object_detections_from_video_clip_and_text_config["box_threshold"],
            text_threshold=self.get_object_detections_from_video_clip_and_text_config["text_threshold"],
            cuda=self.get_object_detections_from_video_clip_and_text_config["cuda"]
        )

    def get_action_captions_from_video_clip(self, video_clip: VideoClip):
        # TODO do not use dict, only use list as return type

        # reset the seed to ensure reproducibility
        self.reset_seed()

        final_action_captions = None
        if self.get_action_captions_from_video_clip_config["model_name"] not in ["LaViLa", "CogAgent"]:
            raise ValueError(f"Model name {self.get_action_captions_from_video_clip_config['model_name']} "
                             f"is not supported for action captioning.")

        if self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"] is not None:

            # load pre-inferred action captions if a path for them is given in the config
            all_action_captions = read_json_file(
                file_path=self.get_action_captions_from_video_clip_config["pre_inferred_action_captions_path"])
            action_captions = all_action_captions[video_clip.id]

            final_action_captions = get_clip_data_from_video_data(
                video_data=action_captions,
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.original_fps
            )

            logger.warning("Using pre-inferred action captions. The pre-inferred action captions "
                           "should represent intervals of 1 second for each second of the video (180 captions for"
                           "EgoSchema).")

        elif self.get_action_captions_from_video_clip_config["model_name"] == "LaViLa":

            # infer action captions from the video clip using LaViLa
            sample_rate = self.get_action_captions_from_video_clip_config["resample_rate"]

            # get the full video tensor data
            resampled_video_clip = video_clip.get_resampled_video_clip(sample_rate=sample_rate)
            video_clip_data = resampled_video_clip.data
            start_frame = video_clip.sampled_indices[0]

            # get captions for the video_clip_data using LaViLa
            action_captions = infer_transcript_from_video_clip_using_action_captions(
                video_clip=video_clip_data,
                start_frame=start_frame,
                fps=sample_rate,
                original_fps=video_clip.original_fps,
                interval_in_seconds=self.get_action_captions_from_video_clip_config["interval_in_seconds"],
                temperature=self.get_action_captions_from_video_clip_config["temperature"],
                top_p=self.get_action_captions_from_video_clip_config["top_p"],
                max_text_length=self.get_action_captions_from_video_clip_config["max_new_tokens"],
                num_return_sequences=self.get_action_captions_from_video_clip_config["num_return_sequences"],
                early_stopping=self.get_action_captions_from_video_clip_config["early_stopping"],
                num_seg=self.get_action_captions_from_video_clip_config["num_seg"],
                cuda=True,
                modelzoo_dir_path=self.get_action_captions_from_video_clip_config["modelzoo_dir_path"],
                checkpoint_download_url=self.get_action_captions_from_video_clip_config["checkpoint_download_url"],
                checkpoint_file=self.get_action_captions_from_video_clip_config["checkpoint"]
            )

            # reduce to the very first action caption of each interval for now
            # (remark: since we use ppl for action caption selection, this is a deprecated artifact)
            final_action_captions = [action_captions[0] for action_captions in action_captions.values()]
            final_action_captions = dict(zip(action_captions.keys(), final_action_captions))

        elif self.get_action_captions_from_video_clip_config["model_name"] == "CogAgent":

            start_frame = video_clip.sampled_indices[0]

            # get captions for the video_clip_data using CogAgent
            action_captions = infer_transcript_from_video_clip_using_frame_captions(
                video_clip=video_clip.data,
                start_frame=start_frame,
                original_fps=video_clip.original_fps,
                frame_prompt=self.get_action_captions_from_video_clip_config["frame_prompt"],
                model_id=self.get_action_captions_from_video_clip_config["model_id"],
                tokenizer_id=self.get_action_captions_from_video_clip_config["tokenizer_id"],
                device=self.get_action_captions_from_video_clip_config["device"],
                precision=self.get_action_captions_from_video_clip_config["precision"],
                quantization=self.get_action_captions_from_video_clip_config["quantization"],
                temperature=self.get_action_captions_from_video_clip_config["temperature"],
                max_new_tokens=self.get_action_captions_from_video_clip_config["max_new_tokens"],
                do_sample=self.get_action_captions_from_video_clip_config["do_sample"]
            )

            # reduce to the very first action caption of each interval for now
            # (remark: since we use ppl for action caption selection, this is a deprecated artifact)
            final_action_captions = [action_captions[0] for action_captions in action_captions.values()]
            final_action_captions = dict(zip(action_captions.keys(), final_action_captions))

        assert final_action_captions is not None, "Action captions should have been inferred."
        return list(final_action_captions.values())

    def get_temporal_grounding_from_video_clip_and_text(self, video_clip: VideoClip, text: str):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        text_temporal_grounding = infer_temporal_grounding_score_from_video_and_text(
            video=video_clip.data,
            text=text,
            config_dir=self.get_temporal_grounding_from_video_clip_and_text_config["config_dir"],
            checkpoint_path=self.get_temporal_grounding_from_video_clip_and_text_config["checkpoint"],
            clip_model_version=self.get_temporal_grounding_from_video_clip_and_text_config["clip_model_version"],
            output_feat_size=self.get_temporal_grounding_from_video_clip_and_text_config["output_feat_size"],
            half_precision=self.get_temporal_grounding_from_video_clip_and_text_config["half_precision"],
            jit=self.get_temporal_grounding_from_video_clip_and_text_config["jit"],
            resize_size=self.get_temporal_grounding_from_video_clip_and_text_config["resize"],
            gpu_id=self.get_temporal_grounding_from_video_clip_and_text_config["gpu_id"]
        )

        foreground_indicators = text_temporal_grounding["foreground_indicators"]
        boundary_offsets = text_temporal_grounding["boundary_offsets"]
        saliency_scores = text_temporal_grounding["saliency_scores"].squeeze()

        # make a list of the foreground indicators
        foreground_indicators_list = [foreground_indicators[i].item() for i in range(foreground_indicators.size(0))]
        logger.debug("Derived foreground indicators")

        # derive the best boundary offset indices
        k = self.get_temporal_grounding_from_video_clip_and_text_config.get("top_k_intervals", 1)
        logger.debug(f"Deriving top {k} boundary offsets.")
        logger.debug(f"Boundary offsets: {boundary_offsets}.")
        logger.debug(f"Foreground indicators: {foreground_indicators}.")
        logger.debug(f"Flattened foreground indicators: {foreground_indicators.flatten()}.")
        _, top_k_indices = torch.topk(foreground_indicators.flatten(), k=k)
        logger.debug(f"Top {k} indices: {top_k_indices}.")
        top_k_indices = top_k_indices.tolist()
        logger.debug(f"Top {k} indices (converted to list): {top_k_indices}.")

        # initialize the relevance indicators with zeros
        relevance_indicators = [0 for _ in range(len(video_clip))]

        # iteratively update the relevance indicators for the top k intervals
        for top_i_index in top_k_indices:
            top_i_boundary_offset = boundary_offsets[top_i_index].tolist()
            logger.debug(f"Top {top_i_index} boundary offset: {top_i_boundary_offset}.")

            # optimistic flooring of start index
            start_index = max(0, top_i_index + math.floor(top_i_boundary_offset[0] * len(video_clip)))
            logger.debug(f"Start index: {start_index}.")

            # optimistic ceiling of end index
            end_index = min(top_i_index + math.ceil(top_i_boundary_offset[1] * len(video_clip)), len(video_clip) - 1)
            logger.debug(f"End index: {end_index}.")

            # update the relevance indicators
            # i.e., set all relevance indicators between start and end index to 1
            relevance_indicators = [1 if start_index <= i <= end_index else relevance_indicators[i] for i in
                                    range(len(video_clip))]
            logger.debug(f"Relevance indicators: {relevance_indicators}.")
        logger.debug(f"Derived relevance indicators: {relevance_indicators}")

        # deprecated derivation of top 1 boundary offset only
        # top_1_index = torch.argmax(foreground_indicators)
        # top_1_boundary_offset = boundary_offsets[top_1_index].tolist()
        # optimistic flooring of start index
        # top_1_start_index = max(0, top_1_index + math.floor(top_1_boundary_offset[0] * len(video_clip)))
        # optimistic ceiling of end index
        # top_1_end_index = min(top_1_index + math.ceil(top_1_boundary_offset[1] * len(video_clip)), len(video_clip) - 1)
        # set the relevance indicators to 1 for the interval, 0 for rest
        # relevance_indicators = [1 if top_1_start_index <= i <= top_1_end_index else 0 for i in range(len(video_clip))]

        salience_indicators = []
        num_saliency_scores = 1 if saliency_scores.dim() == 0 else saliency_scores.size(0)
        for i in range(num_saliency_scores):
            saliency_score = saliency_scores[i].item() if num_saliency_scores > 1 else saliency_scores.item()
            saliency_score = max(0.0, min(1.0, saliency_score))
            salience_indicators.append(saliency_score)
        logger.debug("Derived salience indicators")

        return {
            "foreground_indicators": foreground_indicators_list,
            "relevance_indicators": relevance_indicators,
            "salience_indicators": salience_indicators
        }

    def get_summary_from_noisy_perceptive_data(
            self,
            action_captions: list[str],
            object_detections: list[str],
            temporal_grounding: list[str],
            interleaved_action_captions_and_object_detections: list[str],
            video_clip: VideoClip,
            question: Optional[str] = "",
            options: Optional[dict[str, str]] = None,
            words: int = 500
    ):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        if self.get_summary_from_noisy_perceptive_data_config["pre_inferred_summaries_path"] is not None:
            logger.debug("Using pre-inferred summaries.")

            # load pre-inferred summaries json file
            summaries = read_json_file(
                file_path=self.get_summary_from_noisy_perceptive_data_config["pre_inferred_summaries_path"])

            # get the summaries entry for the video clip
            video_summaries = summaries[video_clip.id]

            # get the frame interval of the clip
            start_frame = video_clip.sampled_indices[0]
            end_frame = video_clip.sampled_indices[-1]

            # define a tolerance threshold for the frame interval (due to noise in sampling process and rounding)
            # e.g. if different GPUs were used or if there was a seeding issue
            tolerance = video_clip.original_fps // 2

            # define the whole video start and end frame
            whole_video_start_frame = 0
            whole_video_end_frame = video_clip.original_num_frames - 1

            # if the video clip covers the whole video, try to take the whole video summary first
            if start_frame == whole_video_start_frame and end_frame == whole_video_end_frame:
                try:
                    # note that a single whole video summary is deprecated, now we differentiate between the types of data
                    if len(action_captions) != 0:
                        return video_summaries["root_action_captions_summaries"]
                    elif len(object_detections) != 0:
                        return video_summaries["root_object_detections_summaries"]
                    elif len(temporal_grounding) != 0:
                        return video_summaries["root_temporal_grounding_summaries"]
                    else:
                        raise ValueError("Please provide action captions, object detections, "
                                         "or temporal grounding to summarize.")
                except KeyError:
                    logger.warning("No whole video summary found. Trying to find the summary "
                                   "for the clip as long as the video.")

            # look for the summaries of the clip in the pre-inferred summaries
            # first of all, get the index of the clip in the summaries
            index = None
            for i, clip_boundary in enumerate(video_summaries["clip_boundaries"]):
                # check if the clip boundary is within the frame interval
                if start_frame - tolerance <= clip_boundary[0] <= start_frame + tolerance and \
                        end_frame - tolerance <= clip_boundary[-1] <= end_frame + tolerance:
                    index = i
            if index is None:
                raise ValueError("The clip boundaries of the video clip are not in the pre-inferred summaries.")

            # find out what type of summary we actually want
            if len(action_captions) != 0:
                summary_type = "action_caption_summaries"
                fallback_summary_type = "n.a."
            elif len(object_detections) != 0:
                summary_type = "object_detections_summaries"
                fallback_summary_type = "unspecific_object_detection_summaries"
            elif len(temporal_grounding) != 0:
                summary_type = "temporal_grounding_summaries"
                fallback_summary_type = "n.a."
            else:
                raise ValueError(
                    "Please provide action captions, object detections, or temporal grounding to summarize.")

            # get the defined summary for the clip
            summaries = video_summaries.get(summary_type, None)

            if summaries is None:
                # if the defined summary is not available, use the fallback summary
                summaries = video_summaries.get(fallback_summary_type, None)

            summary = summaries[index]

            return summary

        # this function only supports summarization of one type of data at once
        if len(action_captions) != 0:

            assert len(object_detections) == 0, "If action captions are given, object detections should be empty."
            assert len(temporal_grounding) == 0, "If action captions are given, temporal grounding should be empty."
            assert len(interleaved_action_captions_and_object_detections) == 0, "If action captions are given, " \
                                                                                "interleaved action captions and object " \
                                                                                "detections should be empty."

            prompt_template = self.get_summary_from_noisy_perceptive_data_config[
                "action_caption_prompt_template"]
            completion_start = self.get_summary_from_noisy_perceptive_data_config[
                "action_caption_completion_start"]

            noisy_data = action_captions
            logger.debug("Using action captions for summarization.")

        elif len(object_detections) != 0:

            assert len(action_captions) == 0, "If object detections are given, action captions should be empty."
            assert len(temporal_grounding) == 0, "If object detections are given, temporal grounding should be empty."
            assert len(interleaved_action_captions_and_object_detections) == 0, "If object detections are given, " \
                                                                                "interleaved action captions and object " \
                                                                                "detections should be empty."

            prompt_template = self.get_summary_from_noisy_perceptive_data_config[
                "object_detection_prompt_template"]
            completion_start = self.get_summary_from_noisy_perceptive_data_config[
                "object_detection_completion_start"]
            noisy_data = object_detections
            logger.debug("Using object detections for summarization.")

        elif len(temporal_grounding) != 0:

            assert len(action_captions) == 0, "If temporal grounding is given, action captions should be empty."
            assert len(object_detections) == 0, "If temporal grounding is given, object detections should be empty."
            assert len(interleaved_action_captions_and_object_detections) == 0, "If temporal grounding is given, " \
                                                                                "interleaved action captions and object " \
                                                                                "detections should be empty."

            prompt_template = self.get_summary_from_noisy_perceptive_data_config[
                "temporal_grounding_prompt_template"]
            completion_start = self.get_summary_from_noisy_perceptive_data_config[
                "temporal_grounding_completion_start"]
            noisy_data = temporal_grounding
            logger.debug("Using temporal grounding for summarization.")

        elif len(interleaved_action_captions_and_object_detections) != 0:

            assert len(action_captions) == 0, "If interleaved action captions and object detections are given, " \
                                              "action captions should be empty."
            assert len(object_detections) == 0, "If interleaved action captions and object detections are given, " \
                                                "object detections should be empty."
            assert len(temporal_grounding) == 0, "If interleaved action captions and object detections are given, " \
                                                 "temporal grounding should be empty."

            prompt_template = self.get_summary_from_noisy_perceptive_data_config[
                "interleaved_action_captions_and_object_detections_prompt_template"]
            completion_start = self.get_summary_from_noisy_perceptive_data_config[
                "interleaved_action_captions_and_object_detections_completion_start"]
            noisy_data = interleaved_action_captions_and_object_detections
            logger.debug("Using interleaved action captions and object detections for summarization.")

        else:
            raise ValueError("Please provide action captions, object detections, or temporal grounding to summarize.")

        if options is None:
            options = {
                "option 0": "",
                "option 1": "",
                "option 2": "",
                "option 3": "",
                "option 4": ""
            }

        # replace c if given
        if question is not None and self.get_summary_from_noisy_perceptive_data_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)

        def chunks(data, chunk_size=5):
            for j in range(0, len(data), chunk_size):
                yield data[j:j + chunk_size]

        while len(noisy_data) != 1:
            # split current action_captions into chunks of size interval_span
            new_noisy_data = []
            caption_i = 0
            for noisy_data_chunk in chunks(noisy_data,
                                           chunk_size=self.get_summary_from_noisy_perceptive_data_config.get(
                                               "interval_span")):
                # textify the interval
                # interval_text = "\n".join(action_caption_chunk)

                logger.debug(f"Noisy Data Chunk: {noisy_data_chunk}")
                # remove dots at end of caption ends if they exist
                noisy_data_chunk = [caption[:-1] if caption[-1] == "." else caption for caption in
                                    noisy_data_chunk]
                logger.debug(f"Preprocessed Noisy Data Chunk Chunk: {noisy_data_chunk}")

                delimiter = ". " if not len(temporal_grounding) != 0 else ""
                interval_text = delimiter.join(noisy_data_chunk)
                logger.debug(f"Interval text: {interval_text}")

                prompt = prompt_template.format(
                    interval_text=interval_text,
                    question=question,
                    option_0=options["option 0"],
                    option_1=options["option 1"],
                    option_2=options["option 2"],
                    option_3=options["option 3"],
                    option_4=options["option 4"],
                    length=len(noisy_data_chunk),
                    words=words
                )
                logger.debug(f"Noisy Data Summarization Prompt: {prompt}")

                # summarize the interval
                summary = self.get_completion_from_text(
                    text=prompt,
                    completion_start=completion_start,
                    max_new_tokens=self.get_summary_from_noisy_perceptive_data_config["max_new_tokens"],
                    temperature=self.get_summary_from_noisy_perceptive_data_config["temperature"]
                )
                summary = summary.strip()

                if self.get_summary_from_noisy_perceptive_data_config["no_recursion"]:
                    # add interval like in ProViQ
                    if summary[-1] == ".":
                        summary = summary[:-1]
                    summary = f"{summary} ({caption_i * len(noisy_data_chunk)}-{(caption_i + 1) * len(noisy_data_chunk)})."
                    caption_i += 1

                # replace linebreaks with whitespaces
                if self.get_summary_from_noisy_perceptive_data_config.get("remove_linebreaks"):
                    summary = summary.replace("\n", " ")
                    # replace multi-whitespaces
                    summary = " ".join(summary.split())

                # add summary to new_action_captions
                new_noisy_data.append(summary)

            # update the action_captions(these are now the summaries from this iteration)
            noisy_data = new_noisy_data

            # break the loop if single_round is True
            if self.get_summary_from_noisy_perceptive_data_config["no_recursion"]:
                break

        if not self.get_summary_from_noisy_perceptive_data_config["no_recursion"]:
            assert len(noisy_data) == 1, "The noidy data should be summarized into one summary."
            summary = noisy_data[0]
        else:
            summary = " ".join(noisy_data)

        return summary

    def get_completion_from_text(self, text: str, completion_start: str, max_new_tokens: Optional[int],
                                 temperature: Optional[float]):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        # build the model if it should only be loaded when needed
        if self.load_models_only_when_needed:
            self.llm.build_model()

        completion = self.llm.get_completion(
            prompt=text,
            completion_start=completion_start,
            max_new_tokens=max_new_tokens if max_new_tokens else None,
            temperature=temperature if temperature else None
        )

        # destroy the model if it should only be loaded when needed (free up GPU space, avoid OOM errors on limited GPU)
        if self.load_models_only_when_needed:
            self.llm.destroy_model()

        return completion

    def get_unspecific_objects_from_video_clip(self, video_clip: VideoClip):
        # TODO do not use dict, only use list as return type

        # reset the seed to ensure reproducibility
        self.reset_seed()

        if self.get_unspecific_objects_from_video_clip_config["pre_inferred_object_detections_path"]:

            # load pre-inferred action captions if a path for them is given in the config
            object_detections = read_json_file(
                file_path=self.get_unspecific_objects_from_video_clip_config["pre_inferred_object_detections_path"])
            object_detections = object_detections[video_clip.id]

            logger.debug(f"Sampled indices: {video_clip.sampled_indices}.")
            logger.debug(f"Original FPS: {video_clip.original_fps}.")

            final_object_detections = get_clip_data_from_video_data(
                video_data=object_detections,
                sampled_indices=video_clip.sampled_indices,
                fps=video_clip.original_fps
            )

        else:
            # infer the object detections (this is time-consuming)
            start_frame = video_clip.sampled_indices[0]

            # get unspecific objects for the video_clip_data using CogAgent
            objects_per_frame = infer_transcript_from_video_clip_using_frame_captions(
                video_clip=video_clip.data,
                start_frame=start_frame,
                original_fps=video_clip.original_fps,
                frame_prompt=self.get_unspecific_objects_from_video_clip_config["frame_prompt"],
                model_id=self.get_unspecific_objects_from_video_clip_config["model_id"],
                tokenizer_id=self.get_unspecific_objects_from_video_clip_config["tokenizer_id"],
                device=self.get_unspecific_objects_from_video_clip_config["device"],
                precision=self.get_unspecific_objects_from_video_clip_config["precision"],
                quantization=self.get_unspecific_objects_from_video_clip_config["quantization"],
                temperature=self.get_unspecific_objects_from_video_clip_config["temperature"],
                max_new_tokens=self.get_unspecific_objects_from_video_clip_config["max_new_tokens"],
                do_sample=self.get_unspecific_objects_from_video_clip_config["do_sample"]
            )
            logger.debug("Inferred unspecific object detections from video clip.")

            # reduce to the very first action caption of each interval for now
            # (remark: since we use ppl for action caption selection, this is a deprecated artifact)
            raw_object_detections = [action_captions[0] for action_captions in objects_per_frame.values()]
            raw_object_detections = dict(zip(objects_per_frame.keys(), raw_object_detections))

            # use as many objects as specified in the config
            final_object_detections = {}
            for interval, completion in raw_object_detections.items():
                # assuming that we prompted the model to list objects (e.g. "provide an enumerated list")
                objects_in_completion = parse_list(completion)

                # only use the specified number of objects per frame
                if len(objects_in_completion) > 0:
                    assert self.get_unspecific_objects_from_video_clip_config['num_objects_per_frame'] > 0, \
                        "The number of objects per frame should be greater than 0 (num_objects_per_frame)."
                    num_objects = self.get_unspecific_objects_from_video_clip_config['num_objects_per_frame']

                    # get the specified number of objects
                    objects = objects_in_completion[:num_objects]

                    # remove dots and linebreaks from objects
                    objects = [obj.replace(".", "") for obj in objects]
                    objects = [obj.replace("\n", "") for obj in objects]

                    # strip each object
                    objects = [obj.strip() for obj in objects]

                    # remove commas from the ends of each object
                    objects = [obj[:-1] if obj[-1] == "," else obj for obj in objects]

                    # remove ", and" from the ends of each object
                    objects = [obj[:-5] if obj[-5:] == ", and" else obj for obj in objects]
                else:
                    objects = []

                final_object_detections[interval] = objects

        # no, we do not want to remove duplicate objects since we want to preserve the temporal information as well!!!
        # we also do not remove duplicate action captions
        # removed on 04.05.2024 11:45
        # remove duplicates
        # objects = list(dict.fromkeys(objects))

        logger.info(f"Extracted unspecific objects from video clip: {final_object_detections}")

        return list(final_object_detections.values())

    def get_specific_object_confidences_from_video_clip(self, video_clip: VideoClip, question: str,
                                                        options: dict[str, str]):
        # reset the seed to ensure reproducibility
        self.reset_seed()

        if self.get_specific_objects_from_video_clip_config.get("replace_c", True):
            question = replace_c_with_camera_wearer(question)
            options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

        # format the prompt
        prompt = self.get_specific_objects_from_video_clip_config["prompt_template"].format(
            question=question,
            option_0=options["option 0"],
            option_1=options["option 1"],
            option_2=options["option 2"],
            option_3=options["option 3"],
            option_4=options["option 4"]
        )
        completion_start = self.get_specific_objects_from_video_clip_config["completion_start"]

        # get completion prediction from llm
        completion = self.get_completion_from_text(
            text=prompt,
            completion_start=completion_start,
            max_new_tokens=self.get_specific_objects_from_video_clip_config["max_new_tokens"],
            temperature=self.get_specific_objects_from_video_clip_config["temperature"]
        )
        logger.debug(f"Derived llm completion about objects of the task to pay "
                     f"attention to while watching the video: {completion}")

        # parse the completion
        objects = parse_list(text=completion)

        # filter the objects
        objects = filter_list_of_objects(objects)

        logger.debug(f"Derived objects of the task to pay attention to while watching the video: {objects}")

        confidences_per_object = {}
        for o in objects:
            # infer the object detections of the object in the video clip
            object_detections = self.get_object_detections_from_video_clip_and_text(video_clip=video_clip, text=o)

            # calculate mean confidence of the object detection
            mean_confidence = sum([obj["probability"] for obj in object_detections]) / len(object_detections)

            # in GroundingDINO they use 0.3 as the box threshold, so we use it as inflection point
            mean_confidence = 1 / (1 + np.exp(-10 * (mean_confidence - 0.3)))

            confidences_per_object[o] = mean_confidence

        return confidences_per_object
