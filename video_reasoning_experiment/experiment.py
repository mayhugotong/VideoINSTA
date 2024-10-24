import importlib
import logging
import numpy as np
import os.path
import random
import shutil
import time
import torch

from abc import ABC, abstractmethod
from api.api import API
from datasets.load import load_data
from datasets.utils import create_video_data_from_video_path, read_json_file, write_json_file, slugify
from dotenv import load_dotenv
from video_reasoning.operation.conclusion.llm_based import OptionsCandidateBasedOnConcatenatedLexicalStates, \
    IterativeMergeConclusion
from video_reasoning.operation.conclusion.rule_based import NoConclusion
from video_reasoning.operation.decision.rule_based import NoDecision
from video_reasoning.state.spatial import SpatialClipState
from video_reasoning.state.temporal import TemporalClipState
from video_reasoning.structure.videoinsta import Clip, VideoINSTA
from video_reasoning.controller import VideoReasoningController
from video_reasoning.operation.state.derivation import DeriveClipStates, DeriveRootClipState
from video_reasoning.operation.state.flag import Wait
from video_reasoning.operation.state.rating import NoRatings, AnswerabilityRating
from video_reasoning.operation.structure.split import Split, DPCKNNSplit
from video_reasoning.operation.structure.merge import Merge
from video_reasoning.state.task import Task
from video_reasoning.state.clip import ClipState
from video_reasoning.state.video import VideoClip
from video_reasoning_experiment.utils import get_human_time

logger = logging.getLogger("root")


# deprecated
class VideoReasoningExperiment(ABC):
    """
    This class represents an abstract experiment that has a secret environment for private keys (e.g. API keys) accessed
    from a .env file and a data configuration accessed from a .yaml file.
    """

    def __init__(self, config: dict):
        """
        Abstract Experiment instances are used as the basis for specific experiment instances, so they are usually not
        instantiated directly but rather via a super call from a child class.
        It requires a configuration that specifies the path to a secret .env file for private keys (e.g. API keys) as
        well as a data-related configuration.

        :param config: A dictionary specifying the experiment configuration.
        """
        self.experiment_path = config.get("experiment_path")

        # read secret environment variables (e.g. API keys), note that this will probably not work with SLURM
        load_dotenv(config.get("secret_env"), override=True)

        logger.info("Initialized dataset and dataloader")

    def conduct(self, mode: str = "train"):
        """
        This function is the interface for conducting experiments. The mode of the experiment is specified by the mode
        parameter and this function maps the call to the specific implementation.

        :param mode: Either train, eval or test.
        """
        modes = {
            "train": self._train,
            "eval": self._eval,
            "test": self._test
        }
        modes[mode]()

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def _eval(self):
        pass

    @abstractmethod
    def _test(self):
        pass


class VideoReasoningVideoINSTAExperiment(VideoReasoningExperiment):
    """
    This class represents an experiment that uses VideoINSTA to reason through a given video and
    a corresponding task. A task is a combination of a task or question and its answer options.
    """

    def __init__(self, config: dict):
        """
        Instances of this class represent specific experiments that are defined with a configuration similar to the
        example configuration in ./config/foundation_models/default_video_describer.yaml.

        :param config: A dictionary specifying the experiment configuration.
        """
        super().__init__(config)

        if torch.cuda.is_available():
            # Get the number of available CUDA devices
            num_devices = torch.cuda.device_count()
            logger.info("Using the following CUDA visible devices:")
            for i in range(num_devices):
                logger.info(f"    - Device {i}: {torch.cuda.get_device_name(i)}")
            logger.warning("Note that this framework will use the assigned device indices instead of the real ones.")
        else:
            logger.warning("CUDA is not available. The framework will run on CPU. This is only possible if "
                           "pre-extracted data is used.")

        # initialize path and dir parameters
        self.videos_path = config.get("videos_path")
        self.tasks_path = config.get("tasks_path")
        self.answers_path = config.get("answers_path")
        self.predictions_path = os.path.join(self.experiment_path, "predictions.json")
        self.conclusions_path = os.path.join(self.experiment_path, "conclusions.json")
        self.decisions_path = os.path.join(self.experiment_path, "decisions.json")
        self.accuracy_path = os.path.join(self.experiment_path, "accuracy.json")
        self.states_path = os.path.join(self.experiment_path, "states")
        self.resume_path = config.get("resume_path", None)
        self.iterate_through_videos = config.get("iterate_through_videos", False)

        # create experiment paths and files
        self._create_experiment_paths_and_files(config)

        # initialize video-related parameters
        self.sample_rate = config.get("sample_rate")
        self.start_video = config.get("start_video", 0)
        self.end_video = config.get("end_video", 10 ** 9)

        # initialize subset indices if given
        self.subset_indices = config.get("subset_indices", None)

        # initialize VideoINSTA controller parameters
        self.max_iterations = config.get("controller").get("max_iterations")

        # initialize random seed
        self.random_seed = config.get("random_seed")
        self.reset_seed_for_each_function = config.get("reset_seed_for_each_function")
        self.reset_seed_for_each_video = config.get("reset_seed_for_each_video")

        # initialize state parameters
        self.spatial_clip_state_config = config.get("state").get("spatial_clip_state")
        self.temporal_clip_state_config = config.get("state").get("temporal_clip_state")
        self.lexical_representation = config.get("state").get("lexical_representation")

        # initialize the operations
        self._initialize_operations(config.get("operations"))

        # initialize the API
        self._initialize_api(config.get("api"))

        # initialize data
        self.data = load_data(
            answers_path=self.answers_path,
            tasks_path=self.tasks_path,
            normalize=config.get("normalize_data", False)
        )
        logger.info(f"Loaded data from {self.answers_path} and {self.tasks_path}.")

        # load the predictions (empty if starting new experiment)
        self.predictions = read_json_file(file_path=self.predictions_path)
        logger.info(f"Loaded predictions from {self.predictions_path}.")

        logger.info("Initialized static video reasoning VideoINSTA experiment")

    def _create_experiment_paths_and_files(self, config):
        # create experiment paths and files
        if not self.resume_path:
            logger.info("No resume path specified, starting new experiment...")

            paths = [
                self.predictions_path,
                self.conclusions_path,
                self.decisions_path
            ]

            # create predictions file if it does not exist, i.e. if the experiment is not resumed
            for path in paths:
                if not os.path.exists(path):
                    write_json_file(data={}, file_path=path)
                    logger.info(f"Created json file at {path}")

            logger.info(f"Starting experiment, saving predictions at {self.predictions_path}")
        else:
            logger.info("Resume path specified, resuming experiment...")

            resume_path = config["resume_path"]
            existing_predictions_path = os.path.join(resume_path, "predictions.json")
            existing_states_path = os.path.join(resume_path, "states")

            # break if the predictions file does not exist
            if not os.path.exists(existing_predictions_path):
                err_msg = f"Predictions file at {existing_predictions_path} does not exist"
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            # break if the states directory does not exist
            if not os.path.exists(existing_states_path):
                err_msg = f"States directory at {existing_states_path} does not exist"
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

            shutil.copytree(src=resume_path, dst=self.experiment_path, dirs_exist_ok=True)
            logger.info(f"Copied existing experiment from {resume_path} to {self.experiment_path}")

            logger.info(f"Resuming experiment, skipping videos for which predictions already "
                        f"have been made in the past at {resume_path}")

    def _initialize_operations(self, operations_config):
        # add new operations here
        available_operations = {
            # structural
            "Split": Split,
            "DPCKNNSplit": DPCKNNSplit,
            "Wait": Wait,
            "Merge": Merge,
            # state
            "DeriveClipStates": DeriveClipStates,
            "DeriveRootClipState": DeriveRootClipState,
            "NoRatings": NoRatings,
            "AnswerabilityRating": AnswerabilityRating,
            # decisions
            "NoDecision": NoDecision,
            # conclusions
            "OptionsCandidateBasedOnConcatenatedLexicalStates": OptionsCandidateBasedOnConcatenatedLexicalStates,
            "NoConclusion": NoConclusion,
            "IterativeMergeConclusion": IterativeMergeConclusion
        }

        # there must always be a split operation
        self.split_operation = available_operations[operations_config.get("split").get("class")](
            **operations_config.get("split").get("params"))

        # there must always be a wait operation
        self.wait_operation = available_operations[operations_config.get("wait").get("class")](
            **operations_config.get("wait").get("params"))

        # there must always be a merge operation
        self.merge_operation = available_operations[operations_config.get("merge").get("class")](
            **operations_config.get("merge").get("params"))

        # there must always be a derive_clip_state operation
        self.derive_clip_state_operation = available_operations[
            operations_config.get("derive_clip_states").get("class")](
            **operations_config.get("derive_clip_states").get("params"))

        # there must always be a derive_universal_state operation
        self.derive_root_clip_state_operation = available_operations[
            operations_config.get("derive_root_clip_state").get("class")](
            **operations_config.get("derive_root_clip_state").get("params"))

        # there must always be a derive_rating operation
        self.derive_rating_operation = available_operations[operations_config.get("ratings").get("class")](
            **operations_config.get("ratings").get("params"))

        # there must always be a decision operation
        self.decision = available_operations[operations_config.get("decision").get("class")](
            split=self.split_operation,
            merge=self.merge_operation,
            wait=self.wait_operation,
            **operations_config.get("decision").get("params"))

        # there must always be a conclusion operation
        self.conclusion = available_operations[operations_config.get("conclusion").get("class")](
            **operations_config.get("conclusion").get("params"))

    def _initialize_api(self, api_config):
        # add seed to OpenAI configs
        get_completion_from_text_config = api_config.get("get_completion_from_text")
        llm_class = get_completion_from_text_config.get("llm_class")
        if "openai" in llm_class.lower():
            get_completion_from_text_config["seed"] = self.random_seed

        self.api = API(
            load_models_only_when_needed=api_config.get("load_models_only_when_needed"),
            get_object_detections_from_video_clip_and_text_config=api_config.get(
                "get_object_detections_from_video_clip_and_text"),
            get_action_captions_from_video_clip_config=api_config.get("get_action_captions_from_video_clip"),
            get_temporal_grounding_from_video_clip_and_text_config=api_config.get(
                "get_temporal_grounding_from_video_clip_and_text"),
            get_summary_from_noisy_perceptive_data_config=api_config.get(
                "get_summary_from_noisy_perceptive_data"),
            get_completion_from_text_config=get_completion_from_text_config,
            get_unspecific_objects_from_video_clip_config=api_config.get("get_unspecific_objects_from_video_clip"),
            get_specific_objects_from_video_clip_config=api_config.get("get_specific_objects_from_video_clip"),
            random_seed=self.random_seed,
            reset_seed_for_each_function=self.reset_seed_for_each_function
        )

    def _train(self):
        logger.error("This framework does currently not support training experiments.")
        raise NotImplementedError

    def _eval(self):
        logger.info("Starting evaluation experiment...")
        logger.info(f"Starting video reasoning for videos {self.start_video} to {self.end_video} "
                    f"(i.e. {self.end_video - self.start_video}) videos in total)...")

        # measure the total execution time
        start_time_total = time.time()

        # count the number of correct predictions
        num_samples_correct = 0

        # count the total number of samples for which predictions have been made
        num_samples_total = 0

        # remember the visited video ids
        visited_video_ids = []

        # loop through the videos, skipping videos
        #     - for which predictions already exist
        #     - which are not in the specified video range
        #     - which have already been processed if self.iterate_through_videos is True
        #     - which are not in the specified subset indices
        for video_count, item in enumerate(self.data):
            # measure execution time per iteration / video
            start_time = time.time()

            # stop if the number of videos is reached
            if video_count == self.end_video:
                logger.info(f"Reached end of specified video range {self.start_video} to {self.end_video}.")
                break

            video_id = item["video_id"]
            question = item["question"]
            options = item["options"]

            # skip videos which are not in the specified video range
            if video_count < self.start_video:
                logger.info(f"Skipping video {video_id} because of specified video range...")
                continue

            # skip videos which have already been processed if self.iterate_through_videos is True
            if self.iterate_through_videos and video_id in visited_video_ids:
                logger.info(f"Skipping video {video_id} because it has already been processed"
                            f"and self.iterate_through_videos is set True...")
                continue

            # skip videos which are not in the specified subset indices
            if self.subset_indices is not None and video_count not in self.subset_indices:
                logger.info(f"Skipping video {video_id} because it is not in the specified subset indices...")
                continue

            # remember the visited video id
            visited_video_ids.append(video_id)

            # clear caches
            torch.cuda.empty_cache()
            importlib.invalidate_caches()

            # (re-)set the seed if specified or if it is the first video
            if self.reset_seed_for_each_video or video_count == self.start_video:
                # set the random seed (video-wise, so we can reproduce results for a certain video)
                # moreover, the seed will be re-set in the api for each "neural module" call (ensures comparability)
                # therefore, even when changing input sizes (e.g. of prompts),
                # the results with same seeds should be comparable
                random.seed(self.random_seed)
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)
                torch.cuda.manual_seed(self.random_seed)
                logger.info(f"Random seed has been reset to {self.random_seed} "
                            f"to ensure reproducibility and comparability.")
                # TODO are the following lines needed?
                # torch.use_deterministic_algorithms(True)
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False

            logger.info(f"Starting video reasoning for video {video_id}...")
            logger.info(f"Question: {question}.")
            logger.info(f"Options: {options}")

            # load and initialize video data
            # video_path = os.path.join(self.videos_path, f"{video_id}.mp4")
            video_path = next((os.path.join(self.videos_path, f"{video_id}{ext}") for ext in ['.mp4', '.mkv', '.avi'] if
                               os.path.exists(os.path.join(self.videos_path, f"{video_id}{ext}"))), None)
            if video_path is None:
                raise ValueError(
                    f"No video file found for video_id {video_id} with supported formats ['.mp4', '.mkv', '.avi']")
            video, metadata = create_video_data_from_video_path(video_path=video_path, sample_rate=self.sample_rate)
            logger.debug(f"original sampled indices: {metadata['sample_indices']}")
            video_clip = VideoClip.from_metadata(video, metadata)
            # video_clip = video_clip.get_trimmed_video_clip(0, 480)
            logger.info(f"Loaded and initialized video data from {video_path} with sample rate "
                        f"{self.sample_rate} and {len(video_clip)} frames.")

            # create the states save path for the current video
            states_save_path = os.path.join(self.states_path, video_id, slugify(question))

            # initialize the task
            task = Task(question=question, options=options)

            # initialize the root state (here is the first actual instantiation of the state class)
            # this root state is the universal clip state representing the perceptive data of the whole video
            root_state = ClipState(
                video_clip=video_clip,
                task=task,
                lexical_representation=self.lexical_representation,
                spatial_clip_state=SpatialClipState(
                    video_clip=video_clip,
                    task=task,
                    use_action_captions=self.spatial_clip_state_config.get("use_action_captions"),
                    use_object_detections=self.spatial_clip_state_config.get("use_object_detections"),
                    use_action_captions_summary=self.spatial_clip_state_config.get("use_action_captions_summary"),
                    use_object_detections_summary=self.spatial_clip_state_config.get(
                        "use_object_detections_summary"),
                    lexical_representation=self.lexical_representation
                ),
                temporal_clip_state=TemporalClipState(
                    video_clip=video_clip,
                    task=task,
                    lexical_representation=self.lexical_representation,
                    use_foreground=self.temporal_clip_state_config.get("use_foreground"),
                    use_relevance=self.temporal_clip_state_config.get("use_relevance"),
                    use_salience=self.temporal_clip_state_config.get("use_salience"),
                    use_temporal_grounding_summary=self.temporal_clip_state_config.get(
                        "use_temporal_grounding_summary")
                )
            )

            # initialize the root clip
            root_clip = Clip(state=root_state)

            # initialize VideoINSTA structure
            video_reasoning_structure = VideoINSTA(root_clip=root_clip)

            # initialize the video reasoning controller
            video_reasoning_controller = VideoReasoningController(
                video_reasoning_structure=video_reasoning_structure,
                task=task,
                api=self.api,
                states_save_path=states_save_path,
                split_operation=self.split_operation,
                merge_operation=self.merge_operation,
                derive_clip_state_operation=self.derive_clip_state_operation,
                derive_universal_state_operation=self.derive_root_clip_state_operation,
                rating=self.derive_rating_operation,
                decision=self.decision,
                conclusion=self.conclusion,
                max_iterations=self.max_iterations,
                predictions_save_path=self.predictions_path,
                conclusions_save_path=self.conclusions_path,
                decisions_save_path=self.decisions_path
            )

            # execute the video reasoning, choose one of the options as answer
            choice = video_reasoning_controller.single_choice_reasoning()

            # remember the number of correct answers
            if choice == item["answer"]:
                num_samples_correct += 1

            # remember the number of total samples
            num_samples_total += 1

            # calculate the current accuracy and log it
            current_accuracy = num_samples_correct / num_samples_total
            logger.info(f"The accuracy of {num_samples_total} videos is {current_accuracy}.")

            # measure execution time per iteration / video
            end_time = time.time()
            logger.info(f"Reasoning for video {video_id} took {end_time - start_time} seconds (i.e. "
                        f"{get_human_time(end_time - start_time)}).")

        logger.info("Evaluating the predicted answers...")

        # evaluate the predictions iteration-wise
        all_predictions = read_json_file(file_path=self.predictions_path)

        num_correct_per_iteration = [0] * self.max_iterations
        num_total = [0] * self.max_iterations

        # iterate through the predictions
        for video_id, video_predictions in all_predictions.items():

            # there can be more than one prediction per video
            # (since datasets like NExT-QA have multiple questions per video)
            for question, question_prediction in video_predictions.items():

                # get the data entry corresponding to the video_id and question
                corresponding_data_entry = [item for item in self.data if
                                            item["video_id"] == video_id and
                                            item["question"] == question][0]

                # get the ground truth
                ground_truth = corresponding_data_entry["answer"]

                # compare the ground truth with the prediction for each iteration
                for iteration, prediction in question_prediction.items():
                    logger.debug(f"Comparison of video {video_id} and question {question}: "
                                 f"GT: {ground_truth} - Prediction: {prediction}")

                    # get the index for this iteration
                    index = int(iteration) - 1

                    # remember the number of correct predictions for this iteration
                    if prediction == ground_truth:
                        num_correct_per_iteration[index] += 1

                    # remember total number of predictions for this iteration
                    num_total[index] += 1

        # fill up the rest of the iterations with the last valid value
        for i in range(len(num_correct_per_iteration)):
            if num_correct_per_iteration[i] == 0 and i > 0:
                num_correct_per_iteration[i] = num_correct_per_iteration[i - 1]

        # calculate the accuracy per iteration
        accuracy_per_iteration = {str(i + 1): {
            "num_correct": num_correct,
            "num_total": num_total[i],
            "accuracy": num_correct / num_total[i] if num_total[i] > 0 else "N/A"
        } for i, num_correct in enumerate(num_correct_per_iteration)}

        for iteration, accuracy in accuracy_per_iteration.items():
            logger.info(f"Number of correct predictions for iteration {iteration}: {accuracy['num_correct']}")
            logger.info(f"Number of samples for iteration {iteration}: {accuracy['num_total']}")
            logger.info(f"Accuracy for iteration {iteration}: {accuracy['accuracy']}")

        # save the accuracies
        write_json_file(data=accuracy_per_iteration, file_path=self.accuracy_path)

        # measure the total execution time
        end_time_total = time.time()
        logger.info(f"Finished evaluation experiment in {end_time_total - start_time_total} seconds (i.e. "
                    f"{get_human_time(end_time_total - start_time_total)}).")

    def _test(self):
        logger.error("This framework does currently not support testing experiments.")
        raise NotImplementedError
