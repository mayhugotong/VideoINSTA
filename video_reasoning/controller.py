import logging
import os

from api.api import API
from datasets.utils import write_json_file, read_json_file
from video_reasoning.structure.videoinsta import VideoINSTA
from video_reasoning.operation.base import Operation
from video_reasoning.state.task import Task

logger = logging.getLogger("root")


class VideoReasoningController:
    def __init__(
            self,
            video_reasoning_structure: VideoINSTA,
            task: Task,
            api: API,
            split_operation: Operation,
            merge_operation: Operation,
            derive_clip_state_operation: Operation,
            derive_universal_state_operation: Operation,
            rating: Operation,
            decision: Operation,
            conclusion: Operation,
            states_save_path: str,
            predictions_save_path: str,
            conclusions_save_path: str,
            decisions_save_path: str,
            max_iterations: int,
            start_from: int = 0
    ):
        # initialize the video reasoning structure
        self.structure = video_reasoning_structure

        # initialize the operations
        self.split_operation = split_operation
        self.merge_operation = merge_operation
        self.derive_clip_state_operation = derive_clip_state_operation
        self.derive_universal_state_operation = derive_universal_state_operation
        self.rating = rating
        self.decision = decision
        self.conclusion = conclusion

        # initialize the task
        self.task = task

        # initialize the API
        self.api = api

        # initialize the paths to save the predictions and states
        self.predictions_save_path = predictions_save_path
        self.conclusions_save_path = conclusions_save_path
        self.decisions_save_path = decisions_save_path
        self.states_save_path = states_save_path

        # initialize structure parameters
        self.start_from = start_from
        self.max_iterations = max_iterations
        self.iteration = 0
        self.continue_execution = True

        # initialize the dicts to save the intermediate results
        self.conclusion_per_iteration = {}
        self.decisions_per_iteration = {}

        logger.info("Initialized video reasoning controller.")

    def single_choice_reasoning(self) -> str:

        # continue the reasoning until the stopping criterion is met
        while self.continue_execution:

            # 0) UNIVERSAL STATE DERIVATION (i.e. perceive data of the whole video)
            self.derive_universal_state_operation.execute(structure=self.structure, api=self.api, target=None)

            # 1) SPLIT (split all sub clips that are decidable and expand the structure using specified split operation)
            # remark: splittable clips are equal to decidable sub clips
            splittable_clips = self.structure.get_decidable_sub_clips()
            for clip in splittable_clips:
                self.split_operation.execute(structure=self.structure, api=self.api, target=clip)

            # 2) CLIP STATES DERIVATION (i.e. perceive the data of each clip in the structure that has not been derived yet)
            self.derive_clip_state_operation.execute(structure=self.structure, api=self.api, target=None)

            # 3) RATINGS DERIVATION (only need to derive the ratings for states of clips if they are not derived yet)
            self.rating.execute(structure=self.structure, api=self.api, target=None)

            # 4) CONCLUSION (conclude the final answer)
            conclusion = self.conclusion.execute(structure=self.structure, api=self.api, target=None)

            # 5) DECISION (decide the next operations for each sub clip that is not waiting)
            decided_operations, targets = self.decision.execute(structure=self.structure, api=self.api, target=None)

            # 6) APPLY THE DECISIONS (execute the decided operations on the decidable sub clips of the structure)
            for i in range(len(decided_operations)):
                for target in targets[i]:
                    decided_operations[i].execute(structure=self.structure, api=self.api, target=target)

            # 6) SAVE THE CONCLUSION, DECISIONS, AND STATES
            self.conclusion_per_iteration[self.iteration] = conclusion
            self.decisions_per_iteration[self.iteration] = str(decided_operations)
            self._save()

            # 7) increase the iteration
            self.iteration += 1

            # 8) STOPPING CRITERION (if max iteration is reached or all clips are waiting)
            stopping_criterion = self.iteration == self.max_iterations or all([clip.state.waiting for clip in self.structure.clips])
            self.continue_execution = False if stopping_criterion else True

        # return the prediction from the last iteration as the final prediction
        return self.conclusion_per_iteration[self.iteration - 1]["final_prediction"]

    def _save(self):
        # save the final prediction
        old_predictions_data = read_json_file(file_path=self.predictions_save_path)
        old_video_entry = old_predictions_data.get(self.structure.root.state.video_clip.id, {})
        old_question_entry = old_video_entry.get(self.task.question, {})
        new_question_entry = old_question_entry | {
            self.iteration: self.conclusion_per_iteration[self.iteration]["final_prediction"]
        }
        new_video_entry = old_video_entry | {
            self.task.question: new_question_entry
        }
        new_predictions_data = old_predictions_data | {
            self.structure.root.state.video_clip.id: new_video_entry
        }
        write_json_file(file_path=self.predictions_save_path, data=new_predictions_data)
        logger.info(f"Saved final prediction to {self.predictions_save_path}.")

        # save the conclusions
        old_conclusions_data = read_json_file(file_path=self.conclusions_save_path)
        old_video_entry = old_conclusions_data.get(self.structure.root.state.video_clip.id, {})
        old_question_entry = old_video_entry.get(self.task.question, {})
        new_question_entry = old_question_entry | self.conclusion_per_iteration
        new_video_entry = old_video_entry | {
            self.task.question: new_question_entry
        }
        new_conclusions_data = old_conclusions_data | {
            self.structure.root.state.video_clip.id: new_video_entry
        }
        write_json_file(file_path=self.conclusions_save_path, data=new_conclusions_data)
        logger.info(f"Saved conclusion to {self.conclusions_save_path}.")

        # save the decisions
        old_decisions_data = read_json_file(file_path=self.decisions_save_path)
        old_video_entry = old_decisions_data.get(self.structure.root.state.video_clip.id, {})
        old_question_entry = old_video_entry.get(self.task.question, {})
        new_question_entry = old_question_entry | self.decisions_per_iteration
        new_video_entry = old_video_entry | {
            self.task.question: new_question_entry
        }
        new_decisions_data = old_decisions_data | {
            self.structure.root.state.video_clip.id: new_video_entry
        }
        write_json_file(file_path=self.decisions_save_path, data=new_decisions_data)
        logger.info(f"Saved decisions to {self.decisions_save_path}.")

        # save the states
        all_clip_states = [clip.state.get_json_representation() for clip in self.structure.clips]
        sub_clips = [clip.state.get_json_representation() for clip in self.structure.get_sub_clips()]
        all_states_save_path = os.path.join(self.states_save_path, f"{self.iteration}_all.json")
        leaves_states_save_path = os.path.join(self.states_save_path, f"{self.iteration}_leaves.json")
        if not os.path.exists(self.states_save_path):
            os.makedirs(self.states_save_path)
        write_json_file(file_path=all_states_save_path, data=all_clip_states)
        write_json_file(file_path=leaves_states_save_path, data=sub_clips)
        logger.info(f"Saved all clip states to {all_states_save_path}.")
        logger.info(f"Saved sub clip states to {leaves_states_save_path}.")
