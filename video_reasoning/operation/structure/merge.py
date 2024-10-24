import logging
from typing import Optional

from api.api import API
from video_reasoning.structure.videoinsta import Clip, VideoINSTA, Relation
from video_reasoning.operation.base import Operation

logger = logging.getLogger("root")


class Merge(Operation):
    def __init__(self):
        super().__init__()

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target: Optional[list[Clip]]) -> None:
        # get the states of the source clips
        source_clip_states = [source_clip.state for source_clip in target]

        # collect all video clips and merge them
        merged_clip_state = source_clip_states[0]
        for i in range(1, len(source_clip_states)):
            # get the current state
            current_state = source_clip_states[i]

            # merge the current clip state with the combined state
            merged_clip_state = merged_clip_state.merge(current_state)

        # create a new clip with the merged state
        target_clip = Clip(state=merged_clip_state)

        relations = [Relation(source=source_clip, target=target_clip)
                     for target_clip in [target_clip]
                     for source_clip in target]

        structure.add_clips([target_clip])
        structure.add_relations(relations)

        logger.info(f"Executed structure merging operation: Merge")

    def __str__(self):
        return "Merge"
