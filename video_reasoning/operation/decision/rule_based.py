import logging
from typing import Optional

from api.api import API
from video_reasoning.operation.base import Operation
from video_reasoning.operation.decision.base import Decision
from video_reasoning.operation.state.flag import Wait
from video_reasoning.operation.structure.split import Split
from video_reasoning.operation.structure.merge import Merge
from video_reasoning.structure.videoinsta import Clip, VideoINSTA

logger = logging.getLogger("root")


class NoDecision(Decision):

    def __init__(self, split: Split, merge: Merge, wait: Wait):
        super().__init__(split=split, merge=merge, wait=wait)

    def _execute(
            self,
            structure: Optional[VideoINSTA],
            api: Optional[API],
            target=Optional[Clip]
    ) -> tuple[list[Operation], list[list[Clip]]]:
        decidable_sub_clips = structure.get_decidable_sub_clips()

        operations = []
        for clip in decidable_sub_clips:
            # do nothing (wait for being merged)
            logger.debug(f"Decided the NoDecision operation for clip ({clip.state.video_clip.sampled_indices[0]}, "
                         f"{clip.state.video_clip.sampled_indices[-1]})...")
            operations.append(self.wait)

        logger.info(f"Executed rule-based decision operation: NoDecision")

        # all decidable sub clips are returned as each of them will be set in the waiting state
        return [self.wait], [decidable_sub_clips]
