import logging
from typing import Optional

from api.api import API
from video_reasoning.structure.videoinsta import VideoINSTA, Clip
from video_reasoning.operation.base import Operation

logger = logging.getLogger("root")


class NoConclusion(Operation):
    def __init__(self):
        super().__init__()

    def _execute(self, graph: Optional[VideoINSTA], api: Optional[API], target=Optional[Clip]) -> dict[str, any]:
        logger.info("Executed rule-based conclusion operation: NoConclusion -> no conclusion.")

        return {
            "final_prediction": "no conclusion"
        }
