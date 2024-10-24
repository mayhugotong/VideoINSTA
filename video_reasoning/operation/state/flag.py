import logging
from typing import Optional

from api.api import API
from video_reasoning.structure.videoinsta import Clip, VideoINSTA
from video_reasoning.operation.base import Operation

logger = logging.getLogger("root")


class Wait(Operation):
    def __init__(self):
        super().__init__()

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target: Optional[Clip]) -> None:
        target.state.waiting = True
        logger.info(f"Executed state flag operation: Wait")

    def __str__(self):
        return "Wait"
