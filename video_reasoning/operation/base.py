from abc import ABC, abstractmethod
from typing import Union, Optional

from api.api import API
from video_reasoning.structure.videoinsta import VideoINSTA, Clip


class Operation(ABC):

    @abstractmethod
    def _execute(self, **kwargs) -> any:
        raise NotImplementedError

    def execute(self, structure: Optional[VideoINSTA], api: Optional[API], target: Optional[Union[Clip, list[Clip]]]) -> any:
        return self._execute(structure=structure, api=api, target=target)
