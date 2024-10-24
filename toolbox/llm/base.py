from abc import abstractmethod, ABC
from typing import Optional


class LLM(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, completion_start: str, max_new_tokens: Optional[int],
                       temperature: Optional[float]) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def destroy_model(self):
        raise NotImplementedError
