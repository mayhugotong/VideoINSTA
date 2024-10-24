from __future__ import annotations


class Task:
    def __init__(self, question: str, options: dict[str, str]):
        self.question: str = question
        self.options: dict[str, str] = options

    def get_json_representation(self) -> dict:
        return {
            "question": self.question,
            "options": self.options
        }
