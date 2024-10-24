from __future__ import annotations

import logging
import time

import openai
import os

from toolbox.llm.base import LLM
from openai import OpenAI
from typing import Optional

logger = logging.getLogger("root")


class OpenAILLM(LLM):
    def __init__(
            self,
            openai_model_id: str,
            openai_token: str,
            temperature: float,
            max_new_tokens: int,
            frequency_penalty: float,
            presence_penalty: float,
            seed: int,
            api_type: str,
            system_prompt: Optional[str],
            use_seed: bool
    ):
        self.openai_model_id = openai_model_id
        self.openai_token = openai_token
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.api_type = api_type
        self.system_prompt = system_prompt if system_prompt else "You are a intelligent system."
        self.use_seed = use_seed

        logger.warning(f"Using OpenAI API key found in environment variable OPENAI_API_KEY. "
                       f"Make sure to specify the API key in the .env file in your project root. ")

        openai.my_api_key = self.openai_token
        self.client = OpenAI(
            api_key=self.openai_token
        )
        logger.debug(f"Initialized OpenAI client of OpenAI LLM {self.openai_model_id}")

    @staticmethod
    def initialize_openai_llm_from_config(config: dict[str, any]) -> OpenAILLM:
        return OpenAILLM(
            openai_model_id=config["llm_name"],
            openai_token=os.getenv("OPENAI_API_KEY"),
            temperature=config["temperature"],
            max_new_tokens=config["max_new_tokens"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
            seed=config["seed"],
            api_type=config["api_type"],
            system_prompt=config["system_prompt"],
            use_seed=config["use_seed"]
        )

    def get_completion(self, prompt: str, completion_start: str, max_new_tokens: Optional[int],
                       temperature: Optional[float]) -> str:
        # for ChatGPT, we just append the completion start to the prompt
        prompt = prompt + completion_start
        logger.debug(f"Sending prompt to OpenAI's ChatGPT API: {prompt}")

        # initialize params for exponential backoff strategy
        num_retries = 0
        delay = 16
        max_delay = 8 * 60 * 60

        # define on-call (optional) hyperparameters if given
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature

        while True:
            try:

                if self.api_type == "completions":
                    # completions API (chat-less, compare https://platform.openai.com/docs/api-reference/completions)
                    response = self.client.completions.create(
                        prompt=prompt,
                        model=self.openai_model_id,
                        presence_penalty=self.presence_penalty,
                        frequency_penalty=self.frequency_penalty,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=self.seed if self.use_seed else None
                    )

                    completion = response['choices'][0]['text']

                    logger.debug(f"Received completion from OpenAI's ChatGPT API: {completion}")

                    return completion

                if self.api_type == "chat":
                    # chat completions API (compare https://platform.openai.com/docs/api-reference/chat)
                    response = self.client.chat.completions.create(
                        model=self.openai_model_id,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        presence_penalty=self.presence_penalty,
                        frequency_penalty=self.frequency_penalty,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=self.seed if self.use_seed else None
                    )

                    logger.debug(f"Received response from OpenAI's ChatGPT API: {response}")

                    completion = response.choices[0].message.content

                    logger.debug(f"Received completion from OpenAI's ChatGPT API: {completion}")

                    return completion

                raise ValueError(f"Invalid API type given to OpenAILLM: {self.api_type}")

            except openai.OpenAIError as e:
                num_retries += 1

                logger.warning(e)
                logger.warning(f"OpenAI API error occurred ({num_retries} times), "
                               f"waiting for {delay} seconds and try again (exponential backoff)...")

                time.sleep(delay)
                delay *= 2
                delay = min(delay, max_delay)

    def build_model(self):
        logger.debug("OpenAI LLM does not require a model to be built since it uses the remote OpenAI API.")

    def destroy_model(self):
        logger.debug("OpenAI LLM does not require a model to be destroyed since it uses the remote OpenAI API.")
