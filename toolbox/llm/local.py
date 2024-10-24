from __future__ import annotations

import gc
import logging
import os
import torch

from toolbox.llm.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional

logger = logging.getLogger("root")


class HuggingFaceLLM(LLM):
    def __init__(
            self,
            hf_model_id: str,
            hf_token: str,
            precision: int,
            do_sample: bool,
            temperature: float,
            top_p: float,
            max_new_tokens: int,
            repetition_penalty: float,
            use_cache: bool,
            system_prompt: Optional[str]
    ):
        self.hf_model_id = hf_model_id
        self.hf_token = hf_token

        self.precision = precision
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.use_cache = use_cache
        self.system_prompt = system_prompt if system_prompt else "You are a intelligent system."

        self.BOSYS = "<<SYS>>\n"
        self.EOSYS = "<</SYS>>\n\n"

        self.BOINST = "[INST] "
        self.EOINST = " [/INST]"

        self.BOS = "<s>"
        self.EOS = "</s>"

        self._tokenizer = None
        self._model = None
        self._pipe = None

        logger.warning(f"Using HuggingFace API key found in environment variable HUGGINGFACE_API_KEY. "
                       f"Make sure to specify the API key in the .env file in your project root. ")
        logger.debug(f"Initialized local HuggingFace LLM {self.hf_model_id} (not loaded yet)")

    @staticmethod
    def initialize_huggingface_llm_from_config(config: dict[str, any]) -> HuggingFaceLLM:
        return HuggingFaceLLM(
            hf_model_id=config["llm_name"],
            hf_token=os.getenv("HUGGINGFACE_API_KEY"),
            precision=config["precision"],
            do_sample=config["do_sample"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_new_tokens=config["max_new_tokens"],
            repetition_penalty=config["repetition_penalty"],
            use_cache=config["use_cache"],
            system_prompt=config.get("system_prompt", None)
        )

    def build_model(self):
        # initialize the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.hf_model_id,
            return_tensors="pt",
            token=self.hf_token
        )
        logger.debug(f"Initialized and loaded tokenizer of HuggingFace LLM {self.hf_model_id}")

        precisions = {
            15: torch.bfloat16,
            16: torch.float16,
            32: torch.float32
        }

        # initialize the model
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.hf_model_id,
            device_map="auto",
            token=self.hf_token,
            torch_dtype=precisions[self.precision]
        )
        logger.debug(f"Initialized and loaded model of HuggingFace LLM {self.hf_model_id}")

        if "llama-3" in self.hf_model_id.lower():
            eos_token_id = [
                self._tokenizer.eos_token_id,
                self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            eos_token_id = self._tokenizer.eos_token_id

        self._pipe = pipeline(
            task="text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            use_cache=self.use_cache,
            eos_token_id=eos_token_id
        )

        # removed on 20.04.2024 01:04
        # turns out that HuggingFace's pipeline is faster than using the model directly
        # self._pipe = HuggingFacePipeline(pipeline=pipe)

    def destroy_model(self):
        del self._model
        del self._tokenizer
        del self._pipe

        self._model = None
        self._tokenizer = None
        self._pipe = None

        gc.collect()

        torch.cuda.empty_cache()

        logger.debug(f"Destroyed model of HuggingFace LLM {self.hf_model_id} and emptied the CUDA cache.")

    def get_completion(self, prompt: str, completion_start: str, max_new_tokens: Optional[int],
                       temperature: Optional[float]) -> str:
        assert self._pipe is not None, "Model not initialized. Call build_model() first."

        # define on-call (optional) hyperparameters if given
        do_sample = True if temperature is not None else self.do_sample
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature

        if "llama-2" in self.hf_model_id.lower():
            # compare https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5
            # compare https://github.com/meta-llama/llama/blob/main/llama/generation.py
            system_instruction = self.BOS + self.BOSYS + self.BOINST + self.system_prompt + self.EOINST + self.EOSYS + self.EOS
            user_instruction = self.BOS + self.BOINST + prompt + self.EOINST

            # for Llama2 models, we append the completion start to the prompt outside the instruction wrapper
            prompt = system_instruction + user_instruction + completion_start
            logger.debug(f"Completing prompt with HuggingFace's LLM {self.hf_model_id}: {prompt}")
            logger.debug(f"Prompt length in chars: {len(prompt)}")
            logger.debug(f"Prompt length in words: {len(prompt.split())}")

            # TODO more efficient way to get the token length of the prompt (now we are tokenizing twice)
            logger.debug(f"Prompt length in tokens: {len(self._tokenizer(prompt)['input_ids'])}")
        elif "llama-3" in self.hf_model_id.lower():
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            prompt = self._pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt = prompt + completion_start
            logger.debug(f"Applied chat template to prompt: {prompt}")
            logger.debug(f"Prompt length in chars: {len(prompt)}")
            logger.debug(f"Prompt length in words: {len(prompt.split())}")

        # retry the completion generation with less max_new_tokens if an error occurs
        while True:
            try:
                completion = self._pipe(
                    prompt,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )[0]["generated_text"]
                break
            except RuntimeError as e:
                logger.warning(f"Error occurred during completion generation: {e}")
                self.destroy_model()
                self.build_model()

                max_new_tokens //= 2
                if max_new_tokens < 1:
                    err_msg = f"Could not generate completion after multiple retries with less max_new_tokens."
                    logger.error(err_msg)
                    raise ValueError(err_msg)
                logger.warning(
                    f"Rebuild the model and retry the completion generation with max_new_tokens={max_new_tokens // 2}.")

        # remove the prompt from the completion if it is still there
        completion = completion[len(prompt):].strip() if completion.startswith(prompt) else completion.strip()

        logger.debug(f"Generated completion using model of HuggingFace LLM {self.hf_model_id}: {completion}")

        return completion
