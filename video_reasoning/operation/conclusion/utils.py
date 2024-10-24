import logging
import random

from api.api import API
from api.utils import parse_option_answer_from_text, parse_answer_json, parse_option_answer_naive, \
    replace_c_with_camera_wearer, letter_to_option_id, get_single_letter_candidates_from_text, \
    get_free_form_candidate_from_text
from video_reasoning.structure.videoinsta import Clip
from video_reasoning.state.task import Task

logger = logging.getLogger("root")


def get_concatenated_lexical_state_representation_from_clips(clips: list[Clip]) -> str:
    # get the state for each clip
    states = [clip.state for clip in clips]

    # do not use the get_lexical_representation method of the state, because it is too detailed
    # therefore, compose the states manually
    lexical_state_representations = []
    for state in states:
        # get the lexical representation
        lexical_state_representation = state.get_lexical_representation()

        # append the concatenated lexical representation to the list
        lexical_state_representations.append(lexical_state_representation)

    concatenated_lexical_state_representation = "\n".join(lexical_state_representations)

    logger.debug(f"Concatenated lexical state representation without ratings of all sub clips: "
                 f"{concatenated_lexical_state_representation}")

    return concatenated_lexical_state_representation


def derive_options_candidate_from_whole_video_state(
        whole_video_summary: str,
        whole_video_state: str,
        task: Task,
        api: API,
        prompt_template: str,
        completion_start: str,
        max_new_tokens: int,
        replace_c: bool = False,
        parse_strategy: str = "default",
        temperature: float = None,
        whole_video_length_sec: int = 180
) -> (list[str], str):
    # TODO outsource in api
    logger.info("Deriving candidate from whole lexical video state representation using LLM...")

    # get question and options
    question = task.question
    options = task.options

    # replace "c" with "the camera wearer" if specified
    if replace_c:
        # only do the replacements in the data, not in the prompt template
        # removed at 24.05.2024, because we already replace C beforehand
        # re-added at 29.05.2024, because we need to replace C in the data if it will not be replaced before summarization
        whole_video_summary = replace_c_with_camera_wearer(whole_video_summary)
        whole_video_state = replace_c_with_camera_wearer(whole_video_state)

        # see https://ego4d-data.org/docs/data/annotation-guidelines/#pre-annotations-narrations
        # like https://arxiv.org/pdf/2404.04346, they also replace "C" with "the camera wearer" in the dataset
        question = replace_c_with_camera_wearer(question)
        options = {option_id: replace_c_with_camera_wearer(option) for option_id, option in options.items()}

    prompt = prompt_template.format(
        whole_video_summary=whole_video_summary,
        whole_video_state=whole_video_state,
        question=question,
        option_0=options["option 0"],
        option_1=options["option 1"],
        option_2=options["option 2"],
        option_3=options["option 3"],
        option_4=options["option 4"],
        whole_video_length_sec=whole_video_length_sec
    )

    logger.debug(f"Formatted single-choice QA prompt: {prompt}")
    logger.debug(f"Num chars of formatted single-choice QA prompt: {len(prompt)}")
    logger.debug(f"Num words of formatted single-choice QA prompt: {len(prompt.split())}")

    # get the final answer using the LLM
    completion = api.get_completion_from_text(
        text=prompt,
        completion_start=completion_start,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    logger.debug(f"Derived llm completion about final option candidate: {completion}")

    # parse depending on the specified strategy
    if parse_strategy == "default":
        candidate = parse_option_answer_from_text(completion)
    elif parse_strategy == "naive":
        candidate = parse_option_answer_naive(completion)
    elif parse_strategy == "json":
        # define the priority of certain keywords
        keywords_in_priority_order = [
            'best_answer',
            'final_answer',
            'answer',
            'answer_candidate',
            'best_option',
            'final_option',
            'option_candidate',
            'best_letter',
            'final_letter',
            'letter',
            'letter_candidate',
            'best_choice',
            'final_choice',
            'choice',
            'choice_candidate',
            'answer_option'
        ]
        candidate = parse_answer_json(
            text=completion,
            keywords_in_priority_order=keywords_in_priority_order,
            candidate_fun=get_single_letter_candidates_from_text
        )
        if candidate is not None:
            candidate = letter_to_option_id(candidate)
    else:
        raise ValueError(f"Unknown parse strategy: {parse_strategy}")
    logger.debug(f"Derived candidate from concatenated lexical state representation using LLM: {candidate}")

    # randomly choose a completion if the parsing failed
    if not candidate:
        logger.warning("Failed to parse candidate from completion. Randomly chose a candidate.")
        candidate = random.choice(["option 0", "option 1", "option 2", "option 3", "option 4"])
        candidate = [candidate]

    return candidate, completion, prompt


def derive_free_form_candidate_from_whole_video_state(
        whole_video_summary: str,
        whole_video_state: str,
        question: str,
        api: API,
        prompt_template: str,
        completion_start: str,
        max_new_tokens: int,
        replace_c: bool = False,
        parse_strategy: str = "default",
        temperature: float = None,
        whole_video_length_sec: int = 180
) -> (list[str], str):
    # TODO outsource in api
    logger.info("Deriving candidate from whole lexical video state representation using LLM...")

    # replace "c" with "the camera wearer" if specified
    if replace_c:
        # only do the replacements in the data, not in the prompt template
        # removed at 24.05.2024, because we already replace C beforehand
        # re-added at 29.05.2024, because we need to replace C in the data if it will not be replaced before summarization
        whole_video_summary = replace_c_with_camera_wearer(whole_video_summary)
        whole_video_state = replace_c_with_camera_wearer(whole_video_state)

        # see https://ego4d-data.org/docs/data/annotation-guidelines/#pre-annotations-narrations
        # like https://arxiv.org/pdf/2404.04346, they also replace "C" with "the camera wearer" in the dataset
        question = replace_c_with_camera_wearer(question)

    prompt = prompt_template.format(
        whole_video_summary=whole_video_summary,
        whole_video_state=whole_video_state,
        question=question,
        whole_video_length_sec=whole_video_length_sec
    )

    logger.debug(f"Formatted open-ended QA prompt: {prompt}")
    logger.debug(f"Num chars of formatted open-ended QA prompt: {len(prompt)}")
    logger.debug(f"Num words of formatted open-ended QA prompt: {len(prompt.split())}")

    # get the final answer using the LLM
    completion = api.get_completion_from_text(
        text=prompt,
        completion_start=completion_start,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    logger.debug(f"Derived llm completion about final free-form candidate: {completion}")

    # parse depending on the specified strategy
    if parse_strategy == "default":
        logger.warning("Default parse strategy is not directly supported for free-form answers. "
                       "Using naive strategy instead which uses the whole completion as answer.")
        candidate = [completion]
    elif parse_strategy == "naive":
        candidate = [completion]
    elif parse_strategy == "json":
        # define the priority of certain keywords
        keywords_in_priority_order = [
            'final_answer',
            'final-answer',
            'free_form_answer',
            'free-form-answer',
            'free-form_answer',
            'open_ended_answer',
            'open-ended-answer',
            'open-ended_answer',
            'best_answer',
            'answer',
            'answer_candidate',
            'answer-candidate',
            'solution',
            'solution_candidate',
            'solution-candidate',
            'response',
            'response_candidate',
            'response-candidate',
            'final_response',
            'final_solution'
        ]
        candidate = parse_answer_json(
            text=completion,
            keywords_in_priority_order=keywords_in_priority_order,
            candidate_fun=get_free_form_candidate_from_text
        )
        if candidate is None:
            logger.warning("Failed to parse candidate from completion. Using the whole completion as answer.")
            candidate = [completion]
    else:
        raise ValueError(f"Unknown parse strategy: {parse_strategy}")
    logger.debug(f"Derived free-form candidate from concatenated lexical state representation using LLM: {candidate}")

    # use default candidate if parsing failed
    if not candidate:
        logger.warning("Failed to parse free-form candidate from completion. Randomly chose a candidate.")
        candidate = ["I don't know."]

    return candidate, completion, prompt
