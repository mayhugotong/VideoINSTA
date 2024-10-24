# NOTE: this file is only needed for the evaluation of free-form question answering datasets

import argparse
import numpy as np
import os
import random
import torch

from datasets.utils import write_json_file, read_json_file
from datetime import datetime
from dotenv import load_dotenv
from toolbox.llm.local import HuggingFaceLLM


def recursive_read(folder_path, save_dict):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    if not subfolders:
        return

    for subfolder in subfolders:

        conclusions_file_path = os.path.join(subfolder, 'conclusions.json')

        if os.path.exists(conclusions_file_path):

            data = read_json_file(conclusions_file_path)
            for key, value in data.items():
                if key not in save_dict.keys():
                    save_dict[key] = value
                else:
                    for i in range(len(data[key])):
                        question = list(data[key])[i]
                        save_dict[key][question] = data[key][question]
        else:

            recursive_read(subfolder, save_dict)
        return save_dict


def read_conclusions_from_subfolders(root_folder, save_path):
    save_dict = {}
    save_dict = recursive_read(root_folder, save_dict)
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    save_con_pth = save_path + f'/merged_conclusion_{now}.json'
    write_json_file(file_path=save_con_pth, data=save_dict)
    return save_dict


def merge_qap(merged_conclusion_dict, save_path):
    qap_dict = {}
    answer_data = read_json_file(answer_file_path)
    question_data = read_json_file(question_file_path)
    for video_id, value in merged_conclusion_dict.items():
        for question, dict_q in merged_conclusion_dict[video_id].items():
            prediction = merged_conclusion_dict[video_id][question][str(len(dict_q) - 1)]['final_prediction']
            for item in question_data:
                if item['question'] == question and item['video_name'] == video_id[2:]:
                    question_id = item['question_id']
                    break
            for item in answer_data:
                if item['question_id'] == question_id:
                    answer = item['answer']
                    break
            if video_id in qap_dict.keys():
                qap_dict[video_id][question_id] = {}
                qap_dict[video_id][question_id]['question'] = question
                qap_dict[video_id][question_id]['answer'] = answer
                qap_dict[video_id][question_id]['prediction'] = prediction
            else:
                qap_dict[video_id] = {}
                qap_dict[video_id][question_id] = {}
                qap_dict[video_id][question_id]['question'] = question
                qap_dict[video_id][question_id]['answer'] = answer
                qap_dict[video_id][question_id]['prediction'] = prediction

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    save_qa_pth = save_path + f'/merged_qap_{now}.json'
    write_json_file(file_path=save_qa_pth, data=qap_dict)

    return qap_dict


def gpt_based_evaluation(root_folder, save_path, start=None, end=None):
    merged_conclusion_dict = read_conclusions_from_subfolders(root_folder, save_path)
    qap_dict = merge_qap(merged_conclusion_dict, save_path)

    system_prompt = """You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.
                Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:
                ------
                ##INSTRUCTIONS: 
                - Focus on the meaningful match between the predicted answer and the correct answer.\n
                - Consider synonyms or paraphrases as valid matches.\n
                - Evaluate the correctness of the prediction compared to the answer."""

    load_dotenv()
    seed = 217687
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    evaluation_dict = {}
    count = 0
    for video_id, dict_v in qap_dict.items():

        for question_id, qap in dict_v.items():

            if not start and end:
                question = qap['question']
                answer = qap['answer']
                pred = qap['prediction']
                if video_id in evaluation_dict.keys():
                    evaluation_dict[video_id][question_id] = {}
                    evaluation_dict[video_id][question_id]['question'] = question
                    evaluation_dict[video_id][question_id]['answer'] = answer
                    evaluation_dict[video_id][question_id]['prediction'] = pred
                else:
                    evaluation_dict[video_id] = {}
                    evaluation_dict[video_id][question_id] = {}
                    evaluation_dict[video_id][question_id]['question'] = question
                    evaluation_dict[video_id][question_id]['answer'] = answer
                    evaluation_dict[video_id][question_id]['prediction'] = pred

                prompt = "Please evaluate the following video-based question-answer pair:\n\n" \
                         f"Question: {question}\n" \
                         f"Correct Answer: {answer}\n" \
                         f"Predicted Answer: {pred}\n\n" \
                         "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. " \
                         "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING." \
                         "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. " \
                         "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."

                llm = HuggingFaceLLM(
                    # hf_model_id="meta-llama/Llama-2-13b-chat-hf",
                    hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
                    hf_token=os.getenv("HUGGINGFACE_API_KEY"),
                    do_sample=False,
                    temperature=0.0,
                    top_p=0.9,
                    max_new_tokens=512,
                    repetition_penalty=1.2,
                    use_cache=True,
                    precision=15,
                    system_prompt=system_prompt
                )
                llm.build_model()

                completion = llm.get_completion(prompt=prompt, completion_start="", max_new_tokens=10, temperature=None)
                evaluation_dict[video_id][question_id]['evaluation'] = completion
            else:
                if count > end:
                    break
                elif count < start:
                    pass
                else:
                    question = qap['question']
                    answer = qap['answer']
                    pred = qap['prediction']
                    if video_id in evaluation_dict.keys():
                        evaluation_dict[video_id][question_id] = {}
                        evaluation_dict[video_id][question_id]['question'] = question
                        evaluation_dict[video_id][question_id]['answer'] = answer
                        evaluation_dict[video_id][question_id]['prediction'] = pred
                    else:
                        evaluation_dict[video_id] = {}
                        evaluation_dict[video_id][question_id] = {}
                        evaluation_dict[video_id][question_id]['question'] = question
                        evaluation_dict[video_id][question_id]['answer'] = answer
                        evaluation_dict[video_id][question_id]['prediction'] = pred

                    prompt = "Please evaluate the following video-based question-answer pair:\n\n" \
                             f"Question: {question}\n" \
                             f"Correct Answer: {answer}\n" \
                             f"Predicted Answer: {pred}\n\n" \
                             "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. " \
                             "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING." \
                             "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. " \
                             "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."

                    llm = HuggingFaceLLM(
                        # hf_model_id="meta-llama/Llama-2-13b-chat-hf",
                        hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
                        hf_token=os.getenv("HUGGINGFACE_API_KEY"),
                        do_sample=False,
                        temperature=0.0,
                        top_p=0.9,
                        max_new_tokens=512,
                        repetition_penalty=1.2,
                        use_cache=True,
                        precision=15,
                        system_prompt=system_prompt
                    )
                    llm.build_model()

                    completion = llm.get_completion(prompt=prompt, completion_start="", max_new_tokens=10,
                                                    temperature=None)
                    evaluation_dict[video_id][question_id]['evaluation'] = completion
            count += 1

    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    save_eva_path = save_path + f'/evaluation_{now}.json'
    write_json_file(file_path=save_eva_path, data=evaluation_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions based on question-answer pairs.")
    parser.add_argument('--answer_file_path', type=str,
                        default='/path/to/answers/',
                        help="Path to the answer JSON file.")
    parser.add_argument('--question_file_path', type=str,
                        default='/path/to/queries',
                        help="Path to the question JSON file.")
    parser.add_argument('--root_folder', type=str, required=True, help="Root folder path to read subfolders.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the output files.")
    parser.add_argument('--start_sample_index', type=int, default=None)
    parser.add_argument('--end_sample_index', type=int, default=None)

    args = parser.parse_args()

    answer_file_path = args.answer_file_path
    question_file_path = args.question_file_path
    root_folder = args.root_folder
    save_path = args.save_path
    start = args.start_sample_index
    end = args.end_sample_index

    gpt_based_evaluation(root_folder, save_path, start, end)
