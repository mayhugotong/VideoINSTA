import logging

from datasets.utils import read_json_file, read_csv_file, normalize_question, normalize_answer

logger = logging.getLogger("root")


def load_data(answers_path: str, tasks_path: str, normalize: bool = False):
    # TODO implement a data preprocessing function (capital first letter and question mark / dot at the end)
    if "egoschema" in answers_path.lower():
        # load the answers (i.e. ground truths)
        answers = read_json_file(file_path=answers_path)
        logger.info(f"Loaded ground truth answers from {answers_path}.")

        # load the tasks (i.e. questions and options)
        tasks = read_json_file(file_path=tasks_path)
        logger.info(f"Loaded tasks (questions and options) from {tasks_path}.")

        data = []
        for video_id, answer_option_id in answers.items():
            # there is only one task per video in EgoSchema
            task = [task for task in tasks if task["q_uid"] == video_id][0]
            data.append({
                "video_id": video_id,
                "question": normalize_question(task["question"]) if normalize else task["question"],
                "options": {key: (normalize_answer(value) if normalize else value) for key, value in task.items() if
                            key.startswith("option")},
                "answer": f"option {answer_option_id}"
            })

        return data
    elif "nextqa" in answers_path.lower():
        # load all the data
        raw_data = read_csv_file(file_path=answers_path)
        logger.info(f"Loaded data from {answers_path}.")

        data = []
        for row in raw_data:

            # skip header row
            if row[0] == "video":
                continue

            # there can be multiple questions per video in NExT-QA
            data.append({
                "video_id": row[0],
                "question": normalize_question(row[4]) if normalize else row[4],
                "options": {
                    "option 0": normalize_answer(row[8]) if normalize else row[8],
                    "option 1": normalize_answer(row[9]) if normalize else row[9],
                    "option 2": normalize_answer(row[10]) if normalize else row[10],
                    "option 3": normalize_answer(row[11]) if normalize else row[11],
                    "option 4": normalize_answer(row[12]) if normalize else row[12]
                },
                "answer": f"option {row[5]}",
            })

        return data
    elif "intentqa" in answers_path.lower():
        # load all the data
        raw_data = read_csv_file(file_path=answers_path)
        logger.info(f"Loaded data from {answers_path}.")

        data = []
        for row in raw_data:

            # skip header row
            if row[0] == "video_id":
                continue

            # there can be multiple questions per video in IntentQA
            data.append({
                "video_id": row[0],
                "question": normalize_question(row[4]) if normalize else row[4],
                "options": {
                    "option 0": normalize_answer(row[8]) if normalize else row[8],
                    "option 1": normalize_answer(row[9]) if normalize else row[9],
                    "option 2": normalize_answer(row[10]) if normalize else row[10],
                    "option 3": normalize_answer(row[11]) if normalize else row[11],
                    "option 4": normalize_answer(row[12]) if normalize else row[12]
                },
                "answer": f"option {row[5]}",
            })

        return data
    elif "activitynet" in answers_path.lower():
        # load the answers (i.e. ground truths)
        answers = read_json_file(file_path=answers_path)
        logger.info(f"Loaded ground truth answers from {answers_path}.")

        # load the tasks (i.e. questions and options)
        tasks = read_json_file(file_path=tasks_path)
        logger.info(f"Loaded tasks (questions and options) from {tasks_path}.")

        data = []
        for item in answers:
            answer = item['answer']
            q_type = item['type']
            question_id = item['question_id']
            task = [task for task in tasks if task["question_id"] == question_id][0]
            question = normalize_question(task['question']) if normalize else task['question']
            video_name = question_id.rsplit('_', 1)[0]
            data.append({
                "video_id": video_name,
                "question": question,
                "options": {
                    "option 0": "N/A",
                    "option 1": "N/A",
                    "option 2": "N/A",
                    "option 3": "N/A",
                    "option 4": "N/A"
                },
                "answer": normalize_answer(answer) if normalize else answer
            })

        return data
    elif "videovista" in answers_path.lower():
        # load the answers (i.e. ground truths)
        raw_data = read_json_file(file_path=answers_path)
        logger.info(f"Loaded raw data including tasks and answers from {answers_path}.")

        data = []
        for entry in raw_data:
            data.append({
                # remove ".mp4" from the video name
                "video_id": entry["video_name"][:-4],
                "question": entry["Question"],
                "options": {
                    # note that VideoVISTA only has 4 options
                    "option 0": normalize_answer(entry["Answer_Choices"][0]) if normalize else entry["Answer_Choices"][0],
                    "option 1": normalize_answer(entry["Answer_Choices"][1]) if normalize else entry["Answer_Choices"][1],
                    "option 2": normalize_answer(entry["Answer_Choices"][2]) if normalize else entry["Answer_Choices"][2],
                    "option 3": normalize_answer(entry["Answer_Choices"][3]) if normalize else entry["Answer_Choices"][3]
                },
                "answer": f"option {entry['Answer']}"
            })

        return data
    else:
        err_msg = f"Dataset not supported: {answers_path}"
        logger.error(err_msg)
        raise ValueError(err_msg)
