import numpy as np
import os
import sys

from datasets.utils import read_yaml_file, write_json_file, read_json_file


def aggregate_merged_clips(
        latest_experiment_paths,
        experiment_name,
        dataset_name
):
    # each experiment has a "conclusions.json" file
    all_num_iterations = []
    all_merged_clips_before_iterations = []
    all_merged_clips_after_iterations = []
    for exp in latest_experiment_paths:
        split_conclusions = read_json_file(os.path.join(exp, "conclusions.json"))

        split_num_iterations = []
        split_merged_clips_before_iterations = []
        split_merged_clips_after_iterations = []
        for video_id in split_conclusions.keys():
            video_conclusions = split_conclusions[video_id]
            for task in video_conclusions.keys():
                task_conclusion = video_conclusions[task]
                last_iteration_conclusion = list(task_conclusion.values())[-1]

                num_iterations = last_iteration_conclusion["number_of_iterations"]
                # add one because these are actually indices of the iterations
                merged_clips_before = int(list(last_iteration_conclusion["iteration_conclusions"].keys())[0]) + 1
                merged_clips_after = int(list(last_iteration_conclusion["iteration_conclusions"].keys())[-1]) + 1
                split_num_iterations.append(num_iterations)
                split_merged_clips_before_iterations.append(merged_clips_before)
                split_merged_clips_after_iterations.append(merged_clips_after)

        print(
            f"Average number of iterations of split {exp} on {dataset_name}: {sum(split_num_iterations) / len(split_num_iterations)}")
        print(
            f"Average number of merged clips before iterations of split {exp} on {dataset_name}: {sum(split_merged_clips_before_iterations) / len(split_merged_clips_before_iterations)}")
        print(
            f"Average number of merged clips after iterations of split {exp} on {dataset_name}: {sum(split_merged_clips_after_iterations) / len(split_merged_clips_after_iterations)}")
        print("Minimum number of iterations: ", min(split_num_iterations))
        print("Maximum number of iterations: ", max(split_num_iterations))
        print("Minimum number of merged clips before iterations: ", min(split_merged_clips_before_iterations))
        print("Maximum number of merged clips before iterations: ", max(split_merged_clips_before_iterations))
        print("Minimum number of merged clips after iterations: ", min(split_merged_clips_after_iterations))
        print("Maximum number of merged clips after iterations: ", max(split_merged_clips_after_iterations))

        all_num_iterations.extend(split_num_iterations)
        all_merged_clips_before_iterations.extend(split_merged_clips_before_iterations)
        all_merged_clips_after_iterations.extend(split_merged_clips_after_iterations)

    print("====================================")
    print(
        f"Average number of iterations of {experiment_name} on {dataset_name}: {sum(all_num_iterations) / len(all_num_iterations)}")
    print("Minimum number of iterations: ", min(all_num_iterations))
    print("Maximum number of iterations: ", max(all_num_iterations))
    print("====================================")
    print(
        f"Average number of merged clips before iterations of {experiment_name} on {dataset_name}: {sum(all_merged_clips_before_iterations) / len(all_merged_clips_before_iterations)}")
    print("Minimum number of merged clips before iterations: ", min(all_merged_clips_before_iterations))
    print("Maximum number of merged clips before iterations: ", max(all_merged_clips_before_iterations))
    print("====================================")
    print(
        f"Average number of merged clips after iterations of {experiment_name} on {dataset_name}: {sum(all_merged_clips_after_iterations) / len(all_merged_clips_after_iterations)}")
    print("Minimum number of merged clips after iterations: ", min(all_merged_clips_after_iterations))
    print("Maximum number of merged clips after iterations: ", max(all_merged_clips_after_iterations))


def aggregate_accuracies(
        latest_experiment_paths,
        experiment_name,
        dataset_name
):
    # each experiment has an "accuracy.json" file
    accuracies_per_experiment = []
    for exp in latest_experiment_paths:
        # get accuracy of the last iteration
        accuracy = list(read_json_file(os.path.join(exp, "accuracy.json")).values())[-1]

        if isinstance(accuracy, dict):
            num_instances = accuracy["num_total"]
            num_correct = accuracy["num_correct"]
            print(f"Num instances: {num_instances}")
            print(f"Num correct: {num_correct}")
            print(f"Accuracy of {exp} on {dataset_name}: {num_correct / num_instances if num_instances > 0 else 0}")
        else:
            start_video_index = int(exp.split("_from_")[-1].split("_to_")[0])
            end_video_index = int(exp.split("_to_")[-1].split("/")[0])
            num_instances = end_video_index - start_video_index
            num_correct = num_instances * accuracy
            print("WARNING: The accuracy is not a dict, please use the up to date code version, "
                  "otherwise the accuracy could be calculated wrong!")

        accuracies_per_experiment.append((num_correct, num_instances))

    # calculate the weighted average accuracy
    total_num_instances = 0
    total_num_correct = 0
    for num_correct, num_instances in accuracies_per_experiment:
        total_num_correct += num_correct
        total_num_instances += num_instances
    weighted_average_accuracy = total_num_correct / total_num_instances

    print(f"Accuracy of {experiment_name} on {dataset_name}: {weighted_average_accuracy}")


def aggregate_perceptive_data(
        aggregation_type,
        latest_experiment_paths,
        dataset_name,
        model_name,
        model_temperature,
        output_path
):
    # each experiment has a "states" folder, and in there a folder for each task
    latest_experiment_states = [os.path.join(exp, "states") for exp in latest_experiment_paths]
    print(len(latest_experiment_states))

    # each latest experiment state directory contains multiple folders, one for each video id
    all_video_id_paths = []
    for latest_experiment_state in latest_experiment_states:
        video_id_paths = [os.path.join(latest_experiment_state, path) for path in os.listdir(latest_experiment_state)]
        all_video_id_paths.extend(video_id_paths)
    print(len(all_video_id_paths))

    # aggregate the action captions
    data_to_aggregate_per_video_id = {}

    # egoschema has exactly one task per video, so we can just look for the only folder in each video id path
    for video_id_path in all_video_id_paths:
        # get the video id from the path
        video_id = os.path.basename(video_id_path)

        # skip videos for which we already have action captions to speed up
        if video_id in data_to_aggregate_per_video_id:
            continue

        # find the name of the first folder in that path
        # (this works since for EgoSchema we have exactly one task per video
        # and for NextQA we have multiple tasks per video, but the video is always the same)
        task_id = os.listdir(video_id_path)[0]

        # join it with the video id path to get the full path to the action captions
        states_path = os.path.join(video_id_path, task_id)

        # now there could be different json files for the different reasoning levels of the VideoINSTA
        # we are only interested in the first level, since it already contains all action captions
        states = read_json_file(os.path.join(states_path, "0_all.json"))

        # do not take the first clip as it is the root clip which has not everything assigned
        state = states[1]

        if aggregation_type == "action_captions":
            try:
                action_captions = states[0]["spatial_clip_state"]["action_captions"]
            except KeyError:
                action_captions = state["universal_state"]["action_captions"]
                print(f"WARNING: You probably use a deprecated experiment which you want to aggregate...")

            if len(action_captions) == 0:
                spatial_clip_state = state.get("unconditioned_spatial_clip_state", None)
                if spatial_clip_state is None:
                    spatial_clip_state = state.get("spatial_clip_state", None)

                action_captions = spatial_clip_state.get("action_captions", None)

            # make list if it's a dict
            if isinstance(action_captions, dict):
                action_captions = list(action_captions.values())

            data_to_aggregate_per_video_id[video_id] = action_captions
        elif aggregation_type == "object_detections":
            try:
                # deprecated
                spatial_clip_state = state.get("unconditioned_spatial_clip_state", None)
                if spatial_clip_state is None:
                    spatial_clip_state = state.get("spatial_clip_state", None)

                object_detections = spatial_clip_state.get("unspecific_object_detections", None)
                if object_detections is None:
                    object_detections = spatial_clip_state.get("object_detections", None)

                # make a list of lists (because each interval key contains a list of objects)
                data_to_aggregate_per_video_id[video_id] = list(object_detections.values())
            except AttributeError:
                # recent, use root state
                state = states[0]

                spatial_clip_state = state.get("spatial_clip_state")
                object_detections = spatial_clip_state.get("object_detections")

                data_to_aggregate_per_video_id[video_id] = object_detections
        elif aggregation_type == "summaries":
            try:
                # deprecated
                # consider all clips except from the root clip
                non_root_states = states[1:]

                # get the clip boundaries
                clip_boundaries = [
                    (state["video_clip"]["sampled_indices"][0], state["video_clip"]["sampled_indices"][-1])
                    for state in non_root_states]

                # get the whole video summary
                video_summary = non_root_states[0]["universal_state"]["whole_video_summary"]

                # get the clip-wise summaries
                action_caption_summaries = [state["spatial_clip_state"]["action_captions_summary"] for state in
                                            non_root_states]
                specific_object_detection_summaries = [state["spatial_clip_state"]["specific_object_detections_summary"]
                                                       for
                                                       state in non_root_states]
                unspecific_object_detection_summaries = [
                    state["spatial_clip_state"]["unspecific_object_detections_summary"]
                    for state in non_root_states]
                temporal_grounding_summaries = [state["temporal_clip_state"].get("temporal_grounding_summary", "") for
                                                state
                                                in non_root_states]

                assert len(action_caption_summaries) == len(specific_object_detection_summaries) == len(
                    unspecific_object_detection_summaries) == len(temporal_grounding_summaries), \
                    "The number of summaries per clip is not the same for all clips! The data is corrupted!"

                data_to_aggregate_per_video_id[video_id] = {
                    "clip_boundaries": clip_boundaries,
                    "whole_video_summary": video_summary,
                    "action_caption_summaries": action_caption_summaries,
                    "specific_object_detection_summaries": specific_object_detection_summaries,
                    "unspecific_object_detection_summaries": unspecific_object_detection_summaries,
                    "temporal_grounding_summaries": temporal_grounding_summaries
                }

            except KeyError:
                # use the new structure

                # get the root state
                root_state = states[0]

                # get the summaries from the root clip
                root_action_captions_summary = root_state["spatial_clip_state"]["action_captions_summary"]
                root_object_detections_summary = root_state["spatial_clip_state"]["object_detections_summary"]
                root_temporal_grounding_summary = root_state["temporal_clip_state"]["temporal_grounding_summary"]

                # get the non-root states
                non_root_states = states[1:]

                # get the clip boundaries
                clip_boundaries = [
                    (state["video_clip"]["sampled_indices"][0], state["video_clip"]["sampled_indices"][-1])
                    for state in non_root_states]

                # get the clip-wise summaries
                action_caption_summaries = [state["spatial_clip_state"]["action_captions_summary"] for
                                            state in non_root_states]
                object_detections_summaries = [state["spatial_clip_state"]["object_detections_summary"] for
                                               state in non_root_states]
                temporal_grounding_summaries = [state["temporal_clip_state"].get("temporal_grounding_summary", "") for
                                                state in non_root_states]

                data_to_aggregate_per_video_id[video_id] = {
                    "clip_boundaries": clip_boundaries,
                    "root_action_caption_summary": root_action_captions_summary,
                    "root_object_detections_summary": root_object_detections_summary,
                    "root_temporal_grounding_summary": root_temporal_grounding_summary,
                    "action_caption_summaries": action_caption_summaries,
                    "object_detections_summaries": object_detections_summaries,
                    "temporal_grounding_summaries": temporal_grounding_summaries
                }

        else:
            raise ValueError(f"Unknown aggregation type {aggregation_type}")

    if aggregation_type == "action_captions":
        output_file_name = f"{aggregation_type}_{dataset_name}_{model_name}_{model_temperature}.json"
    elif aggregation_type == "object_detections":
        n = len(list(data_to_aggregate_per_video_id.values())[0][0])
        output_file_name = f"{aggregation_type}_{dataset_name}_{model_name}_{n}_{model_temperature}.json"
    elif aggregation_type == "summaries":
        n = len(list(data_to_aggregate_per_video_id.values())[0]["clip_boundaries"])
        output_file_name = f"{aggregation_type}_{dataset_name}_{model_name}_{model_temperature}_{n}clips.json"
    else:
        raise ValueError(f"Unknown aggregation type {aggregation_type}")

    # save the data to a json file
    write_json_file(data_to_aggregate_per_video_id, os.path.join(output_path, output_file_name))

    print(f"Aggregated {aggregation_type} data for {len(data_to_aggregate_per_video_id)} videos")
    print(f"The minimum number of inferences of a video is "
          f"{min([len(captions) for captions in data_to_aggregate_per_video_id.values()])}")
    print(f"The maximum number of inferences of a video is "
          f"{max([len(captions) for captions in data_to_aggregate_per_video_id.values()])}")
    print(f"Saved {aggregation_type} data to {output_path}{output_file_name}")

    if aggregation_type == "summaries":
        print(f"The minimum number of summaries of a video is "
              f"{min([len(entry['clip_boundaries']) for entry in data_to_aggregate_per_video_id.values()])}")
        print(f"The maximum number of summaries of a video is "
              f"{max([len(entry['clip_boundaries']) for entry in data_to_aggregate_per_video_id.values()])}")


def aggregate_temporal_grounding_variance(
        latest_experiment_paths,
        experiment_name,
        dataset_name
):
    # each experiment has a "states" folder, and in there a folder for each task
    latest_experiment_states = [os.path.join(exp, "states") for exp in latest_experiment_paths]
    print(len(latest_experiment_states))

    # each latest experiment state directory contains multiple folders, one for each video id
    all_video_id_paths = []
    for latest_experiment_state in latest_experiment_states:
        video_id_paths = [os.path.join(latest_experiment_state, path) for path in os.listdir(latest_experiment_state)]
        all_video_id_paths.extend(video_id_paths)
    print(len(all_video_id_paths))

    # aggregate the temporal grounding variances
    all_temporal_grounding_variances = {
        "foreground_variance": [],
        "relevance_variance": [],
        "salience_variance": []
    }

    # egoschema has exactly one task per video, so we can just look for the only folder in each video id path
    for video_id_path in all_video_id_paths:

        # find all task paths in the video id path
        task_paths = [os.path.join(video_id_path, path) for path in os.listdir(video_id_path)]

        # iterate through all tasks
        for task_path in task_paths:

            # now there could be different json files for the different reasoning levels of the VideoINSTA
            # we are only interested in the first level, since it already contains all action captions
            states = read_json_file(os.path.join(task_path, "0_all.json"))

            # use all states except the root clip
            states = states[1:]

            foreground_ratios = []
            relevance_ratios = []
            salience_ratios = []
            for state in states:
                temporal_grounding_summary = state["temporal_clip_state"]

                # get the three variables we want to compare in their variance
                foreground_ratio = temporal_grounding_summary["foreground_ratio"]
                relevance_ratio = temporal_grounding_summary["relevance_ratio"]
                salience_ratio = temporal_grounding_summary["salience_ratio"]

                # append the values to the lists
                foreground_ratios.append(foreground_ratio)
                relevance_ratios.append(relevance_ratio)
                salience_ratios.append(salience_ratio)

            # calculate the variance of the three variables
            foreground_variance = np.var(foreground_ratios)
            relevance_variance = np.var(relevance_ratios)
            salience_variance = np.var(salience_ratios)

            # remember the variances
            all_temporal_grounding_variances["foreground_variance"].append(foreground_variance)
            all_temporal_grounding_variances["relevance_variance"].append(relevance_variance)
            all_temporal_grounding_variances["salience_variance"].append(salience_variance)

    # calculate the average variances
    average_foreground_variance = np.mean(all_temporal_grounding_variances["foreground_variance"])
    average_relevance_variance = np.mean(all_temporal_grounding_variances["relevance_variance"])
    average_salience_variance = np.mean(all_temporal_grounding_variances["salience_variance"])

    # print the results
    print(f"Average foreground variance per task of {experiment_name} on {dataset_name}: {average_foreground_variance}")
    print(f"Average relevance variance per task of {experiment_name} on {dataset_name}: {average_relevance_variance}")
    print(f"Average salience variance per task of {experiment_name} on {dataset_name}: {average_salience_variance}")


# usage for action caption aggregation: python aggregate_experiment_results.py "action_captions" "experiments/exp3/best/eval" "extract_action_captions_egoschema_lavila_t10" "./"
# usage for object detection aggregation: python aggregate_experiment_results.py "object_detections" "experiments/exp3/best/eval" "extract_object_detections_egoschema_cogagent_n3_t0" "./"
# usage for summary aggregation: python aggregate_experiment_results.py "summaries" "experiments/exp3/best/eval" "best_05_02_chatgpt35-1106" "./"
# usage for accuracy calculation: python aggregate_experiment_results.py "accuracy" "experiments/exp3/best/eval" "best_05_02_chatgpt35-1106" "./"
# usage for number of merges calculation: python aggregate_experiment_results.py "merged_clips" "experiments/exp3/best/eval" "best_05_02_chatgpt35-1106" "./"
# usage for temporal grounding variance: python aggregate_experiment_results.py "temporal_grounding_variance" "experiments/exp3/best/eval" "best_05_02_chatgpt35-1106" "./"
if __name__ == '__main__':
    print("WARNING: This will just aggregate the results of the MOST RECENT executions af an experiment with the same "
          "configuration yaml file name. Please consider this before running this script! Are yu sure you want to "
          "continue? (y/n)")

    if input() != "y":
        print("Aborted.")
        sys.exit(0)

    # input 1: what should be aggregated?
    agg_type = sys.argv[1]

    # input 1: base-path to experiment dir, e.g. ".\\experiments\\exp3\\best\\eval"
    base_path = sys.argv[2]

    # input 2: experiment base name, e.g. "best_04_29_chatgpt35-1106"
    experiment_base_name = sys.argv[3]
    # experiment_base_name = "best_05_01_chatgpt35-1106"

    # input 3: save-path to save the aggregated action captions, e.g. ".\\"
    save_path = sys.argv[4]
    # save_path = ".\\"

    # find all directories in the base path that start with the experiment base name, i.e. the different splits
    experiment_dirs = []
    for item in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, item)) and item.startswith(f"{experiment_base_name}_from"):
            experiment_dirs.append(os.path.join(base_path, item))
    print(experiment_dirs)

    # find the latest experiment in each directory (the folder names are timestamps)
    latest_experiments = []
    for experiment_dir in experiment_dirs:
        timestamps = os.listdir(experiment_dir)
        latest_timestamp = max(timestamps)
        latest_experiments.append(os.path.join(experiment_dir, latest_timestamp))
    print(latest_experiments)

    # lookup parameters of interest in the configuration yaml files
    dataset_paths = []
    models = []
    temperatures = []
    for latest_experiment in latest_experiments:
        # get the config
        config = read_yaml_file(os.path.join(latest_experiment, f"{experiment_base_name}.yaml"))

        # lookup parameters
        dataset_paths.append(config["tasks_path"])
        if agg_type == "object_detections":
            models.append(config["api"]["get_unspecific_objects_from_video_clip"]["model_id"].lower().split("/")[-1])
            temperatures.append(config["api"]["get_unspecific_objects_from_video_clip"]["temperature"])
        elif agg_type == "action_captions":
            models.append(config["api"]["get_action_captions_from_video_clip"]["model_name"].lower())
            temperatures.append(config["api"]["get_action_captions_from_video_clip"]["temperature"])
        elif agg_type == "accuracy" or agg_type == "merged_clips" or agg_type == "temporal_grounding_variance":
            models = ["whatever"]
            temperatures = ["whatever"]
        elif agg_type == "summaries":
            temperatures.append(config["operations"]["split"]["class"].lower())
            models.append(config["api"]["get_completion_from_text"]["llm_name"].lower())
        else:
            raise ValueError(f"Unknown aggregation type {agg_type}")
    if len(set(dataset_paths)) != 1 or len(set(models)) != 1 or len(
            set(temperatures)) != 1:
        print(set(dataset_paths))
        print(set(models))
        print(set(temperatures))
        raise ValueError("Different dataset paths in the splits! Please check what you are doing!")
    dataset_path = dataset_paths[0]
    model_raw = models[0]
    temperature = temperatures[0]
    # get the name of the dataset
    dataset = ""
    if "egoschema" in dataset_path:
        dataset = "egoschema"
    elif "nextqa" in dataset_path:
        dataset = "nextqa"
    elif "tbd" in dataset_path:
        # TODO add third dataset
        raise NotImplementedError
    if "llama-3" in model_raw:
        model = "llama3"
    elif "gpt-3.5-turbo-0125" in model_raw:
        model = "gpt-3.5-turbo-0125"
    elif "gpt-3.5-turbo-1106" in model_raw:
        model = "gpt-3.5-turbo-1106"
    else:
        model = model_raw
    print(dataset)
    print(dataset_path)
    print(model)
    print(temperature)

    if agg_type == "action_captions" or agg_type == "object_detections" or agg_type == "summaries":
        aggregate_perceptive_data(agg_type, latest_experiments, dataset, model, temperature, save_path)
    elif agg_type == "accuracy":
        aggregate_accuracies(latest_experiments, experiment_base_name, dataset)
    elif agg_type == "merged_clips":
        aggregate_merged_clips(latest_experiments, experiment_base_name, dataset)
    elif agg_type == "temporal_grounding_variance":
        aggregate_temporal_grounding_variance(latest_experiments, experiment_base_name, dataset)
    else:
        raise ValueError(f"Unknown aggregation type {agg_type}")
