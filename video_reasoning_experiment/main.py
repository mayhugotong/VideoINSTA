import argparse

from video_reasoning_experiment.experiment import VideoReasoningVideoINSTAExperiment
from video_reasoning_experiment.utils import create_experiment_dir, parse_config, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        metavar="PATH_TO_CONF_FILE",
        required=True,
        help="relative or absolute path to the configuration file",
    )
    parser.add_argument(
        "--start_video_index",
        type=int,
        metavar="START_VIDEO_INDEX",
        required=False,
        help="the index of the video to start with in the dataset",
    )
    parser.add_argument(
        "--end_video_index",
        type=int,
        metavar="END_VIDEO_INDEX",
        required=False,
        help="the index of the video to end with in the dataset",
    )
    args = parser.parse_args()

    # read configuration file
    configuration_file = args.conf
    configuration = parse_config(configuration_file)

    if args.start_video_index is not None and args.end_video_index is not None:
        configuration["start_video"] = args.start_video_index
        configuration["end_video"] = args.end_video_index

        start = configuration["start_video"]
        end = configuration["end_video"]
    else:
        start = None
        end = None

    # experiment setup
    mode = configuration.get("mode")
    log_lvl = configuration.get("logger").get("level")
    log_fmt = configuration.get("logger").get("format")
    experiment_path = create_experiment_dir(
        conf_file=configuration_file,
        exp_path=configuration.get("experiment_path"),
        run_mode=mode,
        start=start,
        end=end,
    )

    # overwrite overall experiment path with the newly created base_path of the experiment
    configuration["experiment_path"] = experiment_path

    # logger setup
    logger = setup_logger(experiment_path, log_lvl, log_fmt)
    logger.info(
        "Successfully read the given configuration file, created experiment directory and set up logger."
    )
    logger.info(
        f"Starting experiment in mode {mode} using configuration {configuration_file}"
    )

    exp = VideoReasoningVideoINSTAExperiment(configuration)
    exp.conduct(mode=mode)


if __name__ == '__main__':
    main()
