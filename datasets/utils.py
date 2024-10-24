import csv
import json
import logging
import time

import numpy as np
import os.path
import re
import torch
import torchvision
import yaml

from decord import VideoReader, cpu
from tqdm import tqdm

logger = logging.getLogger("root")


def create_video_segments_data_from_video_path(video_path: str,
                                               sample_rate_scale: int = 1,
                                               new_width: int = 384,
                                               new_height: int = 384,
                                               clip_len: int = 8,
                                               num_segments: int = 1):
    """
    This function calculates a down-sampled representation of a video.
    It does so by splitting the video in #num_segments equally sized segments.
    There are #duration frames sampled from each of these segments.
    Moreover, all these sampled frames can be scaled by sample_rate_scale.

    E.g., if num_segments=10, then a video with 642 frames and 60 fps will be divided into 10 segments each containing
    64 frames. Since this video lasts 642 / 60 = 10.7 seconds which will be floored to 10.0, there is no need to clip
    this value. This is because this value will be used as the number of samples per segment and if the video duration
    is less than 8 seconds, it will always be clipped to 8.0 (to avoid too sparse samples). A linear space with this
    number of indices will then be used to determine the indices for each segment. Afterwards, all those indices are
    concatenated and filtered by the given sample_rate_scale. If it is 1 (per default), all sampled indices will be used
    for sampling. In this example this would result in the following frames:
    [0. 7.1111111   14.2222222  ... 63. 70.1111111  77.2222222  ... 639.]
    With sample_rate_scale = 2.0 this yields (only every second entry from the array above):
    [0. 14.2222222  28.4444444 ... 56.888888   70.1111111   ... 639.]

    Inspired by https://github.com/OpenGVLab/Ask-Anything/blob/long_video_support/video_chat/util.py.

    :param video_path: Path to video.
    :param sample_rate_scale: Number of indices to sample from the array with all indices.
    :param new_width: Scaling of the frame width.
    :param new_height: Scaling of the frame height.
    :param clip_len: Minimum number of samples per segment.
    :param num_segments: Number of segments to sample.
    :return: Numpy video data and JSON metadata.
    """
    # read the video file and get metadata
    vr = VideoReader(video_path, width=new_width, height=new_height, num_threads=1, ctx=cpu(0))
    fps = vr.get_avg_fps()
    num_frames_total = len(vr)

    # calculate number of frames per segment (round to floor)
    num_frames_segment = num_frames_total // num_segments

    # calculate duration of the video in seconds
    duration = int(max(num_frames_total // fps, clip_len))

    # use floored duration of the video as number of samples per segment
    samples_per_segment = duration

    # sample as many indices per segment as the video duration
    all_index = []
    for i in range(num_segments):
        index = np.linspace(0, num_frames_segment, num=samples_per_segment)
        index = np.clip(index, 0, num_frames_segment - 1).astype(np.int64)
        index = index + i * num_frames_segment
        all_index.extend(list(index))

    # scale all indices by the given sample rate
    all_index = all_index[::sample_rate_scale]

    # sample the frames
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()

    # create important metadata
    metadata = {
        "video_file": video_path,
        "sample_rate_scale": sample_rate_scale,
        "fps": fps,
        "frames_total": num_frames_total,
        "sample_indices": all_index
    }

    # reshape the video data to dimensions (batch, channel, height, width) for convention reasons
    vide_data = torch.from_numpy(buffer)
    vide_data = vide_data.permute(0, 3, 1, 2)

    return vide_data, metadata


def create_video_data_from_video_path(video_path: str,
                                      sample_rate: float = 0.0,
                                      window_start_index: int = 0,
                                      window_end_index: int = 0) -> (torch.Tensor, dict):
    """
    This function calculates a down-sampled representation of a video.
    First, it applies the given sample_rate to all video indices and divides them into equally sized intervals.
    Second, it filters out all indices that are not part of the window specified by window_start_index and
    window_end_index. Third, it samples the remaining indices from the video file.
    Samples the whole video per default.

    :param video_path: Path to video.
    :param sample_rate: Number of fps to sample. If set to 0.0, then all frames are sampled.
    :param window_start_index: Index to start sampling from (inclusive, start from beginning if 0).
    :param window_end_index: Index from where to stop sampling (exclusive, stop at video end if 0).
    :return: Numpy video data and JSON metadata.
    """
    # read video using one thread and cpu
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))

    # if audio is needed, compare https://github.com/dmlc/decord#avreader
    # avr = AVReader(video_path, num_threads=1, ctx=cpu(0), sample_rate=22050)

    # handle temporal window / segment of video
    fps = vr.get_avg_fps()
    num_frames_total = len(vr)

    # makes sure the start index is not negative and smaller or equal to the end index
    assert 0 <= window_start_index <= window_end_index, "Invalid window_start_index"

    # make sure the end index is greater than the start index and smaller or equal to the total number of frames
    assert window_start_index <= window_end_index <= num_frames_total, "Invalid window_end_index"

    # make sure the sample rate is greater or equal to 0.0 and smaller or equal to the fps
    assert 0 <= sample_rate <= fps, "Invalid sample_rate"

    # set correct end index if it is 0
    window_end_index = window_end_index if window_end_index > 0 else num_frames_total

    # calculate the number of frames inside the window
    num_frames_window = window_end_index - window_start_index

    # calculate the number of samples to be sampled from the video
    num_samples = round(num_frames_window / (fps / sample_rate)) if sample_rate != 0.0 else num_frames_window

    # sample all frames if sample rate is set to 0.0
    num_samples = num_samples

    if num_samples == 0:
        logger.warning("There is no sample in the specified window. This may happen if the sample rate is too sparse "
                       "or the window is too short. Please consider to pre-process with a larger sample rate again. "
                       "Sampling just one frame from the window now...")
        num_samples = 1

    # get a list with total_num_samples many equally distant indices from the video start to the video end
    # subtract 1 from window_end_index to make sure the sampling is exclusive the end index
    total_sample_indices = np.linspace(window_start_index, window_end_index - 1, num=num_samples).astype(int).tolist()
    logger.debug(f"total_sample_indices: {total_sample_indices}")

    # sample the images at the given indices from the video
    buffer = vr.get_batch(total_sample_indices)
    buffer = buffer.asnumpy()

    # create important metadata
    metadata = {
        "video_file": video_path,
        "sample_rate": sample_rate,
        "fps": fps,
        "frames_total": num_frames_total,
        "sample_indices": total_sample_indices
    }

    # reshape the video data to dimensions (batch, channel, height, width) for convention reasons
    vide_data = torch.from_numpy(buffer)
    vide_data = vide_data.permute(0, 3, 1, 2)

    return vide_data, metadata


def create_video_data_from_video_frames_path(video_frames_path: os.path,
                                             sample_rate: float = 0.0,
                                             start_index: int = 0,
                                             end_index: int = 0,
                                             frame_file_format: str = "png"):
    # read metadata file
    metadata = read_json_file(os.path.join(video_frames_path, "metadata.json"))

    # get metadata associated with the given sample_rate
    try:
        metadata = metadata[str(sample_rate)]
    except KeyError as e:
        raise NotPreProcessedWithGivenSampleRateError(
            "There is no entry for the given sample rate in the metadata.json "
            "file. Please make sure you have already pre-processed your data "
            "with the corresponding sample_rate of your needs. This error is "
            f"is caused by the KeyError trying to read sample_rate={e} from "
            "the metadata.json file.")

    # get original sample_rate
    original_sample_rate = metadata["sample_rate"]

    # prevent sample_rate related mismatching
    if original_sample_rate != 0.0 and original_sample_rate != sample_rate:
        raise SampleRateMismatchError(
            "Original sample rate (of the pre-processed persisted frames): "
            f"{original_sample_rate}. Requested sample rate (to be sampled from the frames): "
            f"{sample_rate}. Make sure to pre-process your data with the same sample_rate "
            "first. In cases where the original_sample rate was 0.0, other sample_rates are "
            "accepted. There may be a fitting implementation for cases original_sample_rate "
            "> sample_rate in the future.")

    # get original sample indices
    original_sample_indices = metadata["sample_indices"]

    # get image files of the video frames
    frame_filenames = os.listdir(video_frames_path)

    # remove metadata.json file from the list
    frame_filenames.remove("metadata.json")

    # extract all indices from the filenames
    file_indices = [int(frame_filename.split(".")[0]) for frame_filename in frame_filenames]

    # filter by start and end indices (select the specified window)
    start_index = start_index if start_index >= 0 else 0
    end_index = end_index if end_index > 0 else metadata["frames_total"] - 1

    # just use indices associated with the original sample rate (implicitly through the original sample indices) and
    # slice the indices to the specified window margins
    filtered_file_indices = [index for index in file_indices
                             if index in original_sample_indices and (start_index <= index <= end_index)]

    # assure correct temporal order
    filtered_file_indices.sort()

    if len(filtered_file_indices) == 0:
        raise NoSampleInSpecifiedWindowError(
            "There is no sample in the specified window. This may happen if the sample rate is too sparse. "
            "Please consider to pre-process with a larger sample rate again."
        )

    # recreate needed filenames
    frame_filenames = [os.path.join(video_frames_path, f"{str(file_index)}.{frame_file_format}")
                       for file_index in filtered_file_indices]

    # read the image files and stack resulting image 3d tensors together to the 4d video tensor
    video_data = torch.stack([torchvision.io.read_image(file) for file in frame_filenames])

    return video_data, metadata


def video_collate_fn(batch_data: any):
    """
    This function defines how to represent the differently sized samples.
    In the case of video data of different lengths, a batch will therefore be represented as a list of the tensor data.
    For the NLQ task and its corresponding datasets this means that a batch consists of tuples of the form
    (video, annotations). So a batch is like [(video-1, annotations-1), ..., (video-n, annotations-n)].
    Note that each entry of each tuple in the batch can have a different size or shape.

    :param batch_data: Annotated video data of the shape (video, annotations).
    :return: Batch of annotated video data of the shape [(video-1, annotations-1), ..., (video-n, annotations-n)].
    """
    return list(batch_data)


def save_video_as_frames(video_path: str, root_dir: str, sample_rate: float = 0.0):
    """
    This IO-function reads the video data from a video file and converts frame by frame (depending on the sample_rate)
    into images that are being stored in a directory that is called like the original video file. The image files are
    stored in .png-format and have their frame index as their name.

    :param video_path: Path to the video file.
    :param root_dir: Root directory in which to save the video directory that holds all frame image files.
    :param sample_rate: Number of fps to sample. If set to 0.0, then all frames are sampled.
    """
    video_data, metadata = create_video_data_from_video_path(video_path=video_path, sample_rate=sample_rate)
    video_uid = os.path.basename(video_path).split(".")[0]
    save_path = os.path.join(root_dir, video_uid)
    frame_indices = metadata["sample_indices"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(video_data.size(0)), desc=f"Converting video {video_uid}"):
        frame = video_data[i, :, :, :].float() / 255
        torchvision.utils.save_image(frame, os.path.join(save_path, f"{frame_indices[i]}.png"))

    # append possibly existing metadata.json file with new metadata indicated by the sample_rate as key
    metadata_filename = os.path.join(save_path, "metadata.json")
    content_before = read_json_file(metadata_filename) if os.path.exists(metadata_filename) else {}
    new_entry = {
        str(metadata["sample_rate"]): metadata
    }
    content = content_before | new_entry
    write_json_file(content, metadata_filename)


def average_to_fixed_length(visual_input, num_sample_clips):
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


def read_json_file(file_path: os.path):
    """
    This function reads a json file.
    
    :param file_path: Path to the .json file.
    :return: Json data as dictionary.
    """
    with open(file_path) as f:
        data = json.load(f)
    return data


def write_json_file(data: any, file_path: os.path):
    """
    This function writes a json file.

    :param data: Json data as dictionary.
    :param file_path: Path to the .json file.
    """
    with open(file_path, "w") as outfile:
        json.dump(data, outfile)


def read_csv_file(file_path: os.path):
    """
    This function reads a csv file.

    :param file_path: Path to the .csv file.
    :return: Csv data as list of dictionaries.
    """
    data = []
    with open(file=file_path, mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            data.append(line)
    return data


def read_yaml_file(file_path: os.path):
    """
    This function reads a yaml file.

    :param file_path: Path to the .yaml file.
    :return: Yaml data as dictionary.
    """
    with open(file_path, "r") as stream:
        conf = yaml.safe_load(stream)
    return conf


def slugify(value: str):
    """
    This function converts a string into a slug.

    :param value: String to be converted into a slug.
    :return: Slug.
    """
    pattern = r'[^\w\-_.\s]'
    slug = re.sub(pattern, '', value)
    slug = slug.lower().replace(" ", "_")

    # limit the slug to 50 characters
    slug = slug[:50]

    # add timestamp to the slug to avoid overwriting
    slug = f"{slug}_{int(time.time())}"

    return slug


def normalize_question(question: str):
    # make sure the first letter is uppercase
    question = question[0].upper() + question[1:]

    # make sure the question ends with a question mark or a dot, and if not add a question mark
    if question[-1] not in ["?", "."]:
        question = question + "?"

    return question


def normalize_answer(answer: str):
    # make sure the first letter is uppercase
    answer = answer[0].upper() + answer[1:]

    # make sure the answer ends with a dot, and if not add a dot
    if answer[-1] != ".":
        answer = answer + "."

    return answer


class SampleRateMismatchError(Exception):
    pass


class NotPreProcessedWithGivenSampleRateError(Exception):
    pass


class NoSampleInSpecifiedWindowError(Exception):
    pass
