from __future__ import annotations

import logging
import os
import torch

from datasets.utils import create_video_data_from_video_path

logger = logging.getLogger("root")


class VideoClip:
    def __init__(
            self,
            data: torch.Tensor,
            path: os.path,
            original_fps: float,
            original_num_frames: int,
            sampled_fps: float,
            sampled_indices: list[int],
            feature_data: dict[str, torch.Tensor] = None
    ):
        self.data: torch.Tensor = data
        self.feature_data: dict[str, torch.Tensor] = feature_data if feature_data else {}

        self.path: os.path = path
        self.uid: str = self.path_to_uid(path)
        self.id: str = self.path_to_id(path)

        self.original_fps: float = original_fps
        self.original_num_frames: int = original_num_frames

        self.sampled_fps: float = sampled_fps
        self.sampled_indices: list[int] = sampled_indices
        self.sampled_num_frames: int = len(sampled_indices)

        assert self.sampled_num_frames == len(self.sampled_indices) == self.data.shape[0], \
            "Sampled number of frames, length of sampled indices and video data shape do not match"

        logger.info(f"Initialized video clip with uid {self.uid} and id {self.id}.")

    @staticmethod
    def from_metadata(video_clip_data: torch.Tensor, video_clip_metadata: dict,
                      feature_data: dict[str, torch.Tensor] = None) -> VideoClip:
        """
        Creates a new video clip from the video clip data and the video clip metadata.

        :param video_clip_data: Video clip data from which to initialize the VideoClip instance.
        :param video_clip_metadata: Video clip metadata from which to initialize the VideoClip instance.
        :param feature_data: Video clip feature data.
        :return: A new VideoClip instance with properties of the metadata.
        """
        return VideoClip(
            data=video_clip_data,
            path=video_clip_metadata["video_file"],
            original_fps=video_clip_metadata["fps"],
            original_num_frames=video_clip_metadata["frames_total"],
            sampled_fps=video_clip_metadata["sample_rate"],
            sampled_indices=video_clip_metadata["sample_indices"],
            feature_data=feature_data
        )

    @staticmethod
    def path_to_uid(path: os.path) -> str:
        return path.replace("./", "").replace("/", "_").replace(".mp4", "").replace(".avi", "").replace(".mkv", "")

    @staticmethod
    def path_to_id(path: os.path) -> str:
        return str(os.path.basename(path).replace(".mp4", "").replace(".avi", "").replace(".mkv", ""))

    def __str__(self):
        return f"Video Clip:\n    uid: {self.uid}\n    sampled_fps: {self.sampled_fps}\n    sampled_num_frames: {self.sampled_num_frames}\n    sampled_from: {self.sampled_indices[0]}\n    sampled_to: {self.sampled_indices[-1]}"

    def __eq__(self, other):
        return self.uid == other.uid and self.sampled_indices == other.sampled_indices

    def __len__(self):
        assert len(self.sampled_indices) == self.sampled_num_frames == self.data.shape[0], \
            "Sampled number of frames, length of sampled indices and video data shape do not match"
        return self.sampled_num_frames

    def get_resampled_video_clip(self, sample_rate: float = 0.0):
        assert sample_rate > 0.0, "Sample rate must be greater than 0.0"
        assert sample_rate <= self.original_fps, "Sample rate must be smaller or equal to the original fps"

        if sample_rate == self.sampled_fps:
            return self

        start_frame_index = self.sampled_indices[0]
        end_frame_index = self.sampled_indices[-1]

        if sample_rate >= 1.5 and start_frame_index == end_frame_index:
            logger.warning(f"Video clip with uid {self.uid} and id {self.id} has only one frame. "
                           f"Using a trick to resample the video clip through expanding the one frame clip to a clip of"
                           f"new size fps * original_fps, such that this expanded clip captures the prior timeframe of"
                           f"the down-sampled 1-frame representation. "
                           f"This is necessary since the sample rate is greater or equal to 1.5 and therefore more "
                           f"than just one frame needs to be sampled. In general, please avoid single frame clips.")
            # expand the start and end frame index respectively
            start_frame_index -= (self.original_fps * self.sampled_fps) // 2
            end_frame_index += (self.original_fps * self.sampled_fps) // 2

            # but still make sure to have valid indices within the total number of video frames
            start_frame_index = max(start_frame_index, 0)
            end_frame_index = min(end_frame_index, self.original_num_frames - 1)

        logger.debug(f"Resampling video clip with uid {self.uid} and id {self.id} from "
                     f"{self.sampled_fps} to {sample_rate} using"
                     f"start_frame_index {start_frame_index} and end_frame_index {end_frame_index} + 1")

        # resample the video clip data tensor (i.e. [batch, channel, height, width])
        # (+ 1 because the end index is exclusive, but we took the real sampled index before)
        new_data, new_metadata = create_video_data_from_video_path(video_path=self.path,
                                                                   sample_rate=sample_rate,
                                                                   window_start_index=start_frame_index,
                                                                   window_end_index=end_frame_index + 1)

        assert new_metadata["fps"] == self.original_fps, "Resampled video clip has wrong fps"
        assert new_metadata["frames_total"] == self.original_num_frames, \
            "Resampled video clip has wrong number of frames"

        # resample the feature_data if available
        new_feature_data = {}
        if self.feature_data:
            for feature_name, feature_data in self.feature_data.items():
                # TODO implement resampling of feature data when required
                new_feature_data[feature_name] = feature_data

        logger.info(f"Resampled video clip with uid {self.uid} and id "
                    f"{self.id} from {self.sampled_fps} to {sample_rate}.")

        return VideoClip(
            data=new_data,
            path=self.path,
            original_fps=self.original_fps,
            original_num_frames=self.original_num_frames,
            sampled_fps=sample_rate,
            sampled_indices=new_metadata["sample_indices"],
            feature_data=new_feature_data
        )

    def get_merged_video_clip(self, other: VideoClip) -> VideoClip:
        """
        Merges the current video clip with the other video clip by concatenating the video clip data
        and adapting all parameters that depend on it.

        :param other: The other video clip to merge with.
        :type other: VideoClip
        :return: A new video clip with concatenated video clip data and concatenated sampled indices.
        :rtype: VideoClip
        """
        assert self.path == other.path, "Video clip paths do not match"
        assert self.uid == other.uid, "Video clip uids do not match"
        assert self.original_fps == other.original_fps, "Video clip original fps do not match"
        assert self.original_num_frames == other.original_num_frames, \
            "Video clip original number of frames do not match"

        # concatenate the video clip data tensors (i.e. [batch, channel, height, width])
        new_data = torch.cat((self.data, other.data), dim=0)

        # concatenate the list about sampled indices
        new_sampled_indices = self.sampled_indices + other.sampled_indices

        # concatenate features if available
        new_feature_data = {}
        if self.feature_data:
            for feature_name, feature_data in self.feature_data.items():
                new_feature_data[feature_name] = torch.cat((feature_data, other.feature_data[feature_name]), dim=0)

        # get new sampled fps
        new_sampled_fps = min(self.sampled_fps, other.sampled_fps)

        logger.info(f"Merged video clip with uid {self.uid} and id {self.id} from "
                    f"{self.sampled_indices[0]} - {self.sampled_indices[-1]} with "
                    f"video clip with uid {other.uid} and id {other.id} from "
                    f"{other.sampled_indices[0]} - {other.sampled_indices[-1]} to "
                    f"{new_sampled_indices[0]} - {new_sampled_indices[-1]}.")

        return VideoClip(
            data=new_data,
            path=self.path,
            original_fps=self.original_fps,
            original_num_frames=self.original_num_frames,
            sampled_fps=new_sampled_fps,
            sampled_indices=new_sampled_indices,
            feature_data=new_feature_data
        )

    def get_json_representation(self) -> dict:
        return {
            "uid": self.uid,
            "id": self.id,
            "path": self.path,
            "original_fps": self.original_fps,
            "original_num_frames": self.original_num_frames,
            "sampled_fps": self.sampled_fps,
            "sampled_indices": self.sampled_indices
        }
