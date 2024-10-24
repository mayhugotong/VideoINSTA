import logging
import math
import torch

from api.api import API
from transformers import CLIPImageProcessor
from typing import Optional
from video_reasoning.state.clip import ClipState
from video_reasoning.state.spatial import SpatialClipState
from video_reasoning.state.temporal import TemporalClipState
from video_reasoning.state.video import VideoClip
from video_reasoning.structure.videoinsta import Clip, VideoINSTA, Relation
from video_reasoning.operation.base import Operation

logger = logging.getLogger("root")


class Split(Operation):
    def __init__(self, num_splits: int = 4):
        super().__init__()
        self.num_splits = num_splits

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target: Optional[Clip]) -> None:
        source_video_clip = target.state.video_clip

        indices = torch.tensor(list(range(len(source_video_clip))))
        splits = torch.chunk(indices, self.num_splits)

        target_clips = []
        for split in splits:
            # get the start and end index of the split
            start_list_index = split[0].item()
            end_list_index = split[-1].item()

            # trim the video data to the split
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]

            # trim the sampled indices to the split
            split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

            # create a new video clip from the split
            split_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )

            # create a new clip with new state for the new clip
            new_clip = Clip(
                state=ClipState(
                    video_clip=split_video_clip,
                    task=target.state.task,
                    lexical_representation=target.state.lexical_representation,
                    spatial_clip_state=SpatialClipState(
                        video_clip=split_video_clip,
                        task=target.state.spatial_clip_state.task,
                        use_action_captions=target.state.spatial_clip_state.use_action_captions,
                        use_object_detections=target.state.spatial_clip_state.use_object_detections,
                        use_action_captions_summary=target.state.spatial_clip_state.use_action_captions_summary,
                        use_object_detections_summary=target.state.spatial_clip_state.use_object_detections_summary,
                        lexical_representation=target.state.spatial_clip_state.lexical_representation
                    ),
                    temporal_clip_state=TemporalClipState(
                        video_clip=split_video_clip,
                        task=target.state.temporal_clip_state.task,
                        use_temporal_grounding_summary=target.state.temporal_clip_state.use_temporal_grounding_summary,
                        lexical_representation=target.state.temporal_clip_state.lexical_representation,
                        use_relevance=target.state.temporal_clip_state.use_relevance,
                        use_foreground=target.state.temporal_clip_state.use_foreground,
                        use_salience=target.state.temporal_clip_state.use_salience
                    )
                )
            )

            target_clips.append(new_clip)

        # apply the split to the structure
        relations = [Relation(source=source_clip, target=target_clip)
                     for target_clip in target_clips
                     for source_clip in [target]]

        structure.add_clips(target_clips)
        structure.add_relations(relations)

        logger.info(f"Executed structure split operation: Split")

    def __str__(self):
        return f"Split(num_splits={self.num_splits})"


class DPCKNNSplit(Operation):
    def __init__(
            self,
            num_clusters: int = 4,
            k: int = 5,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            use_density_for_score: bool = True,
            use_density_minimum_as_border: bool = False,
            reset_seed: bool = False
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.k = k
        self.clip_model_name = clip_model_name
        self.use_density_for_score = use_density_for_score
        self.use_density_minimum_as_border = use_density_minimum_as_border
        self.reset_seed = reset_seed

    def _execute(self, structure: Optional[VideoINSTA], api: Optional[API], target: Optional[Clip]) -> None:
        # treat the split as an api function regarding seed reset since it is sensitive to non-deterministic behavior
        # this might cause conflicts with summaries that have been extracted before this seed reset was added,
        # that's why it can be controlled by the hyperparameter reset_seed (defaults to False since that was the
        # behavior before the seed reset was added)
        if self.reset_seed:
            api.reset_seed()

        # get the state of the source clip
        source_clip_state = target.state

        # get the video clip of the state of the source clip
        source_video_clip = source_clip_state.video_clip

        video = source_video_clip.data

        # the following cluster center selection using DPCKNN is inspired by:
        # https://github.com/PKU-YuanGroup/Chat-UniVi
        # 1. get image features
        encoder = CLIPImageProcessor.from_pretrained(self.clip_model_name)
        inputs = encoder(images=video, return_tensors="pt")
        video_features = inputs["pixel_values"]
        logger.debug(f"video_features.shape: {video_features.shape}")
        # num_frames, num_channels, 224, 224

        # 2. transform image features to token features, i.e. flatten to patches of size 1024
        features = video_features.view(video.size(0), -1, 1024)
        logger.debug(f"features.shape: {features.shape}")
        # num_frames, (num_channels * 224 * 224) / 1024, 1024

        # 3. get mean along the second dimension, i.e. mean pooling
        cls_features = torch.mean(features, dim=1, keepdim=False).unsqueeze(0).clone()
        token_dict = {'x': cls_features,
                      'token_num': cls_features.size(1),
                      'idx_token': torch.arange(cls_features.size(1))[None, :].repeat(cls_features.size(0), 1),
                      'agg_weight': cls_features.new_ones(cls_features.size(0), cls_features.size(1), 1),
                      'mask': None}
        logger.debug(f"cls_features.shape: {cls_features.shape}")
        logger.debug(f"token_dict['idx_token'].shape: {token_dict['idx_token'].shape}")
        logger.debug(f"token_dict['agg_weight'].shape: {token_dict['agg_weight'].shape}")

        # 4. cluster tokens with DPC-KNN algorithm
        k = len(source_video_clip) if len(source_video_clip) < self.k else self.k
        num_clusters = len(source_video_clip) if len(source_video_clip) < self.num_clusters else self.num_clusters
        idx_cluster, cluster_num, centers, cluster_borders = DPCKNNSplit.cluster_dpc_knn(token_dict,
                                                                                         num_clusters,
                                                                                         k,
                                                                                         token_mask=token_dict["mask"],
                                                                                         use_density_for_score=self.use_density_for_score)
        logger.debug(f"idx_cluster: {idx_cluster}")
        logger.debug(f"cluster_num: {cluster_num}")
        logger.debug(f"cluster_borders: {cluster_borders}")

        # get the cluster centers and sort them in ascending order
        centers = centers[0].tolist()
        centers.sort()
        logger.debug(f"centers: {centers}")

        # remove the first frame from the centers since they will be processed anyway in the next steps
        centers.remove(0) if 0 in centers else None
        logger.debug(f"centers after removing first frame: {centers}")

        # get the borders between the clusters
        borders = [0]
        if self.use_density_minimum_as_border:
            borders.extend(cluster_borders)
            borders.append(video.size(0))
        else:
            # get the borders of the clusters (middle between two centers)
            for i in range(len(centers)):
                if i == len(centers) - 1:
                    borders.append(video.size(0))
                else:
                    # floor the middle between two centers, such that the last frame is still a valid index
                    border = int(math.floor((centers[i + 1] - centers[i]) / 2 + centers[i]))
                    borders.append(border)
        logger.debug(f"boundaries (borders): {borders}")

        target_clips = []
        for i in range(1, len(borders)):
            # get the list indices for the new video clip
            start_list_index = borders[i - 1]
            end_list_index = borders[i]
            logger.debug(f"start_index (relative) of new video clip: {start_list_index}")
            logger.debug(f"end_index (relative) of new video clip: {end_list_index}")

            # trim the video data to the split
            split_data = source_video_clip.data[start_list_index:end_list_index + 1]

            # trim the sampled indices to the split
            split_sampled_indices = source_video_clip.sampled_indices[start_list_index:end_list_index + 1]

            # create a new video clip from the split
            split_video_clip = VideoClip(
                data=split_data,
                path=source_video_clip.path,
                original_fps=source_video_clip.original_fps,
                original_num_frames=source_video_clip.original_num_frames,
                sampled_fps=source_video_clip.sampled_fps,
                sampled_indices=split_sampled_indices
            )

            # create a new clip with new state for the new clip
            new_clip = Clip(
                state=ClipState(
                    video_clip=split_video_clip,
                    task=source_clip_state.task,
                    lexical_representation=source_clip_state.lexical_representation,
                    spatial_clip_state=SpatialClipState(
                        video_clip=split_video_clip,
                        task=source_clip_state.spatial_clip_state.task,
                        use_action_captions=source_clip_state.spatial_clip_state.use_action_captions,
                        use_object_detections=source_clip_state.spatial_clip_state.use_object_detections,
                        use_action_captions_summary=source_clip_state.spatial_clip_state.use_action_captions_summary,
                        use_object_detections_summary=source_clip_state.spatial_clip_state.use_object_detections_summary,
                        lexical_representation=source_clip_state.spatial_clip_state.lexical_representation
                    ),
                    temporal_clip_state=TemporalClipState(
                        video_clip=split_video_clip,
                        task=source_clip_state.temporal_clip_state.task,
                        use_temporal_grounding_summary=source_clip_state.temporal_clip_state.use_temporal_grounding_summary,
                        lexical_representation=source_clip_state.temporal_clip_state.lexical_representation,
                        use_relevance=source_clip_state.temporal_clip_state.use_relevance,
                        use_salience=source_clip_state.temporal_clip_state.use_salience,
                        use_foreground=source_clip_state.temporal_clip_state.use_foreground
                    )
                )
            )

            target_clips.append(new_clip)

        # apply the expansion to the structure
        relations = [Relation(source=source_clip, target=target_clip)
                     for target_clip in target_clips
                     for source_clip in [target]]

        structure.add_clips(target_clips)
        structure.add_relations(relations)

        logger.info(f"Executed structure split operation: DPCKNNSplit")

    @staticmethod
    def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None, use_density_for_score=True):
        """Cluster tokens with DPC-KNN algorithm.
        Source: https://github.com/PKU-YuanGroup/Chat-UniVi
        Return:
            idx_cluster (Tensor[B, N]): cluster index of each token.
            cluster_num (int): actual cluster number. The same with
                input cluster number
        Args:
            token_dict (dict): dict for token information
            cluster_num (int): cluster number
            k (int): number of the nearest neighbor used for local density.
            token_mask (Tensor[B, N]): mask indicate whether the token is
                padded empty token. Non-zero value means the token is meaningful,
                zero value means the token is an empty token. If set to None, all
                tokens are regarded as meaningful.
            use_density_for_score (bool): whether to use density to compute the score or not.
        """
        with torch.no_grad():
            x = token_dict["x"]
            B, N, C = x.shape
            logger.debug(f"x.shape: {x.shape}")

            # compute pairwise distance matrix
            dist_matrix = torch.cdist(x.float(), x.float()) / (C ** 0.5)

            if token_mask is not None:
                token_mask = token_mask > 0
                # in order to not affect the local density, the distance between empty tokens
                # and any other tokens should be the maximal distance.
                dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

            # get local density
            dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
            logger.debug(f"dist_nearest: {dist_nearest.shape}")
            density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
            logger.debug(f"density: {density.shape}")

            if token_mask is not None:
                # the density of empty token should be 0
                density = density * token_mask

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(x.dtype)
            dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            # (case distinction is added by maximotus, not part of the original implementation)
            score = dist * density if use_density_for_score else dist
            _, index_down = torch.topk(score, k=cluster_num, dim=-1)
            logger.debug(f"score: {score}")
            logger.debug(f"index_down: {index_down}")

            # assign tokens to the nearest center
            dist_matrix = DPCKNNSplit.index_points(dist_matrix, index_down)
            idx_cluster = dist_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
            idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

            # make list from index_down tensor
            index_down_list = index_down.tolist()[0]
            index_down_list.sort()
            logger.debug(f"index_down_list: {index_down_list}")

            # get the frame indices of the density minimums between cluster centers as borders
            # (this is added by maximotus, not part of the original implementation)
            borders = []
            for j in range(cluster_num - 1):
                # get the current and next cluster center indices
                current_cluster_center_index = index_down_list[j]
                next_cluster_center_index = index_down_list[j + 1]

                # slice the density tensor to get the density values
                # between the current cluster center and the next cluster center (excluding both)
                density_slice = density[:, current_cluster_center_index + 1:next_cluster_center_index - 1]
                logger.debug(f"density_slice: {density_slice.shape}")

                # get the frame index of the minimum density value
                if density_slice.size(1) == 0:
                    min_density_idx = current_cluster_center_index + 1
                else:
                    min_density_idx = density_slice.argmin(dim=1).item() + current_cluster_center_index + 1
                logger.debug(f"min_density_idx: {min_density_idx}")

                # add the frame index of the minimum density value to the borders list
                borders.append(min_density_idx)

        return idx_cluster, cluster_num, index_down, borders

    @staticmethod
    def index_points(points, idx):
        """Sample features following the index.
        Source: https://github.com/PKU-YuanGroup/Chat-UniVi
        Returns:
            new_points:, indexed points data, [B, S, C]

        Args:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def __str__(self):
        return f"DPCKNNSplit(num_clusters={self.num_clusters}, k={self.k}, clip_model_name={self.clip_model_name})"
