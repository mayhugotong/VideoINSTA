# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import os
import urllib.request
from collections import OrderedDict

import time
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import decord
from tqdm import tqdm
from toolbox.lavila_video_captioner.lavila.models import models
from toolbox.lavila_video_captioner.lavila.data.video_transforms import Permute, TemporalCrop, SpatialCrop
from toolbox.lavila_video_captioner.lavila.data.datasets import get_frame_ids, video_loader_by_frames
from toolbox.lavila_video_captioner.lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL, \
    VCLM_OPENAI_TIMESFORMER_LARGE_GPT2_XL, CLIP_OPENAI_TIMESFORMER_BASE
from toolbox.lavila_video_captioner.lavila.models.tokenizer import MyGPT2Tokenizer
from toolbox.lavila_video_captioner.lavila.models.utils import inflate_positional_embeds
from toolbox.lavila_video_captioner.lavila.utils.preprocess import generate_tokenizer
from toolbox.lavila_video_captioner.narrator_inference import decode_one


def main(args):
    num_seg = 4
    interval_in_seconds = 1

    # vr = decord.VideoReader(args.video_path, width=336, height=336, num_threads=1, ctx=decord.cpu(0))
    # fps = vr.get_avg_fps()
    # interval_in_seconds = 1
    # interval_in_frames = int(fps * interval_in_seconds)
    # frame_ids = get_frame_ids(0, interval_in_frames, num_segments=num_seg, jitter=False)
    # frames = video_loader_by_frames(vr, frame_ids)

    # ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
    # ckpt_name = 'vclm_openai_timesformer_large_gpt2_xl.pt_htm.jobid_341080.ep_0001.pth'
    # ckpt_name = 'vclm_openai_timesformer_base_gpt2_base.pt_ego4d.jobid_319630.ep_0002.md5sum_68a71f.pth'
    # ckpt_name = 'clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth'
    ckpt_name = 'ckpt_base.pt'
    ckpt_path = os.path.join('toolbox/lavila_video_captioner/modelzoo/', ckpt_name)
    print('ckpt_path: {}'.format(ckpt_path))
    os.makedirs('toolbox/lavila_video_captioner/modelzoo/', exist_ok=True)
    if not os.path.exists(ckpt_path):
        print('downloading model to {}'.format(ckpt_path))
        # urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name),
        #                            ckpt_path)
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/{}'.format(ckpt_name),
            ckpt_path)
        # urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/htm_aa/{}'.format(ckpt_name),
        #                            ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')

    print(f"ckpt: {ckpt.keys()}")
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v
        # state_dict[k.replace('text_decoder.', '')] = v
        # state_dict[k.replace('text_decoder.', '')] = v

    print(f"state_dict: {state_dict.keys()}")

    # instantiate the model, and load the pre-trained weights
    # model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
    #     text_use_cls_token=False,
    #     project_embed_dim=256,
    #     gated_xattn=True,
    #     timesformer_gated_xattn=False,
    #     freeze_lm_vclm=False,  # we use model.eval() anyway
    #     freeze_visual_vclm=False,  # we use model.eval() anyway
    #     freeze_visual_vclm_temporal=False, # TODO
    #     num_frames=num_seg,
    #     drop_path_rate=0.
    # )
    # model = VCLM_OPENAI_TIMESFORMER_LARGE_GPT2_XL(
    #     text_use_cls_token=False,
    #     project_embed_dim=256,
    #     gated_xattn=True,
    #     timesformer_gated_xattn=False,
    #     freeze_lm_vclm=False,  # we use model.eval() anyway
    #     freeze_visual_vclm=False,  # we use model.eval() anyway
    #     num_frames=num_seg,
    #     drop_path_rate=0.
    # )

    old_args = ckpt['args']
    print('old_args: {}'.format(old_args))

    model = getattr(models, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
        timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
        timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
        freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
        freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
        num_frames=old_args.clip_length,
        drop_path_rate=0,
    )

    model.cuda()
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=old_args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)

    model.cuda()
    model.eval()

    # tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)

    tokenizer = generate_tokenizer(old_args.model)
    crop_size = 224 if '336PX' not in old_args.model else 336

    start_time = time.time()

    vr = decord.VideoReader(args.video_path, width=224, height=224, num_threads=1, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    duration = int(len(vr) // fps)

    print("Starting inference")

    results = {}
    for i in tqdm(range(duration + 1),
                  desc="Infer action captions for an interval around each frame in the video clip..."):
        interval_in_frames = int(fps * interval_in_seconds)
        print(f"interval_in_frames: {interval_in_frames}")

        start_frame = i * interval_in_frames
        print(f"start_frame: {start_frame}")
        end_frame = start_frame + interval_in_frames
        print(f"end_frame: {end_frame}")

        if i == duration:
            end_frame = len(vr) - 1

        frame_ids = get_frame_ids(start_frame, end_frame, num_segments=num_seg, jitter=False)
        print(f"frame_ids: {len(frame_ids)}, {frame_ids}")
        frames = video_loader_by_frames(vr, frame_ids)
        print(f"frames: {frames.size()}")

        # transforms on input frames
        # crop_size = 224
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001],
                                            std=[68.5005327, 66.6321579, 70.32316305])
        ])
        frames = val_transform(frames)
        frames = frames.unsqueeze(0)  # fake a batch dimension
        print(f"frames: {frames.size()}")

        with torch.no_grad():
            frames = frames.cuda(non_blocking=True)
            image_features = model.encode_image(frames)
            generated_text_ids, ppls = model.generate(
                image_features,
                tokenizer,
                target=None,  # free-form generation
                max_text_length=77,
                top_k=None,
                top_p=0.95,  # nucleus sampling
                num_return_sequences=10,  # number of candidates: 10
                temperature=0.2,
                early_stopping=True,
            )

        interval_results = {}
        print(f"Interval {i} - {i + 1}:")
        for j in range(10):
            generated_text_str = decode_one(generated_text_ids[j], tokenizer)
            interval_results[j] = generated_text_str
            print('{}: {}'.format(j, generated_text_str))

        # remember the results
        results[f"{i}s - {i + 1}s"] = interval_results

    end_time = time.time()

    # save results
    save_path = f"./video_transcript_results_interval_{interval_in_seconds}s.json"
    with open(save_path, "w") as outfile:
        json.dump(results, outfile)

    print('elapsed video inference time: {:.3f} seconds'.format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila narrator demo')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--video-path', default='assets/3c0dffd0-e38e-4643-bc48-d513943dc20b_012_014.mp4', type=str,
                        help='video path')
    args = parser.parse_args()
    main(args)
