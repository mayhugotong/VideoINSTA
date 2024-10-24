import logging
import math
import os
import re

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import urllib.request

from collections import OrderedDict

from datasets.utils import create_video_data_from_video_path
from toolbox.lavila_video_captioner.lavila.data.video_transforms import Permute
from toolbox.lavila_video_captioner.lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL, \
    VCLM_OPENAI_TIMESFORMER_LARGE_GPT2_XL, VCLM_OPENAI_TIMESFORMER_BASE_GPT2
from toolbox.lavila_video_captioner.lavila.models.tokenizer import MyGPT2Tokenizer
from tqdm import tqdm

from toolbox.lavila_video_captioner.lavila.models.utils import inflate_positional_embeds
from toolbox.lavila_video_captioner.lavila.utils.preprocess import generate_tokenizer

logger = logging.getLogger("root")


def decode_one(generated_ids, tokenizer):
    # get the index of <EOS>
    if tokenizer.eos_token_id == tokenizer.bos_token_id:
        if tokenizer.eos_token_id in generated_ids[1:].tolist():
            eos_id = generated_ids[1:].tolist().index(tokenizer.eos_token_id) + 1
        else:
            eos_id = len(generated_ids.tolist()) - 1
    elif tokenizer.eos_token_id in generated_ids.tolist():
        eos_id = generated_ids.tolist().index(tokenizer.eos_token_id)
    else:
        eos_id = len(generated_ids.tolist()) - 1
    generated_text_str = tokenizer.tokenizer.decode(generated_ids[1:eos_id].tolist())
    return generated_text_str


def infer_caption_from_video_clip(video_clip, temperature=0.7, top_p=0.95, num_return_sequences=10, max_text_length=77,
                                  cuda=False, early_stopping=True, num_seg=4):
    """
    This function infers a caption from a video clip.
    It does so by linearly sampling 4 frames of the video clip, and then feeding them to the model.
    This works well for short clips, but for longer clips, it is better to divide it into small chunks and iteratively
    infer captions for the short clips and combine them to higher levels using e.g. an LLM.
    The latter can be done with the other function in this module, i.e. infer_transcript_from_video_clip(...).
    """
    ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
    ckpt_path = os.path.join('modelzoo/', ckpt_name)
    os.makedirs('modelzoo/', exist_ok=True)
    if not os.path.exists(ckpt_path):
        logger.info('downloading model to {}'.format(ckpt_path))
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name),
                                   ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # instantiate the model, and load the pre-trained weights
    model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
        text_use_cls_token=False,
        project_embed_dim=256,
        gated_xattn=True,
        timesformer_gated_xattn=False,
        freeze_lm_vclm=False,  # we use model.eval() anyway
        freeze_visual_vclm=False,  # we use model.eval() anyway
        num_frames=num_seg,
        drop_path_rate=0.
    )
    model.load_state_dict(state_dict, strict=True)
    if cuda:
        model.cuda()
    model.eval()

    frames = video_clip.numpy()
    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]

    # transforms on input frames
    crop_size = 336
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001],
                                        std=[68.5005327, 66.6321579, 70.32316305])
    ])
    frames = val_transform(frames)
    frames = frames.unsqueeze(0)  # fake a batch dimension

    tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
    with torch.no_grad():
        if cuda:
            frames = frames.cuda(non_blocking=True)
        image_features = model.encode_image(frames)
        generated_text_ids, ppls = model.generate(
            image_features,
            tokenizer,
            target=None,  # free-form generation
            max_text_length=max_text_length,
            top_k=None,
            top_p=top_p,  # nucleus sampling
            num_return_sequences=num_return_sequences,  # number of candidates: 10
            temperature=temperature,
            early_stopping=early_stopping,
        )

    captions = {}
    for j in range(num_return_sequences):
        generated_text_str = decode_one(generated_text_ids[j], tokenizer)
        captions[j] = generated_text_str
        logger.debug('{}: {}'.format(j, generated_text_str))

    return captions


def infer_transcript_from_video_clip_using_action_captions(
        video_clip,
        fps,
        original_fps,
        interval_in_seconds: int = 1,
        temperature=0.2,
        top_p=0.95,
        num_return_sequences=10,
        max_text_length=77,
        cuda=False,
        early_stopping=True,
        num_seg=4,
        start_frame=0,
        modelzoo_dir_path: str = "./toolbox/lavila_video_captioner/modelzoo",
        checkpoint_download_url: str = "https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator",
        checkpoint_file: str = "vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth"
):
    """
    This function infers a transcript from a video clip.
    It does so by dividing the video clip into chunks of interval_in_seconds, and then feeding them to the LaViLa model.
    """
    logger.warning("Only use this function with appropriate frame rate "
                   "of video_clip (be sure you know what you are doing)!!!")

    logger.info(f"Inferring action captions for the video clip of shape {video_clip.size()}...")

    # load the pre-trained model
    ckpt_path = os.path.join(modelzoo_dir_path, checkpoint_file)
    os.makedirs(modelzoo_dir_path, exist_ok=True)
    if not os.path.exists(ckpt_path):
        logger.debug('downloading model to {}'.format(ckpt_path))
        urllib.request.urlretrieve(f"{checkpoint_download_url}/{checkpoint_file})", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # instantiate the model, and load the pre-trained weights
    if "336px" in checkpoint_file:
        logger.debug("Using the 336px timesformer model...")
        model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
            text_use_cls_token=False,
            project_embed_dim=256,
            gated_xattn=True,
            timesformer_gated_xattn=False,
            freeze_lm_vclm=False,  # we use model.eval() anyway
            freeze_visual_vclm=False,  # we use model.eval() anyway
            freeze_visual_vclm_temporal=False,
            num_frames=num_seg,
            drop_path_rate=0.
        )
        crop_size = 336

        tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
        logger.debug("Initialized LaViLa tokenizer")
    elif "ckpt_base.pt" in checkpoint_file:
        logger.debug("Using the base timesformer model...")
        old_args = ckpt['args']

        model = VCLM_OPENAI_TIMESFORMER_BASE_GPT2(
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

        logger.debug('Inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=old_args.clip_length,
            load_temporal_fix='bilinear',
        )

        tokenizer = generate_tokenizer(old_args.model)
        logger.debug("Initialized LaViLa tokenizer")

        crop_size = 224
    else:
        logger.debug("Using the 224px (default) timesformer model...")
        model = VCLM_OPENAI_TIMESFORMER_LARGE_GPT2_XL(
            text_use_cls_token=False,
            project_embed_dim=256,
            gated_xattn=True,
            timesformer_gated_xattn=False,
            freeze_lm_vclm=False,  # we use model.eval() anyway
            freeze_visual_vclm=False,  # we use model.eval() anyway
            freeze_visual_vclm_temporal=False,
            num_frames=num_seg,
            drop_path_rate=0.
        )
        crop_size = 224

        tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
        logger.debug("Initialized LaViLa tokenizer")

    model.load_state_dict(state_dict, strict=True)
    if cuda:
        model.cuda()
    model.eval()
    logger.debug("Initialized LaViLa model")

    logger.debug("Starting inference")
    logger.debug(f"video_clip.size(): {video_clip.size()}")
    results = {}

    # round up so that we don't miss the last interval
    num_intervals = math.ceil(video_clip.size(0) / (fps * interval_in_seconds))
    # duration = int(video_clip.size(0) // fps)
    # interval_in_frames = int(fps * interval_in_seconds)
    for i in tqdm(range(num_intervals),
                  desc=f"Infer action captions for an interval of {num_seg} frames "
                       f"for each {interval_in_seconds} second in the video clip..."):
        # start_frame = i * interval_in_frames
        # end_frame = start_frame + interval_in_frames
        # if i == duration:
        #     end_frame = video_clip.size(0) - 1

        # frame_ids = get_frame_ids(start_frame, end_frame, num_segments=num_seg, jitter=False)
        # frame_ids = torch.tensor(frame_ids)

        start = i * num_seg
        end = start + num_seg if i != num_intervals - 1 else video_clip.size(0)

        frame_ids = torch.arange(start=start, end=end, step=1)

        logger.debug(f"frame_ids: {frame_ids}")
        frames = torch.index_select(video_clip, 0, frame_ids).numpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]

        interval_size = end - start
        if interval_size != num_seg:
            # pad the end
            pad_size = num_seg - interval_size

            # get last frame of the tensor
            last_frame = frames[-1]

            # repeat the last frame
            frames += [last_frame] * pad_size

        frames = torch.stack(frames, dim=0)

        # transforms on input frames
        # input is a tensor of shape (num_seg, 3, height, width)
        # output is a tensor of shape (1, 3, num_seg, 336, 336)
        val_transform = transforms.Compose([
            Permute([1, 0, 2, 3]),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001],
                                            std=[68.5005327, 66.6321579, 70.32316305])
        ])
        frames = val_transform(frames)
        frames = frames.unsqueeze(0)  # fake a batch dimension

        with torch.no_grad():
            if cuda:
                frames = frames.cuda(non_blocking=True)
            image_features = model.encode_image(frames)
            generated_text_ids, ppls = model.generate(
                image_features,
                tokenizer,
                target=None,  # free-form generation
                max_text_length=max_text_length,
                top_k=None,
                top_p=top_p,  # nucleus sampling
                num_return_sequences=num_return_sequences,  # default number of candidates: 10
                temperature=temperature,
                early_stopping=early_stopping,
            )

        # get the caption with the lowest perplexity
        ppls = [ppl.item() for ppl in ppls]
        logger.debug(f"Perplexities: {ppls}")

        # get the index of the caption with the lowest perplexity
        index_lowest_ppl = ppls.index(min(ppls))

        # get the caption with the lowest perplexity
        generated_text_str_lowest_ppl = decode_one(generated_text_ids[index_lowest_ppl], tokenizer)
        logger.debug(f"BEST CAPTION (with lowest perplexity): {generated_text_str_lowest_ppl}")

        # remove repetitive words if they are behind each other
        # e.g. "my dog dog loves playing with my neighbors dog dog" -> "my dog loves playing with my neighbors dog"
        caption = ' '.join([word for i, word in enumerate(generated_text_str_lowest_ppl.split()) if
                            i == 0 or word != generated_text_str_lowest_ppl.split()[i - 1]])

        # remove "#O"
        caption = caption.replace("#O", "")
        caption = caption.replace("#o", "")

        # remove "#C"
        caption = caption.replace("#C", "")
        caption = caption.replace("#c", "")

        # remove "#unsure"
        caption = caption.replace("#unsure", "")
        caption = caption.replace("# unsure", "")

        # remove "#summary"
        caption = caption.replace("#summary", "")
        caption = caption.replace("# summary", "")

        # remove standalone "#"
        caption = caption.replace("#", "")

        # remove double whitespaces
        caption = caption.replace("  ", " ")

        # remove leading and trailing whitespaces
        caption = caption.strip()

        # replace " ." with "."
        caption = caption.replace(" .", ".")

        # add dot at the end if not there
        if caption[-1] != ".":
            caption = caption + "."

        # replace "C" with "the camera wearer"
        pattern = r'\bC\b'
        caption = re.sub(pattern, 'the camera wearer', caption, flags=re.IGNORECASE)

        # make other chars lowercase
        caption = caption.lower()

        # make first char uppercase
        caption = caption[0].upper() + caption[1:]

        logger.debug(f"FINAL CAPTION: {caption}")

        # uhm, due to deprecated output formatting :O
        interval_results = {0: caption}

        # remember the results
        start_frame_index = int(start_frame + (start / num_seg) * original_fps)
        end_frame_index = int(start_frame + (end / num_seg) * original_fps)
        interval = (start_frame_index, end_frame_index)
        results[interval] = interval_results
        # (0, 30): ["dog walking to door", ...]
        # (30, 60): ["dog walking to door", "dog walking to the door", ]
        # (60, 90): ["dog walking to door", ...]

        logger.debug(f"Relative index interval (1 fps, i.e. seconds): ({i}, {i + 1})")
        logger.debug(f"Relative index interval (4 fps): ({start}, {end})")
        logger.debug(f"Absolute index interval (30 fps): ({start_frame_index}, {end_frame_index})")

    logger.info("Inferred action captions for the video clip")

    return results


if __name__ == '__main__':
    video, metadata = create_video_data_from_video_path(
        video_path="../experiments/vid-reas-main/data/egoschema/videos/0074f737-11cb-497d-8d07-77c3a8127391.mp4",
        sample_rate=4.0,
        window_start_index=0,
        window_end_index=300
    )

    # get captions for the video_clip_data using LaViLa
    action_captions = infer_transcript_from_video_clip_using_action_captions(
        video_clip=video,
        start_frame=0,
        fps=4.0,
        original_fps=1.0,
        interval_in_seconds=1,
        temperature=0.7,
        top_p=0.95,
        max_text_length=256,
        num_return_sequences=5,
        early_stopping=True,
        num_seg=4,
        cuda=False,
        modelzoo_dir_path="./toolbox/lavila_video_captioner/modelzoo",
        checkpoint_download_url="https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator",
        checkpoint_file="ckpt_base.pt"
    )

    print(action_captions)
