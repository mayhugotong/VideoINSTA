import logging
import math
import sys

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from toolbox.UniVTG.main.config import TestOptions, setup_model
from toolbox.UniVTG.run_on_video import clip, txt2clip
from toolbox.UniVTG.run_on_video.preprocessing import Preprocessing
from toolbox.UniVTG.utils.basic_utils import l2_normalize_np_array
from torchvision.transforms import v2

logger = logging.getLogger("root")


class DictObj:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def load_model(args):
    logger.debug("Setup config, data and model...")
    opt = TestOptions().parse(args)
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model


def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(round(seconds)))


def infer_temporal_grounding_score_from_video_and_text(
        video: torch.Tensor,
        text: str,
        config_dir: str,
        checkpoint_path: str,
        clip_model_version: str,
        output_feat_size: str,
        half_precision: bool,
        jit: bool,
        resize_size: int,
        gpu_id: int
):
    # trick to assure that the argument parser does not complain about the arguments
    sys.argv = [sys.argv[0]]

    logger.info(f"Inferring temporal grounding score for text: {text} in video of shape {video.size()}...")

    args = {
        "save_dir": config_dir,
        "resume": checkpoint_path,
        "gpu_id": gpu_id,
    }

    args = DictObj(args)

    # device
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.debug(f"device: {device}")
    torch.cuda.empty_cache()

    # feature extractor
    clip_model, _ = clip.load(clip_model_version, device=device, jit=jit)
    output_feat_size = output_feat_size
    half_precision = half_precision

    # video feature extraction
    # TODO assure that video is of shape (batch, channel, height, width)
    assert len(video.shape) == 4
    assert video.shape[1] == 3
    logger.debug(f"video shape: {video.size()}")

    size = resize_size
    b, c, h, w = video.size()
    # if h >= w:
    #     output_size = int(h * size / w), size
    # else:
    #     output_size = size, int(w * size / h)

    # TODO this scales the video to 224x224 and so changes the aspect ratio, maybe better use center crop?
    # TODO center crop however removes information...
    resize = v2.Resize([size, size])
    # center_crop = v2.CenterCrop(size)
    # resize = v2.Resize(output_size)
    video = resize(video)
    # video = center_crop(video)

    # print(f"video shape: {video.size()}")

    preprocess = Preprocessing()
    video = preprocess(video)
    n_chunk = len(video)
    vid_features = torch.cuda.FloatTensor(
        n_chunk, output_feat_size).fill_(0)
    n_iter = int(math.ceil(n_chunk))
    for i in range(n_iter):
        min_ind = i
        max_ind = (i + 1)
        video_batch = video[min_ind:max_ind].to(device)
        batch_features = clip_model.encode_image(video_batch)
        vid_features[min_ind:max_ind] = batch_features
    video_features = vid_features.cpu().detach().numpy()
    if half_precision:
        video_features = vid_features.astype('float16')

    # text feature extraction
    text_features = txt2clip(clip_model, text, "./toolbox/UniVTG/tmp")

    # prepare data
    vid = torch.from_numpy(l2_normalize_np_array(video_features.astype(np.float32)))
    txt = torch.from_numpy(l2_normalize_np_array(text_features.astype(np.float32)))
    # clip_length is specified in opt.json, the authors used 2 (in seconds)
    ctx_l = vid.shape[0]

    # fps = meta_data["fps"]
    # sample_rate = meta_data["sample_rate"]
    # clip_len = 1.0 / sample_rate
    # sampled_indices = torch.from_numpy(np.array(meta_data["sample_indices"]))
    # timestamps = sampled_indices / fps * sample_rate
    # timestamp = ((timestamps + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)
    # timestamp = ((torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    src_vid = src_vid.cuda(args.gpu_id)
    src_txt = src_txt.cuda(args.gpu_id)
    src_vid_mask = src_vid_mask.cuda(args.gpu_id)
    src_txt_mask = src_txt_mask.cuda(args.gpu_id)

    # print(f"src_vid: {src_vid.dtype}")
    # print(f"src_txt: {src_txt.dtype}")
    # print(f"src_vid_mask: {src_vid_mask.dtype}")
    # print(f"src_txt_mask: {src_txt_mask.dtype}")

    # temporal grounding model
    model = load_model(args=args)

    # print(f"model: {model}")

    model.eval()
    with torch.no_grad():
        logger.debug("Infer temporal grounding score of video clip...")
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    # print(f"output: {output}")

    # prepare the model prediction
    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()
    pred_saliency = output['saliency_scores'].cpu()

    # print(f"pred_logits: {pred_logits}")
    # print(f"pred_spans: {pred_spans}")
    # print(f"pred_saliency: {pred_saliency}")

    # prepare the model prediction
    # pred_spans = pred_spans * ctx_l
    # pred_windows = (pred_spans + timestamp) * ctx_l

    # pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
    # pred_confidence = pred_logits

    # TODO clamp to assure that the window is not longer than the video duration
    # spans = torch.clamp(spans, 0, meta["duration"])  # added by Kevin, since window cannot be longer than video duration.

    # print(f"pred_windows: {pred_windows}")
    # print(f"timestamps: {timestamps}")
    # print(f"timestamp: {timestamp}")

    # grounding
    # top1_window = pred_windows[torch.argmax(pred_confidence)].tolist()
    # top5_values, top5_indices = torch.topk(pred_confidence.flatten(), k=min(5, b))
    # top5_windows = pred_windows[top5_indices].tolist()

    # print(f"top1_window: {top1_window}")
    # print(f"top{min(5, b)}_windows: {top5_windows}")

    # q_response = f"For text: {text}"
    # print(q_response)

    # mr_res = " - ".join([convert_to_hms(int(i)) for i in top1_window])
    # mr_response = f"The Top-1 interval is: {mr_res}"
    # print(mr_response)

    # hl_res = convert_to_hms(torch.argmax(pred_saliency).item() * clip_len)
    # hl_response = f"The Top-1 highlight is: {hl_res}"
    # print(hl_response)

    # TODO is a distinction between f and s in our case necessary? compare https://github.com/showlab/UniVTG/issues/2#issuecomment-1667369168

    logger.info("Inferred temporal grounding score")

    return {
        "foreground_indicators": pred_logits,
        "boundary_offsets": pred_spans,
        "saliency_scores": pred_saliency,
    }

# if __name__ == "__main__":
#     # test_video_path = "./data/videos/hitting_baseball.mp4"
#     # test_video_path = "./data/ego4d/v2/full_scale/216e3f0e-ccb9-4d54-ba56-d275fedbf52f.mp4"
#     test_video_path = "./toolbox/UniVTG/examples/charades.mp4"
#     # test_video_path = "./toolbox/UniVTG/examples/ego4d.mp4"
#     video_tensor, video_meta_data = create_video_data_from_video_path(test_video_path, sample_rate=1.0)
#
#     print(f"video_tensor: {video_tensor.size()}")
#
#     # input_query = "when does the player hit the ball with his bat?"
#     # input_query = "Where is the handle of frying pan before I turned on the cooker?"
#     # input_query = "When did I rolled the sleeve of my left hand?"
#     # input_query = "what did I pick from the fridge?"
#     input_query = "a person walks in a doorway drinking some coffee."
#     # input_query = "when do i put the book on the shelf?"
#     score = infer_temporal_grounding_score_from_video_and_text(video_tensor, video_meta_data, input_query)
#     print(f"meta_data: {video_meta_data}")
#     # print(f"score: {score}")
