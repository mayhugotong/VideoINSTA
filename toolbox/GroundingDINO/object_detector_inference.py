import cv2
import logging
import numpy as np
import toolbox.GroundingDINO.groundingdino.datasets.transforms as T
import torch
import torchvision.transforms.functional as F

from toolbox.GroundingDINO.groundingdino.util.inference import load_model, predict, annotate
from toolbox.GroundingDINO.groundingdino.util.vl_utils import create_positive_map_from_span
from torchvision.io import read_image
from tqdm import tqdm

logger = logging.getLogger("root")


def infer_bounding_box_from_image(image_tensor: torch.Tensor,
                                  config_file: str = "groundingdino/config/GroundingDINO_SwinB_cfg.py",
                                  checkpoint_path: str = "weights/groundingdino_swinb_cogcoor.pth",
                                  text_prompt: str = None,
                                  box_threshold: float = 0.3,
                                  text_threshold: float = 0.25,
                                  token_spans: str = None,
                                  cuda: bool = True):
    # pre-process image
    image_pil = F.to_pil_image(image_tensor).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = np.asarray(image_pil)
    image_tensor, _ = transform(image_pil, None)  # 3, h, w

    # load model
    model = load_model(config_file, checkpoint_path, device="cuda" if cuda else "cpu")

    if token_spans is None:
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
    else:
        caption = text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = "cuda" if cuda else "cpu"
        model = model.to(device)
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image_tensor.device)  # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T  # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            logit_phr_num = logit_phr[filt_mask]
            all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
        boxes = torch.cat(all_boxes, dim=0).cpu()
        logits = torch.cat(all_logits, dim=0).cpu()
        phrases = all_phrases

    logger.debug(f"boxes: {boxes}")
    logger.debug(f"logits: {logits}")
    logger.debug(f"phrases: {phrases}")
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    return {
        "boxes": boxes,
        "logits": logits,
        "phrases": phrases,
        "image_with_boxes": annotated_frame
    }


def infer_bounding_boxes_from_video(video_tensor: torch.Tensor,
                                    obj: str,
                                    config_file: str = "groundingdino/config/GroundingDINO_SwinB_cfg.py",
                                    checkpoint_path: str = "weights/groundingdino_swinb_cogcoor.pth",
                                    box_threshold: float = 0.2,
                                    text_threshold: float = 0.0,
                                    cuda: bool = True):
    logger.info(f"Inferring bounding boxes for object: {obj} in video of shape {video_tensor.size()}...")

    # load model
    model = load_model(config_file, checkpoint_path, device="cuda" if cuda else "cpu")

    # infer bounding box for each frame of the given video
    inferences = []
    for i in tqdm(range(video_tensor.shape[0]), desc="Infer bounding boxes for each frame of the video clip..."):
        image_tensor = video_tensor[i, :, :, :]

        # pre-process image
        image_pil = F.to_pil_image(image_tensor).convert("RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # image_source = np.asarray(image_pil)
        image_tensor, _ = transform(image_pil, None)  # 3, h, w

        # predict bounding box
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=obj,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        if logits.size(0) == 0:
            box = None
            probability = 0.0
        else:
            box = boxes[torch.argmax(logits, dim=0)].tolist()
            probability, _ = torch.max(logits, dim=0)
            probability = probability.item()

        inferences.append({
            "box": box,
            "probability": probability
        })

    logger.info("Inferred bounding boxes")

    return inferences


if __name__ == "__main__":
    # image = read_image("weights/test.png")
    image = read_image("toolbox/GroundingDINO/weights/10206.png")
    print(image.shape)

    prompt = "kitchen"
    # token_spans = eval(f"[[[0, 3], [4, 10], [11, 14], [15, 18], [19, 25], [26, 29]]]")
    # token_spans = eval(f"[[[0, {len(prompt)}]]]")

    # pred_dict = infer_bounding_box_from_image(image_tensor=image,
    #                                           text_prompt=prompt,
    #                                           box_threshold=0.07,
    #                                           token_spans=token_spans)
    pred_dict = infer_bounding_box_from_image(image_tensor=image,
                                              text_prompt=prompt,
                                              box_threshold=0.0,
                                              text_threshold=0.0,
                                              config_file="./toolbox/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                                              checkpoint_path="./toolbox/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")

    cv2.imwrite("annotated_image.jpg", pred_dict["image_with_boxes"])

    if pred_dict["logits"].size(0) == 0:
        box = None
        probability = 0.0
    else:
        box = pred_dict["boxes"][torch.argmax(pred_dict["logits"], dim=0)].tolist()
        probability, _ = torch.max(pred_dict["logits"], dim=0)
        probability = probability.item()

    print(f"boxes: {pred_dict['boxes']}")
    print(f"logits: {pred_dict['logits']}")
    print(f"phrases: {pred_dict['phrases']}")
    print(f"num boxes: {len(pred_dict['boxes'])}")
    print(f"num logits: {len(pred_dict['logits'])}")
    print(f"num phrases: {len(pred_dict['phrases'])}")
    print(f"sum: {pred_dict['logits'].sum()}")
    print("===============================")
    print(f"box: {box}")
    print(f"probability: {probability}")
