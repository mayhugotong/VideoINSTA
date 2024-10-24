import logging
import torch

from torchvision.transforms import ToPILImage
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoModelForCausalLM

logger = logging.getLogger("root")


def infer_transcript_from_video_clip_using_frame_captions(
        video_clip: torch.Tensor,
        start_frame: int,
        original_fps: float,
        frame_prompt: str = "describe the image in detail",
        model_id: str = "THUDM/cogagent-chat-hf",
        tokenizer_id: str = "lmsys/vicuna-7b-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "torch.float16",
        quantization: bool = False,
        temperature: float = 0.9,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
):
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_id)

    precision = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
    }[precision]

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
        load_in_4bit=quantization,
        trust_remote_code=True
    ).to(device).eval()

    logger.debug(f"Initialized model {model_id} with precision {precision} on device:{device}")

    f, c, h, w = video_clip.size()
    logger.debug(f"Inferring frame captions for video clip size {video_clip.size()} using prompt \"{frame_prompt}\"...")

    to_pil = ToPILImage()
    results = {}
    for i in tqdm(range(f)):
        frame = to_pil(video_clip[i]).convert('RGB')

        input_by_model = model.build_conversation_input_ids(tokenizer, query=frame_prompt, history=[], images=[frame])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(precision)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(precision)]]

        # add any transformers params here
        gen_kwargs = {"max_length": max_new_tokens,
                      "temperature": temperature,
                      "do_sample": do_sample}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]

        abs_start_frame_index = int(start_frame + i * original_fps)
        abs_end_frame_index = int(start_frame + (i + 1) * original_fps)
        interval = (abs_start_frame_index, abs_end_frame_index)

        # return as a dict with just one entry to fit the format of e.g. LaViLa that can return multiple captions
        results[interval] = {0: response}

        logger.debug(f"Relative index interval (1 fps, i.e. seconds): ({i}, {i + 1})")
        logger.debug(f"Absolute index interval (30 fps): ({abs_start_frame_index}, {abs_end_frame_index})")

    logger.debug(f"Frame inferences: {results}")
    logger.info("Inferred frame captions for the video clip")

    return results
