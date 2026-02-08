import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from comfy.model_management import interrupt_processing

_model_cache = {}


def load_qwen_vl(model_path: str):
    if model_path in _model_cache:
        return _model_cache[model_path]

    if not os.path.isdir(model_path):
        raise RuntimeError(f"模型路径不存在: {model_path}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()
    _model_cache[model_path] = (model, processor)
    return model, processor


def _clean_output(text: str) -> str:
    """
    只保留 assistant 的最终输出内容
    """
    if "assistant" in text:
        text = text.split("assistant")[-1]

    text = text.strip()

    text = re.sub(
        r"^(system|user)\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text


def tag_image(image_path: str, prompt: str, model_path: str) -> str:
    if interrupt_processing:
        raise RuntimeError("任务被用户中断")

    model, processor = load_qwen_vl(model_path)

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # ① 生成 prompt 字符串
    prompt_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # ② 转为 tensor
    inputs = processor(
        images=image,
        text=prompt_text,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    # ⭐ 强制 GPU 同步，确保任务真正结束
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    raw_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return _clean_output(raw_text)
