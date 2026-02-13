import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

# 全局缓存
_model = None
_processor = None
_model_path = None
_device = None

def load_model(model_path, device="cuda"):
    """
    加载 Qwen2-VL 模型和处理器。
    如果已加载相同路径和设备的模型，则直接返回缓存。
    """
    global _model, _processor, _model_path, _device
    if _model is None or _model_path != model_path or _device != device:
        print(f"Loading Qwen2-VL model from {model_path} on {device}...")
        try:
            if device == "cuda":
                _model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                _model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                _model.cpu()
            _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            _model_path = model_path
            _device = device
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    return _model, _processor

def generate_caption(
    image,
    prompt,
    model_path,
    device="cuda",
    max_new_tokens=128,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.0,
    **kwargs
):
    """
    使用 Qwen2-VL 模型为图像生成描述。
    
    Args:
        image: PIL Image 对象或图像路径（字符串）。
        prompt: 文本提示。
        model_path: 模型路径或 HuggingFace 模型 ID。
        device: "cuda" 或 "cpu"。
        max_new_tokens: 最大生成 token 数。
        temperature: 温度参数。
        do_sample: 是否采样。
        top_p: top-p 采样参数。
        repetition_penalty: 重复惩罚。
        **kwargs: 其他传递给 model.generate 的参数。
    
    Returns:
        生成的文本字符串。
    """
    model, processor = load_model(model_path, device)

    # 图像预处理
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        # 尝试将其他类型（如 numpy 数组）转为 PIL
        try:
            image = Image.fromarray(image)
        except:
            raise TypeError("image must be a PIL Image, a file path, or convertible to PIL Image")

    # 构造对话消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        }
    ]

    # 应用聊天模板
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 准备输入
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # 将输入移动到模型所在设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成参数
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        **kwargs
    }

    # 生成
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # 裁剪输入部分
    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]

    # 解码
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text