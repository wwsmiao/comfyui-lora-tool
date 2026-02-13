import torch
import numpy as np
from PIL import Image
from ..utils.qwen2_vl import generate_caption

class LoraToolQwen2VLTagger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 输入图像批次，形状 (B,H,W,C)，值范围 0-1
                "model_path": ("STRING", {
                    "default": "Qwen/Qwen2-VL-2B-Instruct",
                    "tooltip": "本地模型路径或 HuggingFace 模型 ID"
                }),
                "prompt": ("STRING", {
                    "default": "描述这张图片",
                    "multiline": True,
                    "tooltip": "提示词，引导模型生成描述"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "运行设备"
                }),
                "max_new_tokens": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 2048,
                    "step": 1,
                    "tooltip": "最大生成 token 数"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "温度参数，控制随机性"
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否使用采样生成"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "top-p 采样参数"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "重复惩罚系数"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "tag"
    CATEGORY = "LoraTool"

    def tag(self, images, model_path, prompt, device, max_new_tokens, temperature, do_sample, top_p, repetition_penalty):
        # 取批次中的第一张图像
        if images.shape[0] == 0:
            return ("No image provided",)

        img_tensor = images[0].cpu().numpy()  # (H, W, C)
        # 值范围 0-1 转为 0-255
        img_np = (img_tensor * 255).astype(np.uint8)

        # 转换为 PIL Image
        if img_np.shape[-1] == 3:
            pil_img = Image.fromarray(img_np, mode='RGB')
        elif img_np.shape[-1] == 1:
            pil_img = Image.fromarray(img_np[..., 0], mode='L')
        else:
            pil_img = Image.fromarray(img_np)  # 可能出错，但尝试转换

        # 调用生成函数
        try:
            result = generate_caption(
                image=pil_img,
                prompt=prompt,
                model_path=model_path,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        except Exception as e:
            result = f"Error: {str(e)}"

        return (result,)