import os
import torch
from comfy.model_management import interrupt_processing
from ..utils.qwen_vl import tag_image


class LoraToolQwenVLTagger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "G:/models/Qwen-VL-3B-Instruct"
                }),
                "input_path": ("STRING", {
                    "default": ""
                }),
                "prompt": ("STRING", {
                    "default": "请仔细观察这张图片，并用详细、准确的中文自然语言描述它。"
                }),
                "image_exts": ("STRING", {
                    "default": ".png,.jpg,.jpeg,.webp"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "comfyui-lora-tool"

    def run(self, model_path, input_path, prompt, image_exts):

        if not os.path.isdir(input_path):
            return ("输入图片路径不存在",)

        if not os.path.isdir(model_path):
            return ("模型路径不存在",)

        exts = [e.strip().lower() for e in image_exts.split(",")]
        images = [
            f for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in exts
        ]

        total = len(images)
        if total == 0:
            return ("未找到图片文件",)

        success = 0

        for idx, img in enumerate(images, start=1):

            # ⭐⭐⭐ Stop 响应点
            if interrupt_processing:
                print("[Qwen-VL] 收到 Stop 信号，立即中断任务")
                break

            print(f"[Qwen-VL] 正在处理 {idx}/{total} : {img}")

            img_path = os.path.join(input_path, img)
            txt_path = os.path.join(
                input_path,
                os.path.splitext(img)[0] + ".txt"
            )

            try:
                tags = tag_image(
                    image_path=img_path,
                    prompt=prompt,
                    model_path=model_path
                )
            except Exception as e:
                print(f"[Qwen-VL] 失败: {img} -> {e}")
                continue

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(tags)

            success += 1

            # ⭐ GPU 同步，避免 Stop 延迟
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print("[Qwen-VL] 节点执行结束")

        return (f"Qwen-VL 打标完成：{success}/{total}",)
