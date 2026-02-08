# comfyui-lora-tool/nodes/resize_images.py
# 修改说明（按你的最新需求）：
# 1. ❌ 不再做 padding（不再填充任何颜色）
# 2. 图片严格保持“等比缩放后的真实尺寸”
# 3. 为避免 ComfyUI stack 报错：
#    → 本节点【不再输出 IMAGE】，仅负责批量处理与保存
# 4. 仍然支持：自动跳过已处理图片
#
# 设计说明：
# 在 ComfyUI 中，只要输出 IMAGE=batch，就必须 stack，
# 而等比缩放在不 padding 的前提下【数学上必然尺寸不同】。
# 因此这是唯一正确、稳定、不违背你需求的实现方式。

import os
import cv2


class LoraToolResizeImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "target_size": ("INT", {"default": 768, "min": 32, "max": 8192}),
                "resize_by": (["width", "height"],),
                "name_mode": (["keep_original", "number_sequence"],),
                "output_format": (["same_as_original", "png", "jpg", "jpeg"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "comfyui-lora-tool"

    def run(self, input_path, output_path, target_size, resize_by, name_mode, output_format):
        if not os.path.isdir(input_path):
            return ("输入路径不存在",)

        os.makedirs(output_path, exist_ok=True)

        image_exts = (".jpg", ".jpeg", ".png", ".webp")
        files = [f for f in os.listdir(input_path) if f.lower().endswith(image_exts)]

        if not files:
            return ("未找到图片文件",)

        processed = 0
        skipped = 0
        failed = 0

        for idx, name in enumerate(files, start=1):
            src = os.path.join(input_path, name)
            base, ext = os.path.splitext(name)

            out_ext = ext if output_format == "same_as_original" else f".{output_format}"
            out_name = base + out_ext if name_mode == "keep_original" else f"{idx}{out_ext}"
            out_path = os.path.join(output_path, out_name)

            if os.path.exists(out_path):
                skipped += 1
                continue

            img = cv2.imread(src)
            if img is None:
                failed += 1
                continue

            h, w = img.shape[:2]

            if resize_by == "width":
                scale = target_size / w
                new_w = target_size
                new_h = int(h * scale)
            else:
                scale = target_size / h
                new_h = target_size
                new_w = int(w * scale)

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_path, resized)
            processed += 1

        return (
            f"尺寸重设完成：成功 {processed} 张，跳过 {skipped} 张，失败 {failed} 张",
        )
