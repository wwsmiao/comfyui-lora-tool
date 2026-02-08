import os
import shutil
from datetime import datetime


class LoraToolRenameImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "rename_mode": ([
                    "数字序列",
                    "日期_数字序列",
                    "前缀_数字序列",
                    "数字序列_后缀",
                    "前缀_数字序列_后缀"
                ],),
                "start_number": ("INT", {"default": 1, "min": 0}),
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
                "save_mode": ([
                    "覆盖原图",
                    "输出到指定路径"
                ],)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "rename_images"
    CATEGORY = "comfyui-lora-tool"

    def rename_images(self, input_path, output_path, rename_mode, start_number, prefix, suffix, save_mode):
        if not os.path.isdir(input_path):
            return ("输入路径不存在",)

        if save_mode == "输出到指定路径":
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = input_path

        exts = [".png", ".jpg", ".jpeg", ".webp"]
        files = sorted(f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in exts)

        today = datetime.now().strftime("%Y%m%d")
        index = start_number

        for f in files:
            _, ext = os.path.splitext(f)

            def name(i):
                if rename_mode == "日期_数字序列":
                    return f"{today}_{i}{ext}"
                if rename_mode == "前缀_数字序列":
                    return f"{prefix}_{i}{ext}"
                if rename_mode == "数字序列_后缀":
                    return f"{i}_{suffix}{ext}"
                if rename_mode == "前缀_数字序列_后缀":
                    return f"{prefix}_{i}_{suffix}{ext}"
                return f"{i}{ext}"

            src = os.path.join(input_path, f)
            dst = os.path.join(output_path, name(index))

            while os.path.exists(dst):
                index += 1
                dst = os.path.join(output_path, name(index))

            if save_mode == "覆盖原图":
                os.rename(src, dst)
            else:
                shutil.copy2(src, dst)

            index += 1

        return (f"完成，共处理 {len(files)} 张图片",)
