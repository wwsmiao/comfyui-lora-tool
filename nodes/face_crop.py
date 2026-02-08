import os
import cv2
import torch
import numpy as np

from ..utils.face_detect import detect_faces, crop_and_resize_face


class LoraToolFaceCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "image_exts": ("STRING", {"default": ".jpg,.png,.jpeg,.webp"})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "result")
    FUNCTION = "run"
    CATEGORY = "comfyui-lora-tool"

    def run(self, input_path, output_path, target_width, target_height, image_exts):

        if not os.path.isdir(input_path):
            return (None, "输入路径不存在")

        os.makedirs(output_path, exist_ok=True)

        exts = [e.strip().lower() for e in image_exts.split(",")]

        images = [
            f for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in exts
        ]

        if not images:
            return (None, "未找到图片")

        out_tensors = []
        success = 0

        for idx, name in enumerate(images, start=1):
            print(f"[FaceCrop] {idx}/{len(images)} {name}")

            path = os.path.join(input_path, name)
            img = cv2.imread(path)
            if img is None:
                continue

            faces = detect_faces(img)
            if len(faces) == 0:
                continue

            # 最大人脸
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            face = faces[0]

            cropped = crop_and_resize_face(
                img,
                face,
                target_width,
                target_height
            )

            # 保存
            out_name = f"face_{name}"
            cv2.imwrite(os.path.join(output_path, out_name), cropped)

            # 转为 ComfyUI IMAGE tensor
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float() / 255.0
            out_tensors.append(tensor)

            success += 1

        if len(out_tensors) == 0:
            return (None, "未成功裁剪任何人脸")

        batch = torch.stack(out_tensors, dim=0)

        return (batch, f"人脸裁剪完成：{success}/{len(images)}")
