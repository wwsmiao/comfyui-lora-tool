import os
import shutil
from PIL import Image

class LoraToolImageSizeClassifier:
    """
    按图片尺寸分类：
    wh: width > height
    hw: height > width
    ee: width == height
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "ComfyUI Lora Tool"

    def run(self, image_folder):
        if not os.path.isdir(image_folder):
            msg = f"[ImageSizeClassifier] 路径不存在: {image_folder}"
            print(msg)
            return (msg,)

        wh_dir = os.path.join(image_folder, "wh")
        hw_dir = os.path.join(image_folder, "hw")
        ee_dir = os.path.join(image_folder, "ee")

        os.makedirs(wh_dir, exist_ok=True)
        os.makedirs(hw_dir, exist_ok=True)
        os.makedirs(ee_dir, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

        total = 0
        wh = hw = ee = skipped = 0

        for name in os.listdir(image_folder):
            src_path = os.path.join(image_folder, name)

            if not os.path.isfile(src_path):
                continue

            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue

            # 跳过已经分类的图片
            if os.path.dirname(src_path) in (wh_dir, hw_dir, ee_dir):
                continue

            try:
                with Image.open(src_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"[ImageSizeClassifier] 无法读取图片，跳过: {name} ({e})")
                skipped += 1
                continue

            if width > height:
                dst_dir = wh_dir
                wh += 1
            elif height > width:
                dst_dir = hw_dir
                hw += 1
            else:
                dst_dir = ee_dir
                ee += 1

            dst_path = os.path.join(dst_dir, name)

            # 避免覆盖
            if os.path.exists(dst_path):
                skipped += 1
                continue

            shutil.move(src_path, dst_path)
            total += 1

        result_msg = (
            f"[ImageSizeClassifier] 分类完成\n"
            f"总处理: {total}\n"
            f"wh(宽>高): {wh}\n"
            f"hw(高>宽): {hw}\n"
            f"ee(等宽高): {ee}\n"
            f"跳过: {skipped}"
        )

        print(result_msg)
        return (result_msg,)
