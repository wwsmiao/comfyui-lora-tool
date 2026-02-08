import os
import json
from ..utils.baidu_api import baidu_translate


class LoraToolBaiduTranslateTxt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": ""}),
                "translate_mode": ([
                    "中文 → 英文",
                    "英文 → 中文"
                ],),
                "save_mode": ([
                    "覆盖原文",
                    "保存到指定路径"
                ],)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate_txt"
    CATEGORY = "comfyui-lora-tool"

    def translate_txt(self, input_path, output_path, translate_mode, save_mode):
        if not os.path.isdir(input_path):
            return ("输入路径不存在",)

        config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "baidu_key.json")
        with open(config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        appid, key = cfg.get("id"), cfg.get("key")
        if not appid or not key:
            return ("baidu_key.json 配置错误",)

        from_lang, to_lang = ("zh", "en") if "中文" in translate_mode else ("en", "zh")

        if save_mode == "保存到指定路径":
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = input_path

        files = [f for f in os.listdir(input_path) if f.endswith(".txt")]
        count = 0

        for f in files:
            src = os.path.join(input_path, f)
            dst = os.path.join(output_path, f)

            with open(src, "r", encoding="utf-8") as r:
                txt = r.read().strip()
            if not txt:
                continue

            try:
                res = baidu_translate(txt, from_lang, to_lang, appid, key)
            except Exception as e:
                return (str(e),)

            with open(dst, "w", encoding="utf-8") as w:
                w.write(res)

            count += 1

        return (f"翻译完成，共处理 {count} 个 txt 文件",)
