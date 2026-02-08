from .nodes.rename_images import LoraToolRenameImages
from .nodes.baidu_translate import LoraToolBaiduTranslateTxt
from .nodes.qwen_vl_tagging import LoraToolQwenVLTagger
from .nodes.face_crop import LoraToolFaceCrop
from .nodes.resize_images import LoraToolResizeImages
from .nodes.image_size_classifier import LoraToolImageSizeClassifier


NODE_CLASS_MAPPINGS = {
    "LoraToolRenameImages": LoraToolRenameImages,
    "LoraToolBaiduTranslateTxt": LoraToolBaiduTranslateTxt,
    "LoraToolQwenVLTagger": LoraToolQwenVLTagger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraToolRenameImages": "ComfyUI Lora Tool - 批量重命名图片",
    "LoraToolBaiduTranslateTxt": "ComfyUI Lora Tool - 百度翻译 TXT",
    "LoraToolQwenVLTagger": "ComfyUI Lora Tool - Qwen-VL 图片打标"
}

NODE_CLASS_MAPPINGS.update({
    "LoraToolFaceCrop": LoraToolFaceCrop
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "LoraToolFaceCrop": "ComfyUI Lora Tool - 批量人脸裁剪"
})

NODE_CLASS_MAPPINGS.update({
    "LoraToolResizeImages": LoraToolResizeImages
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "LoraToolResizeImages": "ComfyUI Lora Tool - 批量设置图片尺寸"
})

NODE_CLASS_MAPPINGS.update({
    "LoraToolImageSizeClassifier": LoraToolImageSizeClassifier
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "LoraToolImageSizeClassifier": "ComfyUI Lora Tool - 图片尺寸分类"
})
