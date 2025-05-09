"""
ComfyUI 图像到关键词节点
通过大模型API将图片反向转换为关键词描述
"""

from .image_to_prompt_node import ImageToPromptNode

# 这个函数允许节点在ComfyUI中注册
NODE_CLASS_MAPPINGS = {
    "ImageToPrompt": ImageToPromptNode
}

# 节点在UI中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToPrompt": "图像到关键词"
} 