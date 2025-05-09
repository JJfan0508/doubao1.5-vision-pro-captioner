import torch
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIHandler:
    """API处理类，仅用于豆包vision pro模型"""
    
    @staticmethod
    def prepare_doubao_request(img_str, custom_prompt, model_name, detail_level):
        return {
            "model": model_name or "doubao-1.5-vision-pro-250328",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": custom_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}",
                                "detail": detail_level  # 使用选择的深度理解模式
                            }
                        }
                    ]
                }
            ]
        }

    @staticmethod
    def prepare_headers(api_key):
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def parse_response(response_json):
        try:
            # 豆包API响应格式解析
            if "choices" in response_json and len(response_json["choices"]) > 0:
                if "message" in response_json["choices"][0]:
                    return response_json["choices"][0]["message"]["content"]
                elif "content" in response_json["choices"][0]:
                    return response_json["choices"][0]["content"]
            return f"无法解析豆包API响应：{str(response_json)}"
        except Exception as e:
            logger.error(f"解析响应时出错：{str(e)}")
            return f"解析响应失败：{str(e)}"

class ImageToPromptNode:
    """通过豆包大模型API将图片反向转换为关键词的节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": ("STRING", {
                    "default": "doubao-1.5-vision-pro-250328",
                    "multiline": False
                }),
                "detail_level": (["high", "low", "auto"],),  # 添加图像深度理解模式选择
                "api_url": ("STRING", {
                    "default": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                    "multiline": False
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "custom_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "描述这张图片，关注以下方面：主体、风格、光线、色彩、构图、细节"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "image_to_prompt"
    CATEGORY = "分析/反向推导"

    def __init__(self):
        self.model_name_default = "doubao-1.5-vision-pro-250328"
        self.api_url_default = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.custom_prompt_default = "描述这张图片，关注以下方面：主体、风格、光线、色彩、构图、细节"
        self.detail_level_default = "high"  # 默认使用高精度模式

    def image_to_prompt(self, image, api_url, api_key, model_name, detail_level, custom_prompt):
        try:
            # 将PyTorch张量转换为PIL图像
            i = 255. * image.cpu().numpy().squeeze()
            img = Image.fromarray(i.astype('uint8'))
            
            # 将图像转换为base64字符串
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 准备请求
            headers = APIHandler.prepare_headers(api_key)
            payload = APIHandler.prepare_doubao_request(img_str, custom_prompt, model_name, detail_level)
            
            # 发送请求
            logger.info(f"正在发送请求到 {api_url}")
            logger.info(f"请求头: {headers}")
            logger.info(f"请求体: {json.dumps(payload, ensure_ascii=False)}")
            logger.info(f"使用图像深度理解模式: {detail_level}")  # 记录使用的深度理解模式
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                verify=False,  # 禁用SSL验证
                timeout=30
            )
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            logger.info(f"API响应: {json.dumps(result, ensure_ascii=False)}")
            return (APIHandler.parse_response(result),)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求错误: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)
        except Exception as e:
            error_msg = f"处理过程出错: {str(e)}"
            logger.error(error_msg)
            return (error_msg,)

# 这个函数允许节点在ComfyUI中注册
NODE_CLASS_MAPPINGS = {
    "ImageToPrompt": ImageToPromptNode
}

# 节点在UI中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToPrompt": "图像到关键词"
} 