"""
火山引擎 Vision 图像生成提供商
使用字节跳动火山引擎的视觉模型生成图像
"""
import re
from typing import Optional, Tuple
from . import BaseProvider, ImageResult, ProviderConfig


class VolcanoVisionProvider(BaseProvider):
    """火山引擎 Vision 提供商"""

    provider_name = "volcano_vision"
    supported_sizes = ["1024x1024", "1792x1024", "1024x1792"]
    supported_qualities = ["standard", "hd"]
    supported_styles = ["vivid", "natural"]

    def __init__(self, config: ProviderConfig, session):
        super().__init__(config, session)
        self.model = self.config.get("model", "doubao-vision-pro")

    def _get_api_config(self, use_vision: bool = False) -> Tuple[str, str]:
        """获取当前可用的 API 配置
        
        Args:
            use_vision: 是否使用视觉模型配置（backup_api_key/backup_api_url）
        """
        main_key = self.config.get("main_api_key", "")
        main_url = self.config.get("main_api_url", "")
        backup_key = self.config.get("backup_api_key", "")
        backup_url = self.config.get("backup_api_url", "")

        if use_vision and backup_key and backup_url:
            return backup_key, backup_url
        if main_key and main_url:
            return main_key, main_url
        if backup_key and backup_url:
            return backup_key, backup_url

        api_key = main_key or backup_key
        api_url = main_url or backup_url
        if not api_url:
            api_url = "https://ark.cn-beijing.volces.com/api/v3"
        return api_key, api_url

    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        model: str = "",
        api_key: str = "",
        api_url: str = "",
        image_b64_list: list = None,
        **kwargs
    ) -> ImageResult:
        """使用火山引擎 Vision 生成图像"""
        from astrbot.api import logger

        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            model = model or self.config.get("model", "doubao-vision-pro")

            content_parts = []

            if image_b64_list and len(image_b64_list) > 0:
                logger.info(f"[ImageProducer] 火山引擎 Vision 检测到 {len(image_b64_list)} 张参考图片")
                for i, (mime, b64_data) in enumerate(image_b64_list, start=1):
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64_data}",
                            "detail": "high"
                        }
                    })

                content_parts.append({
                    "type": "text",
                    "text": f"""你是一个专业的图像生成提示词工程师。请根据参考图片和用户需求，生成一个极其详细的英文图像生成提示词。

【重要规则】
1. 必须准确描述参考图片的所有视觉元素：主体、颜色、构图、风格、光线、氛围、背景等
2. 用户需求应该融入参考图片的视觉风格中，而不是替代它
3. 生成的提示词必须以英文撰写
4. 提示词应该足够详细（50-200个单词）

用户需求：{prompt}

请直接返回英文提示词，不要有任何其他内容。"""
                })
            else:
                content_parts.append({
                    "type": "text",
                    "text": f"""你是一个专业的图像生成提示词工程师。请根据用户需求生成一个详细的英文图像生成提示词。

用户需求：{prompt}

请直接返回英文提示词，不要有任何其他内容。"""
                })

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                "max_tokens": 2000
            }

            url = f"{api_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]

                        match = re.search(r"!\[.*?\]\((.*?)\)", content)
                        if match:
                            img_src = match.group(1)
                            if img_src.startswith("data:image/"):
                                header, b64_data = img_src.split(",", 1)
                                return ImageResult(success=True, b64_json=b64_data)
                            else:
                                return ImageResult(success=True, image_url=img_src)

                        return ImageResult(success=False, error=f"火山引擎 Vision未返回图片")
                else:
                    error_text = await response.text()
                    logger.error(f"[ImageProducer] 火山引擎 Vision API错误: {response.status} - {error_text}")
                    return ImageResult(success=False, error=f"API错误: {response.status}")

        except Exception as e:
            logger.error(f"[ImageProducer] 火山引擎 Vision生成异常: {e}", exc_info=True)
            return ImageResult(success=False, error=str(e))

        return ImageResult(success=False, error="未知错误")
