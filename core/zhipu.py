"""
智谱 AI 图像生成 Provider (整合普通生成和Chat模式)
支持：
1. 普通模式：使用 /images/generations API 生成图像
2. Chat模式：使用 /chat/completions API 通过视觉模型分析并返回图片
当有参考图片时自动切换到Chat模式
"""

import base64
import re
from typing import Tuple, Dict, Any, List

from .base import BaseProvider, ImageResult


class ZhipuProvider(BaseProvider):
    """智谱 AI 图像生成提供商 (支持普通模式和Chat模式自动切换)"""

    provider_name = "zhipu"
    supported_sizes = ["512x512", "1024x1024", "1792x1024", "1024x1792"]
    supported_qualities = ["standard", "hd"]
    supported_styles = ["vivid", "natural", "realistic", "anime", "illustration"]

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
            # 使用视觉模型配置
            return backup_key, backup_url
        if main_key and main_url:
            return main_key, main_url
        if backup_key and backup_url:
            return backup_key, backup_url
        return main_key or backup_key, main_url or backup_url

    async def _generate_with_chat_api(
        self,
        prompt: str,
        model: str,
        api_key: str,
        api_url: str,
        image_b64_list: List[tuple],
        **kwargs
    ) -> ImageResult:
        """使用Chat接口生成图像（适合有参考图片的场景）"""
        from astrbot.api import logger
        
        content_parts = []

        if image_b64_list and len(image_b64_list) > 0:
            logger.info(f"[ImageProducer] 智谱Chat检测到 {len(image_b64_list)} 张参考图片")
            for i, (mime, b64_data) in enumerate(image_b64_list, start=1):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64_data}",
                        "detail": "high"
                    }
                })
                logger.info(f"[ImageProducer] 已添加第 {i} 张参考图片到请求")

            content_parts.append({
                "type": "text",
                "text": f"""你是一个专业的图像生成提示词工程师。请根据参考图片和用户需求，生成一个极其详细的英文图像生成提示词。

【重要规则】
1. 必须准确描述参考图片的所有视觉元素：主体、颜色、构图、风格、光线、氛围、背景等
2. 用户需求应该融入参考图片的视觉风格中，而不是替代它
3. 生成的提示词必须以英文撰写，这对图像生成模型很重要
4. 提示词应该足够详细（50-200个单词），包含所有重要的视觉细节

用户需求：{prompt}

请直接返回英文提示词，不要有任何其他内容。"""
            })
        else:
            content_parts.append({
                "type": "text",
                "text": f"""你是一个专业的图像生成提示词工程师。请根据用户需求生成一个详细的英文图像生成提示词。

用户需求：{prompt}

请直接返回英文提示词，不要有任何其他内容。提示词应该详细描述图像内容、风格、光线、氛围等元素。"""
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

        logger.info(f"[ImageProducer] 正在调用智谱Chat API生成图片...")

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"[ImageProducer] 智谱Chat API返回: {result}")

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]

                    match = re.search(r"!\[.*?\]\((.*?)\)", content)
                    if match:
                        img_src = match.group(1)
                        if img_src.startswith("data:image/"):
                            header, b64_data = img_src.split(",", 1)
                            logger.info(f"[ImageProducer] 成功从Chat响应中提取图片")
                            return ImageResult(success=True, b64_json=b64_data)
                        else:
                            logger.info(f"[ImageProducer] 从Chat响应获取到URL图片: {img_src[:50]}...")
                            return ImageResult(success=True, image_url=img_src)

                    logger.warning(f"[ImageProducer] 智谱Chat未返回图片，返回内容: {content[:500]}")
                    return ImageResult(success=False, error=f"智谱Chat未返回图片，内容: {content[:200]}")

                return ImageResult(success=False, error="智谱Chat API返回格式异常")
            else:
                error_text = await response.text()
                logger.error(f"[ImageProducer] 智谱Chat API错误: {response.status} - {error_text}")
                return ImageResult(success=False, error=f"API 错误: {response.status} - {error_text}")

    async def _analyze_reference_images(
        self,
        api_url: str,
        api_key: str,
        image_b64_list: List[tuple],
        original_prompt: str
    ) -> str:
        """使用智谱 GLM-4V 分析参考图片，生成详细提示词"""
        from astrbot.api import logger
        try:
            logger.info(f"[ImageProducer] 智谱开始分析 {len(image_b64_list)} 张参考图片")

            url = f"{api_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            content_parts = []
            for i, (mime, b64_data) in enumerate(image_b64_list):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64_data}",
                        "detail": "high"
                    }
                })

            analysis_prompt = f"""请极其详细地分析这张参考图片，并严格基于图片内容创建图像生成提示词。

【重要规则】
1. 你必须准确描述图片中的：主体、颜色、构图、风格、光线、氛围等所有视觉元素
2. 生成的提示词必须包含参考图片的所有关键视觉特征
3. 提示词应该以英文撰写（这对图像生成模型很重要）
4. 用户想要的内容应该融入参考图片的视觉风格中，而不是替代它

用户想要生成的内容: {original_prompt}

请按以下格式返回（只返回提示词，不要其他内容）：
[英文详细描述，融合参考图片风格和用户需求]"""

            content_parts.append({
                "type": "text",
                "text": analysis_prompt
            })

            payload = {
                "model": "glm-4v",
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                "max_tokens": 2000
            }

            logger.info(f"[ImageProducer] 正在调用智谱GLM-4V分析图片...")

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()

                    if "choices" in result and len(result["choices"]) > 0:
                        enhanced_prompt = result["choices"][0]["message"]["content"].strip()
                        logger.info(f"[ImageProducer] 增强后的提示词: {enhanced_prompt}")
                        return enhanced_prompt

                return original_prompt
        except Exception as e:
            logger.error(f"[ImageProducer] 智谱分析参考图片失败: {e}", exc_info=True)
            return original_prompt

    async def _generate_with_images_api(
        self,
        prompt: str,
        model: str,
        size: str,
        api_key: str,
        api_url: str,
        image_b64_list: List[tuple] = None,
        **kwargs
    ) -> ImageResult:
        """使用普通图像生成API生成图像"""
        from astrbot.api import logger
        
        width, height = self._parse_size(size)

        if image_b64_list and len(image_b64_list) > 0:
            enhanced_prompt = await self._analyze_reference_images(
                api_url, api_key, image_b64_list, prompt
            )
        else:
            enhanced_prompt = prompt

        url = f"{api_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "prompt": enhanced_prompt,
            "n": 1,
            "size": f"{width}x{height}",
        }

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                if "data" in result and len(result["data"]) > 0:
                    if "url" in result["data"][0]:
                        image_url = result["data"][0]["url"]
                        return ImageResult(success=True, image_url=image_url)
                    elif "b64_json" in result["data"][0]:
                        b64_data = result["data"][0]["b64_json"]
                        return ImageResult(success=True, b64_json=b64_data)
                return ImageResult(success=False, error="API 返回格式异常")
            else:
                error_text = await response.text()
                return ImageResult(success=False, error=f"API 错误: {response.status} - {error_text}")

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
        auto_switch_mode: bool = True,
        **kwargs
    ) -> ImageResult:
        """调用智谱 AI API 生成图像
        智能模式：当有参考图片时自动使用Chat模式，否则使用普通模式
        """
        from astrbot.api import logger
        
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            has_images = bool(image_b64_list and len(image_b64_list) > 0)
            
            # 智能选择模式
            if has_images and auto_switch_mode:
                logger.info(f"[ImageProducer] 检测到参考图片，使用视觉模型分析后生成")
                # 先用视觉模型分析图片，得到增强提示词
                vision_api_key, vision_api_url = self._get_api_config(use_vision=True)
                vision_model = model or self.config.get("vision_model", "glm-4v")
                enhanced_prompt = await self._analyze_reference_images(
                    vision_api_url, vision_api_key, image_b64_list, prompt
                )
                logger.info(f"[ImageProducer] 视觉模型分析完成，使用增强提示词生成图像")
                # 再用图像生成 API 生成图片
                api_key, api_url = self._get_api_config(use_vision=False)
                gen_model = model or self.config.get("model", "models/text-image-generation")
                return await self._generate_with_images_api(
                    enhanced_prompt, gen_model, size, api_key, api_url, None, **kwargs
                )
            else:
                logger.info(f"[ImageProducer] 使用普通图像生成模式")
                api_key, api_url = self._get_api_config(use_vision=False)
                gen_model = model or self.config.get("model", "models/text-image-generation")
                return await self._generate_with_images_api(
                    prompt, gen_model, size, api_key, api_url, image_b64_list, **kwargs
                )

        except Exception as e:
            logger.error(f"[ImageProducer] 智谱生成图片失败: {e}", exc_info=True)
            return ImageResult(success=False, error=str(e))

    async def test_connection(
        self,
        api_key: str = "",
        api_url: str = ""
    ) -> Tuple[bool, str]:
        """测试智谱 API 连接"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return False, "API Key 未配置"

            url = f"{api_url}/models"
            headers = {
                "Authorization": f"Bearer {api_key}",
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return True, "连接成功"
                elif response.status == 401:
                    return False, "API Key 无效"
                else:
                    return False, f"API 错误: {response.status}"

        except Exception as e:
            return False, f"连接失败: {str(e)}"
