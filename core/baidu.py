"""
百度文心一言 图像生成 Provider (整合普通生成和Chat模式)
支持：
1. 普通模式：使用 images/generations API 生成图像
2. Chat模式：使用 chat/completions API 通过视觉模型分析并返回图片
当有参考图片时自动切换到Chat模式
"""

import re
from typing import Tuple, List

from .base import BaseProvider, ImageResult


class BaiduProvider(BaseProvider):
    """百度文心一言 图像生成提供商 (支持普通模式和Chat模式自动切换)"""

    provider_name = "baidu"
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
            logger.info(f"[ImageProducer] 百度Chat检测到 {len(image_b64_list)} 张参考图片")
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

        logger.info(f"[ImageProducer] 正在调用百度 Chat API...")

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"[ImageProducer] 百度 Chat API返回: {result}")

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]

                    match = re.search(r"!\[.*?\]\((.*?)\)", content)
                    if match:
                        img_src = match.group(1)
                        if img_src.startswith("data:image/"):
                            header, b64_data = img_src.split(",", 1)
                            logger.info(f"[ImageProducer] 成功从百度Chat响应中提取图片")
                            return ImageResult(success=True, b64_json=b64_data)
                        else:
                            logger.info(f"[ImageProducer] 从百度Chat响应获取到URL图片")
                            return ImageResult(success=True, image_url=img_src)

                    logger.warning(f"[ImageProducer] 百度Chat未返回图片")
                    return ImageResult(success=False, error=f"百度Chat未返回图片")

                return ImageResult(success=False, error="百度 Chat API返回格式异常")
            else:
                error_text = await response.text()
                logger.error(f"[ImageProducer] 百度 Chat API错误: {response.status} - {error_text}")
                return ImageResult(success=False, error=f"API 错误: {response.status}")

    async def _analyze_reference_images(
        self,
        api_url: str,
        api_key: str,
        image_b64_list: List[tuple],
        original_prompt: str,
        use_chinese: bool = True
    ) -> str:
        """使用百度 ERNIE-VL 分析参考图片，生成详细提示词"""
        try:
            url = f"{api_url}/rpc/2.0/ai_custom/v1/wenxinworkshop/ernievilg_v/async_gen_images"
            headers = {
                "Content-Type": "application/json"
            }

            image_count = len(image_b64_list)
            first_image = image_b64_list[0]
            mime, b64_data = first_image

            if use_chinese:
                analysis_prompt = f"""请分析这{image_count}张参考图片，并创建一个极其详细的图像生成提示词。

用户想要生成的内容: {original_prompt}

基于所有参考图片，请创建一个详细的提示词：
1. 描述每张参考图片的视觉风格、构图、主体、颜色、光线和氛围
2. 识别所有参考图片的共同风格特征和视觉元素
3. 保留参考图片的关键视觉特征
4. 结合用户的需求: {original_prompt}
5. 提示词应包含：主体、环境、艺术风格、光线、色彩、构图、视角、质量关键词

只返回提示词文本，不要返回其他内容。提示词应该150-300字，足够详细以便用于图像生成。"""
            else:
                analysis_prompt = f"""Please analyze these {image_count} reference images and create an extremely detailed image generation prompt.

The user wants to generate: {original_prompt}

Based on ALL reference images, create a detailed prompt:
1. Describe the visual style, composition, subject, colors, lighting, and atmosphere of each reference image
2. Identify common style features and visual elements across all reference images
3. Preserve key visual features from the references
4. Incorporate the user's request: {original_prompt}
5. The prompt should include: subject, environment, art style, lighting, colors, composition, camera angle, quality keywords

Only return the prompt text, nothing else. The prompt should be 150-300 words, detailed enough for image generation."""

            payload = {
                "access_token": api_key,
                "prompt": analysis_prompt,
                "image_base64": b64_data,
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "data" in result and result["data"]:
                        enhanced_prompt = result["data"].get("prompt", original_prompt)
                        return enhanced_prompt

            return original_prompt
        except Exception as e:
            from astrbot.api import logger
            logger.warning(f"[ImageProducer] 百度分析参考图片失败: {e}")
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
        """使用普通图像生成API生成图像
        支持多模态模型直接接收图片输入
        """
        from astrbot.api import logger

        width, height = self._parse_size(size)

        headers = {
            "Content-Type": "application/json"
        }

        # 检测是否支持多模态输入的模型
        is_multimodal = self._is_multimodal_model(model)

        if is_multimodal and image_b64_list and len(image_b64_list) > 0:
            # 多模态模型：使用 Chat API 传入图片+文字
            logger.info(f"[ImageProducer] 检测到多模态模型 {model}，使用 Chat API 传入 {len(image_b64_list)} 张图片")
            
            url = f"{api_url}"
            
            content_parts = []
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
                "text": prompt
            })

            payload = {
                "access_token": api_key,
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                "max_tokens": 8192,
            }
        else:
            # 传统文生图模型：使用图像生成端点
            url = f"{api_url}"
            payload = {
                "access_token": api_key,
                "text": prompt,
                "resolution": f"{width}x{height}",
            }

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                
                # Chat API 返回格式
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    content = message.get("content", "")
                    if content:
                        return ImageResult(success=True, b64_json=content)
                
                # Images API 返回格式
                if "result" in result:
                    image_url = result["result"].get("image")
                    if image_url:
                        return ImageResult(success=True, image_url=image_url)
                        
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
        vision_processed: bool = False,
        **kwargs
    ) -> ImageResult:
        """调用百度文心一言 API 生成图像
        智能模式：
        1. 多模态模型（如 ERNIE-VL）：直接传入图片+文字
        2. 传统模型+有参考图片：先通过视觉模型分析，再生成
        3. 传统模型+无参考图片：直接文字生成
        """
        from astrbot.api import logger

        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            has_images = bool(image_b64_list and len(image_b64_list) > 0)
            gen_model = model or self.config.get("model", "ernie-vilg-v2")
            
            # 检测是否是多模态模型
            is_multimodal = self._is_multimodal_model(gen_model)

            if is_multimodal and has_images:
                # 多模态模型：直接传入图片+文字，不需要视觉模型分析
                logger.info(f"[ImageProducer] 使用多模态模型 {gen_model}，直接传入 {len(image_b64_list)} 张图片")
                api_key, api_url = self._get_api_config(use_vision=False)
                return await self._generate_with_images_api(
                    prompt, gen_model, size, api_key, api_url, image_b64_list, **kwargs
                )
            elif has_images and auto_switch_mode and not vision_processed:
                # 传统模型+有参考图片：先通过视觉模型分析（仅当 main.py 未处理时）
                logger.info(f"[ImageProducer] 检测到参考图片，使用视觉模型分析后生成")
                vision_api_key, vision_api_url = self._get_api_config(use_vision=True)
                enhanced_prompt = await self._analyze_reference_images(
                    vision_api_url, vision_api_key, image_b64_list, prompt
                )
                logger.info(f"[ImageProducer] 视觉模型分析完成，使用增强提示词生成图像")
                api_key, api_url = self._get_api_config(use_vision=False)
                return await self._generate_with_images_api(
                    enhanced_prompt, gen_model, size, api_key, api_url, image_b64_list, **kwargs
                )
            else:
                # 传统模型+无参考图片 或 vision_processed=True（已由 main.py 处理）：直接使用传入的 prompt
                if vision_processed and has_images:
                    logger.info(f"[ImageProducer] 视觉分析已在 main.py 完成，直接使用最终提示词")
                else:
                    logger.info(f"[ImageProducer] 使用普通图像生成模式")
                api_key, api_url = self._get_api_config(use_vision=False)
                return await self._generate_with_images_api(
                    prompt, gen_model, size, api_key, api_url, image_b64_list, **kwargs
                )

        except Exception as e:
            logger.error(f"[ImageProducer] 百度生成图片失败: {e}", exc_info=True)
            return ImageResult(success=False, error=str(e))

    async def test_connection(
        self,
        api_key: str = "",
        api_url: str = ""
    ) -> Tuple[bool, str]:
        """测试文心一言 API 连接"""
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
