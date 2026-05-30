"""
字节跳动 Seed 图像生成 Provider (支持多模态模型)
"""

from typing import Tuple, Dict, Any, List

from .base import BaseProvider, ImageResult


class SeedProvider(BaseProvider):
    """字节跳动 Seed 图像生成提供商"""

    provider_name = "seed"
    supported_sizes = ["512x512", "1024x1024", "1280x1280", "1792x1024", "1024x1792"]
    supported_qualities = ["standard", "hd"]
    supported_styles = ["vivid", "natural", "realistic"]

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
        """调用字节 Seed API 生成图像
        智能模式：
        1. 多模态模型：直接传入图片+文字
        2. 传统模型+有参考图片：先通过视觉模型分析，再生成
        3. 传统模型+无参考图片：直接文字生成
        """
        from astrbot.api import logger

        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            has_images = bool(image_b64_list and len(image_b64_list) > 0)
            gen_model = model or self.config.get("model", "seed-image")
            
            # 检测是否是多模态模型
            is_multimodal = self._is_multimodal_model(gen_model)

            width, height = self._parse_size(size)
            url = f"{api_url}/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            if is_multimodal and has_images:
                # 多模态模型：直接传入图片+文字
                logger.info(f"[ImageProducer] 使用多模态模型 {gen_model}，直接传入 {len(image_b64_list)} 张图片")
                
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
                    "model": gen_model,
                    "prompt": content_parts,
                    "n": 1,
                    "size": f"{width}x{height}",
                }
            else:
                # 传统文生图模型：只传入文字提示词
                logger.info(f"[ImageProducer] 使用普通图像生成模式")
                payload = {
                    "model": gen_model,
                    "prompt": prompt,
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

        except Exception as e:
            logger.error(f"[ImageProducer] Seed生成图片失败: {e}", exc_info=True)
            return ImageResult(success=False, error=str(e))

    async def test_connection(
        self,
        api_key: str = "",
        api_url: str = ""
    ) -> Tuple[bool, str]:
        """测试 Seed API 连接"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return False, "API Key 未配置"

            url = f"{api_url}/v1/models"
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
