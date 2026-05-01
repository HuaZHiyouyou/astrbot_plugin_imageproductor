"""
Google Gemini Imagen 图像生成 Provider
"""

from typing import Tuple, Dict, Any

from .base import BaseProvider, ImageResult


class GeminiProvider(BaseProvider):
    """Google Gemini Imagen 图像生成提供商"""

    provider_name = "gemini"
    supported_sizes = ["512x512", "1024x1024", "1792x1024", "1024x1792", "2048x2048"]
    supported_qualities = ["standard", "hd"]
    supported_styles = ["vivid", "natural", "artistic"]

    def _get_api_config(self) -> Tuple[str, str]:
        """获取当前可用的 API 配置"""
        main_key = self.config.get("main_api_key", "")
        main_url = self.config.get("main_api_url", "")
        backup_key = self.config.get("backup_api_key", "")
        backup_url = self.config.get("backup_api_url", "")

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
        """调用 Google Imagen API 生成图像"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            model = model or self.config.get("model", "imagen-3.0-generate-002")
            width, height = self._parse_size(size)

            url = f"{api_url}/v1beta/imagen:generateImage"

            payload = {
                "model": model,
                "prompt": prompt,
                "imageSize": {
                    "width": width,
                    "height": height
                },
                "personGeneration": "dont_allow"
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    image_url = result.get("image", {}).get("url", "")
                    if image_url:
                        return ImageResult(success=True, image_url=image_url)
                    return ImageResult(success=False, error="未返回图像 URL")
                else:
                    error_text = await response.text()
                    return ImageResult(success=False, error=f"API 错误: {response.status} - {error_text}")

        except Exception as e:
            return ImageResult(success=False, error=str(e))

    async def test_connection(
        self,
        api_key: str = "",
        api_url: str = ""
    ) -> Tuple[bool, str]:
        """测试 Gemini API 连接"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return False, "API Key 未配置"

            url = f"{api_url}/v1beta/models"
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
