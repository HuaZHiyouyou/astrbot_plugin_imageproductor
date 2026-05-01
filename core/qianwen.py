"""
阿里云 千问 图像生成 Provider
"""

from typing import Tuple, Dict, Any

from .base import BaseProvider, ImageResult


class QianwenProvider(BaseProvider):
    """阿里云 千问 图像生成提供商"""

    provider_name = "qianwen"
    supported_sizes = ["512x512", "1024x1024", "1792x1024", "1024x1792"]
    supported_qualities = ["standard", "hd"]
    supported_styles = ["vivid", "natural", "realistic", "anime", "illustration"]

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
        """调用阿里云千问 API 生成图像"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            model = model or self.config.get("model", "qwen-vl-plus")
            width, height = self._parse_size(size)

            url = f"{api_url}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": f"{width}x{height}",
            }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if "output" in result:
                        image_url = result["output"].get("url")
                        if image_url:
                            return ImageResult(success=True, image_url=image_url)
                    return ImageResult(success=False, error="API 返回格式异常")
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
        """测试千问 API 连接"""
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
