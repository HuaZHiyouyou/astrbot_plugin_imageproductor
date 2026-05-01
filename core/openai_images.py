"""
OpenAI DALL-E 图像生成 Provider
"""

import base64
from typing import Tuple, Dict, Any

from .base import BaseProvider, ImageResult


class OpenAIProvider(BaseProvider):
    """OpenAI DALL-E 图像生成提供商"""

    provider_name = "openai"
    supported_sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    supported_qualities = ["standard", "hd"]
    supported_styles = ["vivid", "natural"]

    def _is_zhipu_compatible(self, api_url: str) -> bool:
        """检测是否为智谱兼容API"""
        zhipu_keywords = ["zhipu", "zhiyuan", "zhipuai", "api.zhipu.ai"]
        return any(keyword in api_url.lower() for keyword in zhipu_keywords)

    def _is_custom_compatible(self, api_url: str) -> bool:
        """检测是否为自定义兼容API（非官方OpenAI）"""
        official_domains = ["api.openai.com"]
        return not any(domain in api_url.lower() for domain in official_domains)

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
        """调用 OpenAI DALL-E API 生成图像"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            model = model or self.config.get("model", "dall-e-3")
            width, height = self._parse_size(size)

            if image_b64_list and len(image_b64_list) > 0:
                return await self._generate_with_image(
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    image_b64_list=image_b64_list,
                    size=f"{width}x{height}",
                    quality=quality,
                    style=style
                )
            else:
                return await self._generate_text_only(
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    width=width,
                    height=height,
                    quality=quality,
                    style=style
                )

        except Exception as e:
            return ImageResult(success=False, error=str(e))

    async def _generate_text_only(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        width: int,
        height: int,
        quality: str,
        style: str
    ) -> ImageResult:
        """纯文字生成图像"""
        url = f"{api_url}/images/generations"
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

        is_zhipu = self._is_zhipu_compatible(api_url)
        is_custom = self._is_custom_compatible(api_url)

        if is_zhipu:
            pass
        elif is_custom:
            pass
        else:
            payload["quality"] = quality
            payload["style"] = style

        if model == "dall-e-2":
            payload.pop("quality", None)
            payload.pop("style", None)

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

    async def _generate_with_image(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        image_b64_list: list,
        size: str,
        quality: str,
        style: str
    ) -> ImageResult:
        """以图生图（编辑参考图）"""
        import base64
        from io import BytesIO
        from PIL import Image
        from curl_cffi import CurlMime

        url = f"{api_url}/images/edits"
        headers = {"Authorization": f"Bearer {api_key}"}

        payload = {
            "model": model,
            "prompt": prompt,
        }

        multipart = CurlMime()
        for index, (mime, b64_data) in enumerate(image_b64_list, start=1):
            raw_bytes = base64.b64decode(b64_data)
            try:
                with Image.open(BytesIO(raw_bytes)) as img:
                    if getattr(img, "is_animated", False):
                        img.seek(0)
                    img = img.convert("RGB")
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=100)
                    file_content = buf.getvalue()
                    file_mime = "image/jpeg"
            except Exception:
                if mime in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
                    file_content = raw_bytes
                    file_mime = mime
                else:
                    return ImageResult(success=False, error="不支持的图片格式")

            multipart.addpart(
                name="image",
                content_type=file_mime,
                filename=f"image_{index}.jpg",
                data=file_content
            )

            if prompt:
                multipart.addpart(
                    name="prompt",
                    content_type="text/plain",
                    data=prompt.encode("utf-8")
                )

        try:
            async with self.session.post(url, headers=headers, data=payload, multipart=multipart) as response:
                multipart.close()
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
            return ImageResult(success=False, error=f"以图生图请求失败: {str(e)}")

    async def test_connection(
        self,
        api_key: str = "",
        api_url: str = ""
    ) -> Tuple[bool, str]:
        """测试 OpenAI API 连接"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return False, "API Key 未配置"

            is_zhipu = self._is_zhipu_compatible(api_url)

            if is_zhipu:
                url = f"{api_url}/models"
            else:
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
