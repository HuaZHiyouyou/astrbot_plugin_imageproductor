"""
Google Gemini 图像生成 Provider (支持以图生图)
"""

import base64
from typing import Tuple, Dict, Any, List

from .base import BaseProvider, ImageResult


class GeminiProvider(BaseProvider):
    """Google Gemini 图像生成提供商 (支持多模态输入)"""

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

    def _build_gemini_context(
        self,
        model: str,
        image_b64_list: List[tuple],
        prompt: str,
    ) -> dict:
        """构建 Gemini 多模态请求上下文"""
        parts = []
        for mime, b64 in image_b64_list:
            parts.append({
                "inlineData": {
                    "mimeType": mime,
                    "data": b64,
                }
            })
        parts.insert(0, {"text": prompt})

        return {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 1,
                "topP": 0.95,
                "maxOutputTokens": 8192,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            ],
        }

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
        """调用 Google Gemini API 生成图像 (支持多模态输入)"""
        try:
            api_key, api_url = self._get_api_config()
            if not api_key:
                return ImageResult(success=False, error="API Key 未配置")

            model = model or self.config.get("model", "gemini-2.0-flash-preview-image")
            width, height = self._parse_size(size)

            url = f"{api_url}/{model}:generateContent"

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

            if image_b64_list and len(image_b64_list) > 0:
                payload = self._build_gemini_context(model, image_b64_list, prompt)
            else:
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }
                    ],
                    "generationConfig": {
                        "responseModalities": ["IMAGE"],
                    },
                }

            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    b64_images = []
                    for item in result.get("candidates", []):
                        parts = item.get("content", {}).get("parts", [])
                        for part in parts:
                            if "inlineData" in part and "data" in part["inlineData"]:
                                data = part["inlineData"]
                                b64_images.append((data["mimeType"], data["data"]))

                    if b64_images:
                        return ImageResult(success=True, b64_json=b64_images[0][1])
                    return ImageResult(success=False, error="未返回图像数据")
                else:
                    error_text = await response.text()
                    return ImageResult(success=False, error=f"API 错误: {response.status} - {error_text}")

        except Exception as e:
            return ImageResult(success=False, error=f"生成图像异常: {str(e)}")

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
