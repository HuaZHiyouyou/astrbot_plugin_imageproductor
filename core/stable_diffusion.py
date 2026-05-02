"""
Stable Diffusion API Provider (支持以图生图)
"""

import base64
from typing import Tuple, Dict, Any, List

from .base import BaseProvider, ImageResult


class StableDiffusionProvider(BaseProvider):
    """Stable Diffusion 本地/远程 API 提供商 (支持 img2img)"""

    provider_name = "stable_diffusion"
    supported_sizes = ["512x512", "768x768", "1024x1024", "512x768", "768x512"]
    supported_qualities = ["standard"]
    supported_styles = ["vivid", "natural"]

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

        default_url = "http://127.0.0.1:7860"
        return main_key or backup_key, main_url or backup_url or default_url

    async def generate_image(
        self,
        prompt: str,
        size: str = "512x512",
        quality: str = "standard",
        style: str = "vivid",
        model: str = "",
        api_key: str = "",
        api_url: str = "",
        image_b64_list: list = None,
        **kwargs
    ) -> ImageResult:
        """调用 Stable Diffusion API 生成图像"""
        try:
            api_key, api_url = self._get_api_config()
            width, height = self._parse_size(size)

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            if image_b64_list and len(image_b64_list) > 0:
                return await self._generate_img2img(
                    api_url, headers, prompt, image_b64_list, width, height, kwargs
                )
            else:
                return await self._generate_txt2img(
                    api_url, headers, prompt, width, height, kwargs
                )

        except Exception as e:
            return ImageResult(success=False, error=str(e))

    async def _generate_txt2img(
        self,
        api_url: str,
        headers: dict,
        prompt: str,
        width: int,
        height: int,
        kwargs: dict
    ) -> ImageResult:
        """纯文字生成图像 (txt2img)"""
        url = f"{api_url}/sdapi/v1/txt2img"

        payload = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "width": width,
            "height": height,
            "steps": kwargs.get("steps", 30),
            "cfg_scale": kwargs.get("cfg_scale", 7),
            "sampler_name": kwargs.get("sampler", "Euler a"),
            "seed": kwargs.get("seed", -1),
            "batch_size": 1
        }

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()

                if result.get("nsfw_content_detected", []):
                    return ImageResult(success=False, error="检测到 NSFW 内容，已阻止生成")

                image_data = result["images"][0]

                if image_data.startswith("data:image"):
                    image_data = image_data.split(",", 1)[1]

                image_bytes = base64.b64decode(image_data)
                return ImageResult(success=True, image_data=image_bytes)
            else:
                error_text = await response.text()
                return ImageResult(success=False, error=f"API 错误: {response.status} - {error_text}")

    async def _generate_img2img(
        self,
        api_url: str,
        headers: dict,
        prompt: str,
        image_b64_list: List[tuple],
        width: int,
        height: int,
        kwargs: dict
    ) -> ImageResult:
        """以图生图 (img2img)"""
        url = f"{api_url}/sdapi/v1/img2img"

        mime, b64_data = image_b64_list[0]

        denoising_strength = kwargs.get("denoising_strength", 0.75)

        payload = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", "") + ", low quality, blurry, distorted",
            "init_images": [f"data:{mime};base64,{b64_data}"],
            "width": width,
            "height": height,
            "steps": kwargs.get("steps", 30),
            "cfg_scale": kwargs.get("cfg_scale", 7),
            "denoising_strength": denoising_strength,
            "sampler_name": kwargs.get("sampler", "Euler a"),
            "seed": kwargs.get("seed", -1),
            "batch_size": 1
        }

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()

                if result.get("nsfw_content_detected", []):
                    return ImageResult(success=False, error="检测到 NSFW 内容，已阻止生成")

                image_data = result["images"][0]

                if image_data.startswith("data:image"):
                    image_data = image_data.split(",", 1)[1]

                image_bytes = base64.b64decode(image_data)
                return ImageResult(success=True, image_data=image_bytes)
            else:
                error_text = await response.text()
                return ImageResult(success=False, error=f"API 错误: {response.status} - {error_text}")

    async def test_connection(
        self,
        api_key: str = "",
        api_url: str = ""
    ) -> Tuple[bool, str]:
        """测试 Stable Diffusion API 连接"""
        try:
            api_key, api_url = self._get_api_config()

            url = f"{api_url}/sdapi/v1/options"
            headers = {"Content-Type": "application/json"}

            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return True, "连接成功"
                elif response.status == 401:
                    return False, "API Key 无效"
                else:
                    return False, f"API 错误: {response.status}"

        except Exception as e:
            return False, f"连接失败: {str(e)}"
