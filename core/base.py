"""
Provider 抽象基类
定义 AI 图像生成平台的通用接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class ImageResult:
    """图像生成结果"""
    success: bool
    image_data: Optional[bytes] = None
    image_url: Optional[str] = None
    error: Optional[str] = None
    b64_json: Optional[str] = None


@dataclass
class ProviderConfig:
    """提供商配置"""
    api_name: str
    api_type: str
    enabled: bool
    main_api_key: str = ""
    main_api_url: str = ""
    backup_api_key: str = ""
    backup_api_url: str = ""
    model: str = ""


class BaseProvider(ABC):
    """AI 图像生成提供商抽象基类"""

    # 子类必须定义的类属性
    provider_name: str = ""
    supported_sizes: list = ["512x512", "1024x1024", "1792x1024", "1024x1792"]
    supported_qualities: list = ["standard", "hd", "ultra"]
    supported_styles: list = ["vivid", "natural", "realistic", "anime", "illustration"]

    def __init__(self, config: Dict[str, Any], session: Any):
        """初始化提供商

        Args:
            config: 完整插件配置
            session: HTTP 会话
        """
        self.config = config
        self.session = session

    @abstractmethod
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
        """生成图像

        Args:
            prompt: 提示词
            size: 图像尺寸
            quality: 图像质量
            style: 图像风格
            model: 模型名称
            api_key: API 密钥
            api_url: API URL
            image_b64_list: 图片base64列表 [(mime, b64_data), ...]
            **kwargs: 其他参数

        Returns:
            ImageResult: 图像生成结果
        """
        pass

    @abstractmethod
    async def test_connection(
        self,
        api_key: str = "",
        api_url: str = ""
    ) -> Tuple[bool, str]:
        """测试连接

        Args:
            api_key: API 密钥
            api_url: API URL

        Returns:
            Tuple[bool, str]: (是否成功, 错误信息)
        """
        pass

    @classmethod
    def get_provider_class(cls, provider_type: str) -> Optional[type]:
        """根据提供商类型获取提供商类

        Args:
            provider_type: 提供商类型标识

        Returns:
            Optional[type]: 提供商类，未找到返回 None
        """
        provider_map = {
            "openai": "OpenAIProvider",
            "gemini": "GeminiProvider",
            "grok": "GrokProvider",
            "seed": "SeedProvider",
            "zhipu": "ZhipuProvider",
            "qianwen": "QianwenProvider",
            "baidu": "BaiduProvider",
            "hunyuan": "HunyuanProvider",
            "stable_diffusion": "StableDiffusionProvider",
        }
        return provider_map.get(provider_type)

    def _parse_size(self, size: str) -> Tuple[int, int]:
        """解析尺寸字符串为宽高

        Args:
            size: 尺寸字符串，如 "1024x1024"

        Returns:
            Tuple[int, int]: (width, height)
        """
        if "x" in size:
            try:
                width, height = size.split("x")
                return int(width), int(height)
            except ValueError:
                pass
        return 1024, 1024

    async def _analyze_reference_images(
        self,
        api_url: str,
        api_key: str,
        image_b64_list: list,
        user_prompt: str
    ) -> str:
        """使用视觉模型分析参考图片，返回图片描述（由 AstrBot LLM 修饰为提示词）

        Args:
            api_url: 视觉模型 API URL
            api_key: 视觉模型 API Key
            image_b64_list: 图片base64列表 [(mime, b64_data), ...]
            user_prompt: 用户需求描述

        Returns:
            str: 图片描述（非完整提示词）
        """
        from astrbot.api import logger
        import re

        try:
            content_parts = []

            # 添加参考图片
            for mime, b64_data in image_b64_list:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64_data}"
                    }
                })

            # 添加分析提示 - 只返回图片描述，不生成完整提示词
            content_parts.append({
                "type": "text",
                "text": f"""请详细描述这张图片的所有视觉元素，包括：
1. 主体内容（人物、物体、场景等）
2. 颜色和色调
3. 构图和布局
4. 艺术风格
5. 光线和阴影
6. 氛围和情感
7. 背景和细节

请用中文描述，尽量详细。"""
            })

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ],
                "max_tokens": 1000
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
                        image_description = result["choices"][0]["message"]["content"]
                        # 清理可能的多余内容
                        image_description = re.sub(r'^```.*?\n', '', image_description)
                        image_description = re.sub(r'\n```$', '', image_description)
                        logger.info(f"[ImageProducer] 视觉模型分析完成，描述长度: {len(image_description)}")
                        return image_description
                else:
                    error_text = await response.text()
                    logger.error(f"[ImageProducer] 视觉模型 API 错误: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"[ImageProducer] 视觉模型分析异常: {e}", exc_info=True)

        # 如果分析失败，使用原始用户描述
        return user_prompt
