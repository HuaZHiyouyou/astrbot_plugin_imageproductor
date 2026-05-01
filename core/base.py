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
