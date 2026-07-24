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
        user_prompt: str,
        use_chinese: bool = True
    ) -> str:
        """使用视觉模型分析参考图片，返回图片描述（由 AstrBot LLM 修饰为提示词）

        Args:
            api_url: 视觉模型 API URL
            api_key: 视觉模型 API Key
            image_b64_list: 图片base64列表 [(mime, b64_data), ...]
            user_prompt: 用户需求描述
            use_chinese: 是否使用中文描述

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

            # 添加分析提示 - 根据配置决定使用中文或英文
            if use_chinese:
                analysis_prompt = f"""请详细描述这张/这些参考图片的所有视觉元素，包括：
1. 主体内容（人物、物体、场景等）
2. 颜色和色调
3. 构图和布局
4. 艺术风格
5. 光线和阴影
6. 氛围和情感
7. 背景和细节

用户需求：{user_prompt}

请用中文详细描述参考图片的视觉特征，后续会将用户需求融入其中。"""
            else:
                analysis_prompt = f"""Please describe all visual elements of these reference image(s) in detail, including:
1. Main subjects (people, objects, scenes, etc.)
2. Colors and tones
3. Composition and layout
4. Art style
5. Lighting and shadows
6. Atmosphere and mood
7. Background and details

User requirements: {user_prompt}

Please describe the visual characteristics of the reference image(s) in detail in English. The user requirements will be integrated later."""

            content_parts.append({
                "type": "text",
                "text": analysis_prompt
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

    def _get_api_config(self, use_vision: bool = False) -> Tuple[str, str]:
        """获取API配置

        Args:
            use_vision: 是否使用视觉模型配置

        Returns:
            Tuple[str, str]: (api_key, api_url)
        """
        if use_vision:
            api_key = self.config.get("backup_api_key", "")
            api_url = self.config.get("backup_api_url", "")
        else:
            api_key = self.config.get("main_api_key", "")
            api_url = self.config.get("main_api_url", "")
        return api_key, api_url

    def _build_url(self, base_url: str, path: str) -> str:
        """构建URL，自动处理末尾斜杠

        Args:
            base_url: 基础URL（如 https://api.example.com/）
            path: 路径（如 /v1/images/generations）

        Returns:
            str: 完整的URL
        """
        # 移除 base_url 末尾的斜杠
        base_url = base_url.rstrip('/')
        # 确保 path 以斜杠开头
        if not path.startswith('/'):
            path = '/' + path
        return f"{base_url}{path}"

    def _is_likely_base64(self, content: str) -> bool:
        """检测内容是否可能是 base64 编码的图片数据
        
        多重特征检测：
        1. 字符集检测：base64 字符（A-Za-z0-9+/=）占比
        2. 长度验证：长度是 4 的倍数，或以 = 结尾
        3. 解码测试：尝试解码前 100 个字符
        4. Magic Number：解码后检查图片文件头
        
        Args:
            content: 待检测的内容
            
        Returns:
            bool: 是否可能是 base64 图片数据
        """
        import base64
        
        if not content or len(content) < 100:
            return False
        
        # 移除可能的空白字符
        clean_content = content.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # 特征1: 计算 base64 字符比例
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        base64_ratio = sum(1 for c in clean_content if c in base64_chars) / len(clean_content)
        
        # 特征2: 检查长度是否是 4 的倍数（标准 base64）
        is_valid_length = len(clean_content) % 4 == 0 or clean_content.endswith('=')
        
        # 特征3: 尝试解码前 100 个字符
        can_decode = False
        has_image_header = False
        try:
            test_data = clean_content[:100] if len(clean_content) >= 100 else clean_content
            # 补齐 padding
            padding = 4 - (len(test_data) % 4)
            if padding != 4:
                test_data += '=' * padding
            decoded = base64.b64decode(test_data)
            can_decode = True
            
            # 特征4: 检查 magic number（图片文件头）
            if len(decoded) >= 8:
                # PNG: 89 50 4E 47 0D 0A 1A 0A
                if decoded[:8] == b'\x89PNG\r\n\x1a\n':
                    has_image_header = True
                # JPEG: FF D8 FF
                elif decoded[:3] == b'\xff\xd8\xff':
                    has_image_header = True
                # GIF: GIF87a or GIF89a
                elif decoded[:6] in (b'GIF87a', b'GIF89a'):
                    has_image_header = True
                # WebP: RIFF....WEBP
                elif decoded[:4] == b'RIFF' and decoded[8:12] == b'WEBP':
                    has_image_header = True
                # BMP: BM
                elif decoded[:2] == b'BM':
                    has_image_header = True
        except Exception:
            pass
        
        # 综合判断：
        # 1. 90%+ 是 base64 字符 + 长度是 4 的倍数
        # 2. 95%+ 是 base64 字符
        # 3. 可解码 + 有图片 magic number
        # 4. 长度 > 10000 + 80%+ 是 base64 字符（长数据放宽条件）
        
        if base64_ratio >= 0.90 and is_valid_length:
            return True
        if base64_ratio >= 0.95:
            return True
        if can_decode and has_image_header:
            return True
        if len(clean_content) > 10000 and base64_ratio >= 0.80:
            return True
            
        return False

    def _rotate_api_key(self):
        """轮询到下一个API Key"""
        api_keys = self.config.get("main_api_keys", [])
        if len(api_keys) <= 1:
            return
        
        current_index = self.config.get("api_key_index", 0)
        next_index = (current_index + 1) % len(api_keys)
        self.config["api_key_index"] = next_index
        self.config["main_api_key"] = api_keys[next_index]
        
        from astrbot.api import logger
        logger.info(f"[ImageProducer] API Key已轮询: {current_index} -> {next_index}")

    def _is_multimodal_model(self, model: str) -> bool:
        """检查是否是多模态模型"""
        multimodal_models = self.config.get("multimodal_models", [])
        if not multimodal_models:
            return False
        return any(m in model.lower() for m in multimodal_models)
