"""
AstrBot 多平台AI图像生成插件 v2.0
支持 GPT/DALL-E、Gemini、Grok、Seed、智谱、千问、文心一言、混元、Stable Diffusion 等多个 AI 图像生成平台
支持工具调用、命令别名、前缀匹配和白名单
"""

import asyncio
import os
import base64
import aiohttp
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from PIL import Image

import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools
from astrbot.api import logger
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.utils.session_waiter import SessionController, session_waiter

from .core import (
    BaseProvider,
    ImageResult,
    OpenAIProvider,
    GeminiProvider,
    GrokProvider,
    SeedProvider,
    ZhipuProvider,
    QianwenProvider,
    BaiduProvider,
    HunyuanProvider,
    StableDiffusionProvider,
    ClaudeVisionProvider,
    DeepSeekVisionProvider,
    VolcanoVisionProvider,
    StepFunVisionProvider,
)
from .core.llm_tools import ImageProducerPromptTool, ImageProducerGenerateTool, ImageProducerPresetTool


# 支持的图片文件格式
SUPPORTED_FILE_FORMATS_WITH_DOT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".mpo",
)

# API类型到Provider名称的映射
API_TYPE_TO_PROVIDER = {
    "OpenAI": "openai",
    "Gemini": "gemini",
    "Grok": "grok",
    "Zhipu": "zhipu",
    "Qianwen": "qianwen",
    "Baidu": "baidu",
    "Hunyuan": "hunyuan",
    "Seed": "seed",
    "Stable_Diffusion": "stable_diffusion",
}

# 平台到视觉模型Provider的映射
PLATFORM_TO_VISION_PROVIDER = {
    "OpenAI": "openai",
    "Gemini": "gemini",
    "Grok": "grok",
    "Zhipu": "zhipu",
    "Qianwen": "qianwen",
    "Baidu": "baidu",
    "Hunyuan": "hunyuan",
}

PROVIDER_CLASS_MAP = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "grok": GrokProvider,
    "seed": SeedProvider,
    "zhipu": ZhipuProvider,
    "qianwen": QianwenProvider,
    "baidu": BaiduProvider,
    "hunyuan": HunyuanProvider,
    "stable_diffusion": StableDiffusionProvider,
    "claude_vision": ClaudeVisionProvider,
    "deepseek_vision": DeepSeekVisionProvider,
    "volcano_vision": VolcanoVisionProvider,
    "stepfun_vision": StepFunVisionProvider,
}


class ImageProducer(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config

        self.provider_configs: Dict[str, Dict[str, Any]] = {}

        prefix_config = self.conf.get("prefix_config", {}).get("items", {})
        self.prefix_enabled: bool = prefix_config.get("prefix_enabled", False)
        prefix_list_str: str = prefix_config.get("prefix_list", "/img,/aimg")
        self.prefix_list: list = [p.strip() for p in prefix_list_str.split(",") if p.strip()] if prefix_list_str else []
        self.coexist_enabled: bool = prefix_config.get("coexist_enabled", False)

        whitelist_config = self.conf.get("whitelist_config", {}).get("items", {})
        self.group_whitelist_enabled: bool = whitelist_config.get("whitelist_enabled", False)
        group_whitelist_str: str = whitelist_config.get("whitelist_groups", "")
        self.group_whitelist: list = [g.strip() for g in group_whitelist_str.split(",") if g.strip()] if group_whitelist_str else []
        self.user_whitelist_enabled: bool = whitelist_config.get("user_enabled", False)
        self.user_whitelist: list = []

        self.provider_map: Dict[str, BaseProvider] = {}

        data_dir = StarTools.get_data_dir("astrbot_plugin_imageproductor")
        self.ai_images_dir = data_dir / "ai_images"
        self.refer_images_dir = data_dir / "refer_images"
        self.save_dir = data_dir / "save_images"
        os.makedirs(self.ai_images_dir, exist_ok=True)
        os.makedirs(self.refer_images_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.session: Optional[aiohttp.ClientSession] = None

        common_config = self.conf.get("common_config", {}).get("items", {})

        provider_type = "OpenAI"
        self.default_platform = "openai"
        self.auto_switch_mode = True
        self.default_size = common_config.get("default_size", "1024x1024")
        self.default_quality = common_config.get("default_quality", "standard")
        self.default_style = common_config.get("default_style", "vivid")
        self.max_concurrent_jobs = common_config.get("max_concurrent_jobs", 5)
        self.enable_nsfw_filter = common_config.get("enable_nsfw_filter", True)
        self.auto_save_images = common_config.get("auto_save_images", True)
        self.save_images = common_config.get("auto_save_images", True)
        self.max_retry = common_config.get("max_retry", 3)
        self.proxy = common_config.get("proxy", "")
        self.timeout = common_config.get("timeout", 300)

        gather_mode_config = self.conf.get("gather_mode_config", {}).get("items", {})
        self.gather_mode_enabled = gather_mode_config.get("gather_mode_enabled", False)

        llm_tool_settings = self.conf.get("llm_tool_settings", {}).get("items", {})
        self.llm_tool_enabled = llm_tool_settings.get("llm_tool_enabled", False)

        self.refer_images = ""  # 暂时为空，预留功能
        self.preset_prompt_list = self.conf.get("preset_prompts", [])
        self.preset_prompt_dict: Dict[str, str] = {}
        self.parse_preset_prompts()

        self.semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        self.running_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        session_kwargs = {
            "timeout": aiohttp.ClientTimeout(total=self.timeout)
        }
        if self.proxy:
            session_kwargs["proxy"] = self.proxy
            logger.info(f"[ImageProducer] 已配置代理: {self.proxy}")
        self.session = aiohttp.ClientSession(**session_kwargs)
        self.init_providers()

        if self.llm_tool_enabled:
            self.context.add_llm_tools(ImageProducerGenerateTool(plugin=self))
            logger.info("[ImageProducer] 已注册 LLM 工具: img_producer_generate")
            
            self.context.add_llm_tools(ImageProducerPromptTool(plugin=self))
            logger.info("[ImageProducer] 已注册 LLM 工具: img_producer_prompt")
            
            self.context.add_llm_tools(ImageProducerPresetTool(plugin=self))
            logger.info("[ImageProducer] 已注册 LLM 工具: img_producer_preset")

    def init_providers(self):
        """从配置结构初始化Providers"""
        provider_keys = ["main_provider", "back_provider", "back_provider2", "back_provider3", "back_provider4", "back_provider5"]
        is_main = True

        for key in provider_keys:
            provider_container = self.conf.get(key, {})
            provider_obj = provider_container.get("items", {})
            if isinstance(provider_obj, dict) and provider_obj.get("enabled", False):
                self._init_single_provider(provider_obj, is_main=is_main)
            is_main = False

    def _init_single_provider(self, provider_obj: Dict[str, Any], is_main: bool = True):
        """初始化单个提供商"""
        api_type = provider_obj.get("api_type", "")
        provider_name = API_TYPE_TO_PROVIDER.get(api_type)

        if not provider_name:
            logger.warning(f"[ImageProducer] 不支持的API类型: {api_type}")
            return

        api_key = provider_obj.get("api_key", "")
        if not api_key:
            logger.warning(f"[ImageProducer] {provider_obj.get('api_name', '提供商')} 未配置API密钥")
            return

        provider_class = PROVIDER_CLASS_MAP.get(provider_name)
        if not provider_class:
            logger.warning(f"[ImageProducer] 未找到Provider类: {provider_name}")
            return

        provider_config = {
            "enabled": True,
            "model": provider_obj.get("model", ""),
            "vision_model": provider_obj.get("vision_model", ""),
            "main_api_key": api_key,
            "main_api_url": provider_obj.get("api_url", ""),
            "backup_api_key": provider_obj.get("vision_api_key", ""),
            "backup_api_url": provider_obj.get("vision_api_url", ""),
            "api_name": provider_obj.get("api_name", "主提供商" if is_main else "备用提供商"),
            "auto_switch": provider_obj.get("auto_switch", True),
        }

        if self.session:
            provider_instance = provider_class(provider_config, self.session)
            self.provider_map[provider_name] = provider_instance
            self.provider_configs[provider_name] = provider_config
            logger.info(f"[ImageProducer] 已加载 {'主' if is_main else '备用'} provider: {provider_name} ({provider_config['api_name']})")

            if is_main:
                self.default_platform = provider_name
                self.auto_switch_mode = provider_config.get("auto_switch", True)

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        try:
            return self.context.is_global_admin(event)
        except Exception:
            return False

    def is_group_allowed(self, event: AstrMessageEvent) -> bool:
        if not self.group_whitelist_enabled:
            return True
        if hasattr(event, 'group_id'):
            group_id = str(event.group_id)
            return group_id in self.group_whitelist
        return True

    def is_user_allowed(self, event: AstrMessageEvent) -> bool:
        if not self.user_whitelist_enabled:
            return True
        if hasattr(event, 'user_id'):
            user_id = str(event.user_id)
            return user_id in self.user_whitelist
        return True

    def parse_preset_prompts(self):
        """解析预设提示词"""
        self.preset_prompt_dict = {}
        for item in self.preset_prompt_list:
            item = item.strip()
            if not item:
                continue
            try:
                tokens = item.split(maxsplit=1)
                if len(tokens) < 2:
                    logger.warning(f"[ImageProducer] 预设提示词格式错误（缺少提示词内容）: {item}")
                    continue
                
                trigger_raw = tokens[0]
                prompt_content = tokens[1]
                
                # 解析触发词
                trigger_list = []
                if trigger_raw.startswith("[") and trigger_raw.endswith("]"):
                    # 多触发词
                    trigger_list = [t.strip() for t in trigger_raw[1:-1].split(",") if t.strip()]
                else:
                    # 单触发词
                    trigger_list = [trigger_raw]
                
                # 注册触发词
                for trigger in trigger_list:
                    if trigger:
                        self.preset_prompt_dict[trigger] = prompt_content
                        logger.debug(f"[ImageProducer] 已注册预设触发词: {trigger}")
            except Exception as e:
                logger.warning(f"[ImageProducer] 解析预设提示词失败: {item}, 错误: {e}")
        logger.info(f"[ImageProducer] 已加载 {len(self.preset_prompt_dict)} 个预设触发词")

    async def _load_refer_images(self) -> List[Tuple[str, str]]:
        """加载预设参考图片"""
        image_b64_list: List[Tuple[str, str]] = []
        if not self.refer_images:
            return image_b64_list

        filenames = [f.strip() for f in self.refer_images.split(",") if f.strip()]
        for filename in filenames:
            try:
                path = self.refer_images_dir / filename
                if path.exists():
                    mime_type = self._get_mime_type(filename)
                    with open(path, "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode("utf-8")
                    image_b64_list.append((mime_type, b64_data))
                    logger.info(f"[ImageProducer] 已加载预设参考图片: {filename}")
                else:
                    logger.warning(f"[ImageProducer] 预设参考图片不存在: {filename}")
            except Exception as e:
                logger.error(f"[ImageProducer] 加载预设参考图片失败 {filename}: {e}")

        if image_b64_list:
            logger.info(f"[ImageProducer] 共加载 {len(image_b64_list)} 张预设参考图片")
        return image_b64_list

    def _get_mime_type(self, filename: str) -> str:
        """根据文件扩展名获取MIME类型"""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
        }
        return mime_map.get(ext, "image/png")

    def get_preset_prompt(self, trigger: str, user_text: str = "") -> Optional[str]:
        """获取并处理预设提示词"""
        if trigger not in self.preset_prompt_dict:
            return None
        
        preset_prompt = self.preset_prompt_dict[trigger]
        
        # 替换占位符
        if "{{user_text}}" in preset_prompt:
            preset_prompt = preset_prompt.replace("{{user_text}}", user_text)
        elif user_text:
            # 如果没有占位符，但有用户文本，将用户文本添加到末尾
            preset_prompt = f"{preset_prompt}, {user_text}"
        
        return preset_prompt

    def _get_message_text(self, event: AstrMessageEvent) -> str:
        if hasattr(event, 'message_str'):
            return event.message_str.strip()
        if hasattr(event, 'get_plain_text'):
            return event.get_plain_text().strip()
        if hasattr(event, 'message'):
            msg = event.message
            if hasattr(msg, 'extract_plain_text'):
                return msg.extract_plain_text().strip()
            return str(msg).strip()
        if hasattr(event, 'content'):
            return str(event.content).strip()
        return ""

    def _extract_prompt(self, text: str) -> str:
        import re
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'^/img\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^/aimg\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^/生图\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^/ai生图\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bimg\s*', '', text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_image_urls_from_event(self, event: AstrMessageEvent) -> List[str]:
        """从事件中提取所有的图片 URL，包括引用回复中的图片"""
        image_urls: List[str] = []
        try:
            if hasattr(event, 'get_messages'):
                for comp in event.get_messages():
                    logger.debug(f"[ImageProducer] 消息组件类型: {type(comp).__name__}, 内容: {comp}")
                    if isinstance(comp, Comp.Reply) and hasattr(comp, 'chain'):
                        # 处理引用回复中的图片
                        for quote in comp.chain:
                            if isinstance(quote, Comp.Image) and quote.url:
                                image_urls.append(quote.url)
                            elif (
                                isinstance(quote, Comp.File)
                                and quote.url
                                and quote.url.startswith("http")
                                and quote.url.lower().endswith(SUPPORTED_FILE_FORMATS_WITH_DOT)
                            ):
                                image_urls.append(quote.url)
                    elif isinstance(comp, Comp.Image) and comp.url:
                        image_urls.append(comp.url)
                    elif (
                        isinstance(comp, Comp.File)
                        and comp.url
                        and comp.url.startswith("http")
                        and comp.url.lower().endswith(SUPPORTED_FILE_FORMATS_WITH_DOT)
                    ):
                        image_urls.append(comp.url)
        except Exception as e:
            logger.warning(f"[ImageProducer] 提取图片 URL 失败: {e}")
        return list(dict.fromkeys(image_urls))  # 去重

    @staticmethod
    def _handle_image(image_bytes: bytes) -> Tuple[str, str]:
        """尝试把图片统一转换成 jpeg 格式，返回 (mime, base64)"""
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                if getattr(img, "is_animated", False):
                    img.seek(0)
                img = img.convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=100)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return ("image/jpeg", b64)
        except Exception as e:
            logger.warning(f"[ImageProducer] 图片处理失败: {e}")
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            return ("image/jpeg", b64)

    async def _fetch_image(self, url: str) -> Optional[Tuple[str, str]]:
        """下载单张图片并转换为 (mime, base64)"""
        for _ in range(3):
            content, success = await self._download_image(url)
            if content is not None:
                return content
            if content is None and success:
                return None
        return None

    async def _fetch_images(self, image_urls: List[str]) -> List[Tuple[str, str]]:
        """下载多张图片并转换为 (mime, base64) 列表"""
        image_b64_list: List[Tuple[str, str]] = []
        for url in image_urls:
            for _ in range(3):
                content, success = await self._download_image(url)
                if content is not None:
                    image_b64_list.append(content)
                    break
                if content is None and success:
                    break
        return image_b64_list

    async def _download_image(self, url: str) -> Tuple[Optional[Tuple[str, str]], bool]:
        """下载图片并返回 (mime, base64) 和是否下载成功的标志"""
        try:
            if not self.session:
                return None, False
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"[ImageProducer] 图片下载失败，状态码: {response.status}")
                    return None, False
                content = await response.read()
                if not content or len(content) > 50 * 1024 * 1024:
                    logger.warning("[ImageProducer] 图片超过 50MB，跳过处理")
                    return None, True
                result = await asyncio.to_thread(ImageProducer._handle_image, content)
                return result, True
        except asyncio.TimeoutError:
            logger.error(f"[ImageProducer] 网络请求超时: {url}")
            return None, False
        except Exception as e:
            logger.error(f"[ImageProducer] 下载图片失败: {url}，错误信息：{e}")
            return None, False

    async def _safe_reply(self, event: AstrMessageEvent, message: str):
        """
        真正100%可靠的消息回复方法
        尝试多种发送方式，直到成功或所有方式都失败
        """
        last_error = None
        tried_methods = []
        
        # 方法1: event.send (最简单)
        try:
            tried_methods.append("event.send")
            await event.send(MessageChain().message(message))
            logger.info(f"[ImageProducer] 消息发送成功 (方法: event.send)")
            return True
        except Exception as e:
            last_error = e
            logger.warning(f"[ImageProducer] event.send 失败: {e}")
        
        # 方法2: event.reply
        try:
            tried_methods.append("event.reply")
            await event.reply(MessageChain().message(message))
            logger.info(f"[ImageProducer] 消息发送成功 (方法: event.reply)")
            return True
        except Exception as e:
            last_error = e
            logger.warning(f"[ImageProducer] event.reply 失败: {e}")
        
        # 方法3: 直接通过 message_obj 发送
        try:
            tried_methods.append("message_obj")
            if hasattr(event, 'message_obj') and hasattr(event.message_obj, 'reply'):
                await event.message_obj.reply(message)
                logger.info(f"[ImageProducer] 消息发送成功 (方法: message_obj.reply)")
                return True
        except Exception as e:
            last_error = e
            logger.warning(f"[ImageProducer] message_obj.reply 失败: {e}")
        
        # 所有方法都失败了
        logger.error(f"[ImageProducer] 所有消息发送方法都失败了! 尝试过: {tried_methods}, 最后错误: {last_error}")
        return False

    @filter.command("img", alias={"aimg", "生图", "ai生图"})
    async def generate_image_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用图像生成功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用图像生成功能")
            return

        gather_mode = self.conf.get("gather_mode_enabled", False)

        if gather_mode:
            await self._generate_image_gather_mode(event)
        else:
            await self._generate_image_direct_mode(event)

    async def _generate_image_direct_mode(self, event: AstrMessageEvent):
        """直接模式：立即生成图片"""
        text = self._get_message_text(event)
        prompt = self._extract_prompt(text)

        # 检查是否是预设触发词
        # 格式：/img <触发词> <文本>
        if prompt:
            parts = prompt.split(maxsplit=1)
            if parts:
                trigger = parts[0].strip()
                user_text = parts[1].strip() if len(parts) > 1 else ""
                
                # 检查是否是预设触发词
                preset_prompt = self.get_preset_prompt(trigger, user_text)
                if preset_prompt:
                    logger.info(f"[ImageProducer] 检测到预设触发词: {trigger}")
                    prompt = preset_prompt

        image_urls = self._extract_image_urls_from_event(event)
        logger.info(f"[ImageProducer] 提取到的图片 URL: {image_urls}")
        image_b64_list = []
        if image_urls:
            logger.info(f"[ImageProducer] 检测到 {len(image_urls)} 张图片，开始下载...")
            image_b64_list = await self._fetch_images(image_urls)
            if image_b64_list:
                logger.info(f"[ImageProducer] 成功下载 {len(image_b64_list)} 张图片")

        if not prompt and not image_b64_list:
            prompt = "Generate a similar image based on the reference image, maintaining the same style, composition, and mood"
        elif prompt and image_b64_list:
            prompt = f"{prompt}, reference image style, maintain the artistic approach and composition from the reference"
        elif not prompt:
            await self._safe_reply(event, "💡 请提供图像生成提示词\n示例: /img 一只可爱的猫咪在草地上玩耍\n使用预设: /img 手办化 一只猫")
            return

        await self._generate_and_send(event, prompt, image_b64_list)

    async def _generate_image_gather_mode(self, event: AstrMessageEvent):
        """收集模式：收集多张图片和文本后生成"""
        text = self._get_message_text(event)
        prompt = self._extract_prompt(text)

        image_urls = self._extract_image_urls_from_event(event)
        image_b64_list = []
        if image_urls:
            logger.info(f"[ImageProducer] 收集模式：检测到 {len(image_urls)} 张图片...")
            image_b64_list = await self._fetch_images(image_urls)

        if prompt:
            prompt = f"{prompt}, reference image style, maintain the artistic approach and composition from the reference"

        operator_id = event.get_sender_id()
        is_cancel = False

        await self._safe_reply(event, f"""📝 <b>绘图收集模式已启用</b>
提示词：{prompt or "（待输入）"}
图片：{len(image_b64_list)} 张

💡 <b>操作说明：</b>
• 发送图片可追加参考图
• 发送文字可追加到提示词
• 发送「<b>开始</b>」立即生成
• 发送「<b>取消</b>」退出操作
• 60 秒内有效""")

        @session_waiter(timeout=60, record_history_chains=False)
        async def waiter(controller: SessionController, sub_event: AstrMessageEvent):
            nonlocal is_cancel, prompt, image_b64_list

            if sub_event.get_sender_id() != operator_id:
                return

            sub_text = self._get_message_text(sub_event).strip()

            if sub_text == "取消":
                is_cancel = True
                await self._safe_reply(sub_event, "✅ 操作已取消")
                controller.stop()
                return

            if sub_text == "开始":
                if not prompt and not image_b64_list:
                    await self._safe_reply(sub_event, "❌ 没有可用的提示词或参考图，请先发送内容")
                    controller.keep(timeout=60, reset_timeout=True)
                    return
                controller.stop()
                return

            sub_image_urls = self._extract_image_urls_from_event(sub_event)
            if sub_image_urls:
                sub_b64_list = await self._fetch_images(sub_image_urls)
                image_b64_list.extend(sub_b64_list)
                logger.info(f"[ImageProducer] 收集模式：追加 {len(sub_b64_list)} 张图片")

            sub_prompt = self._extract_prompt(sub_text)
            if sub_prompt:
                if prompt:
                    prompt = f"{prompt} {sub_prompt}"
                else:
                    prompt = sub_prompt

            if not prompt and not image_b64_list:
                await self._safe_reply(sub_event, "❌ 还没有输入任何内容，请发送图片或文字")
            else:
                await self._safe_reply(sub_event, f"""📝 <b>已收集：</b>
提示词：{prompt or "（待输入）"}
图片：{len(image_b64_list)} 张

💡 继续发送内容，或发送「<b>开始</b>」生成""")

            controller.keep(timeout=60, reset_timeout=True)

        try:
            await waiter(event)
        except Exception as e:
            logger.error(f"[ImageProducer] 收集模式出错: {e}", exc_info=True)
            await self._safe_reply(event, "❌ 处理时发生错误")
            return
        finally:
            if is_cancel:
                return

        if not prompt and not image_b64_list:
            await self._safe_reply(event, "❌ 收集超时，已取消操作")
            return

        if not prompt:
            prompt = "Generate a similar image based on the reference image"

        await self._generate_and_send(event, prompt, image_b64_list)

    @filter.command("提示词", alias={"prompt", "生提示词"})
    async def generate_prompt_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用此功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用此功能")
            return

        text = self._get_message_text(event)
        text = text.replace('/提示词', '').replace('/prompt', '').replace('/生提示词', '').strip()

        if not text:
            await self._safe_reply(event, "💡 请提供图像描述，我将为你生成专业提示词\n示例: /提示词 一只可爱的猫咪在草地上玩耍")
            return

        prompt = await self._generate_prompt(text)
        if prompt:
            await self._safe_reply(event, f"✨ 生成的提示词:\n{prompt}")
        else:
            await self._safe_reply(event, "❌ 提示词生成失败，请稍后重试")

    async def _generate_prompt(self, user_description: str) -> Optional[str]:
        try:
            logger.info(f"[ImageProducer] 开始生成提示词，描述: {user_description[:50]}...")
            result = await asyncio.wait_for(
                self._generate_prompt_internal(user_description),
                timeout=60  # 60秒超时
            )
            if result:
                logger.info(f"[ImageProducer] 提示词生成成功")
            else:
                logger.warning(f"[ImageProducer] 提示词生成返回空值")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[ImageProducer] 生成提示词超时")
            return None
        except Exception as e:
            logger.error(f"[ImageProducer] 生成提示词异常: {e}", exc_info=True)
            return None
    
    async def _generate_prompt_internal(self, user_description: str) -> Optional[str]:
        """
        内部提示词生成 - 尝试调用 AstrBot 的 LLM 能力
        """
        try:
            from astrbot.api.star import StarTools
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a professional AI image generation prompt engineer. Your task is to convert simple user descriptions into detailed, high-quality image generation prompts. Requirements: English only, detailed description of the scene, subject, style, lighting, colors, etc., add quality keywords like masterpiece, best quality, ultra-detailed, high quality, 8k, etc. The generated prompt should be 100-300 words long. Just return the prompt directly, do not include any additional explanations."""
                },
                {
                    "role": "user",
                    "content": f"Please convert this description into a professional image prompt: {user_description}"
                }
            ]
            
            # 使用 AstrBot 的 LLM 能力
            try:
                result = await StarTools.llm_completion(messages)
                if result and hasattr(result, 'get'):
                    content = result.get('content', '')
                    if content:
                        return content.strip()
                
                # 尝试其他可能的返回格式
                if isinstance(result, str):
                    return result.strip()
                elif hasattr(result, 'content'):
                    return str(result.content).strip()
                elif isinstance(result, dict) and 'content' in result:
                    return str(result['content']).strip()
                    
            except Exception as llm_err:
                logger.error(f"[ImageProducer] LLM调用失败: {llm_err}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"[ImageProducer] 提示词生成内部异常: {e}", exc_info=True)
            return None
        return None
        
    async def _llm_tool_job(self, event: AstrMessageEvent, prompt: str, size: str = None, image_b64_list: list = None) -> Dict[str, Any]:
        """
        LLM工具调用的后台任务
        返回字典形式的结果
        """
        try:
            logger.info(f"[ImageProducer] LLM工具任务开始: {prompt[:50]}...")
            use_size = size if size else self.default_size
            
            async with self.semaphore:
                result = await asyncio.wait_for(
                    self.generate_image_internal(
                        platform=self.default_platform,
                        prompt=prompt,
                        size=use_size,
                        quality=self.default_quality,
                        style=self.default_style,
                        image_b64_list=image_b64_list,
                    ),
                    timeout=180
                )
            
            if result.success:
                logger.info(f"[ImageProducer] 图像生成成功")
                image_b64_data = None
                save_path = None
                
                if result.image_data:
                    image_b64_data = base64.b64encode(result.image_data).decode('utf-8')
                    save_path = await self._save_image_to_ai_images(result.image_data, prompt)
                elif result.image_url:
                    save_path = await self._download_and_save_to_ai_images(result.image_url, prompt)
                    if save_path:
                        try:
                            with open(save_path, 'rb') as f:
                                image_b64_data = base64.b64encode(f.read()).decode('utf-8')
                        except:
                            pass
                
                return {
                    "success": True,
                    "image_b64": image_b64_data,
                    "save_path": save_path,
                    "image_url": result.image_url
                }
            else:
                logger.error(f"[ImageProducer] 图像生成失败: {result.error}")
                return {
                    "success": False,
                    "error": result.error
                }
                
        except Exception as e:
            logger.error(f"[ImageProducer] LLM工具任务异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @filter.command("生成", alias={"gen", "做图"})
    async def two_stage_generate_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用此功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用此功能")
            return

        text = self._get_message_text(event)
        text = text.replace('/生成', '').replace('/gen', '').replace('/做图', '').strip()

        # 从事件中提取图片 URL 并下载
        image_urls = self._extract_image_urls_from_event(event)
        image_b64_list = []
        if image_urls:
            logger.info(f"[ImageProducer] 检测到 {len(image_urls)} 张图片，开始下载...")
            image_b64_list = await self._fetch_images(image_urls)
            if image_b64_list:
                logger.info(f"[ImageProducer] 成功下载 {len(image_b64_list)} 张图片")

        if not text and not image_b64_list:
            await self._safe_reply(event, "💡 请提供图像描述，我将为你生成专业提示词并创作图像\n示例: /生成 一只可爱的猫咪在草地上玩耍")
            return

        await self._safe_reply(event, f"🎨 正在生成专业提示词...\n描述: {text or '基于参考图片'}")

        prompt = await self._generate_prompt(text or "基于参考图片生成图像")
        if not prompt:
            await self._safe_reply(event, "❌ 提示词生成失败，尝试直接生成图像...")
            prompt = text or "根据参考图片生成图像"

        await self._safe_reply(event, f"✨ 提示词生成完成\n正在生成图像...")
        await self._generate_and_send(event, prompt, image_b64_list)

    async def _generate_and_send(self, event: AstrMessageEvent, prompt: str, image_b64_list: list = None):
        """
        生成图片并发送给用户，确保100%给用户回复
        
        发送策略：
        1. 无论如何先发送简单的文本消息（最容易成功）
        2. 然后尝试发送图片
        3. 最后尝试发送URL或附加信息
        """
        # === 阶段 1: 准备 ===
        try:
            if not self.provider_map:
                await self._safe_reply(event, "❌ 未配置任何图像生成平台，请先在插件设置中配置API")
                return
            logger.info(f"[ImageProducer] 开始生成图像，提示词: {prompt[:50]}...")
            
            # 先发送"正在生成"的消息（这个必须先成功！）
            await self._safe_reply(event, f"🎨 正在生成图像...\n提示词: {prompt}")
        except Exception as e:
            logger.error(f"[ImageProducer] 发送开始消息失败，但继续尝试: {e}", exc_info=True)
        
        # 加载预设参考图片并合并
        refer_images_b64 = await self._load_refer_images()
        if image_b64_list is None:
            image_b64_list = refer_images_b64
        elif refer_images_b64:
            image_b64_list = refer_images_b64 + image_b64_list
            logger.info(f"[ImageProducer] 合并预设参考图片后共 {len(image_b64_list)} 张")
        
        # === 阶段 2: 生成图片 ===
        result = None
        save_path = None
        image_b64_data = None
        final_message = "✅ 图像生成完成！"

        # 智能选择平台：有图片时自动使用视觉模型
        target_platform = self.default_platform
        has_images = bool(image_b64_list and len(image_b64_list) > 0)
        
        if has_images and self.auto_switch_mode:
            api_type = self.conf.get("provider_type", "")
            vision_provider_name = PLATFORM_TO_VISION_PROVIDER.get(api_type)
            if vision_provider_name and vision_provider_name in self.provider_map:
                target_platform = vision_provider_name
                logger.info(f"[ImageProducer] 检测到图片，自动切换到视觉模型: {target_platform}")
            else:
                logger.info(f"[ImageProducer] 检测到图片，但未配置视觉模型，继续使用默认平台: {target_platform}")
        
        try:
            try:
                async with self.semaphore:
                    result = await asyncio.wait_for(
                        self.generate_image_internal(
                            platform=target_platform,
                            prompt=prompt,
                            size=self.default_size,
                            quality=self.default_quality,
                            style=self.default_style,
                            image_b64_list=image_b64_list,
                        ),
                        timeout=180  # 3分钟超时
                    )
            except asyncio.TimeoutError:
                logger.error(f"[ImageProducer] 生成图像超时（3分钟）")
                final_message = "⏰ 图像生成超时了！请稍后重试。"
                result = None
            except Exception as e:
                logger.error(f"[ImageProducer] 生成过程异常: {e}", exc_info=True)
                final_message = f"⚠️ 生成出错: {str(e)}"
                result = None
        except Exception as e:
            logger.error(f"[ImageProducer] 阶段2总异常: {e}", exc_info=True)
            final_message = "❌ 生成失败，但已尝试处理"
        
        # === 阶段 3: 处理结果（保存到本地）===
        try:
            if result and result.success:
                logger.info("[ImageProducer] 图像生成成功，处理数据...")
                try:
                    if result.image_data:
                        save_path = await self._save_image_to_ai_images(result.image_data, prompt)
                        image_b64_data = base64.b64encode(result.image_data).decode('utf-8')
                    elif result.image_url:
                        logger.info(f"[ImageProducer] 下载图片: {result.image_url}")
                        save_path = await self._download_and_save_to_ai_images(result.image_url, prompt)
                        if save_path:
                            try:
                                with open(save_path, 'rb') as f:
                                    image_b64_data = base64.b64encode(f.read()).decode('utf-8')
                            except Exception as read_err:
                                logger.warning(f"[ImageProducer] 读取保存的图片失败: {read_err}")
                except Exception as e:
                    logger.error(f"[ImageProducer] 处理图片数据异常: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[ImageProducer] 阶段3总异常: {e}", exc_info=True)
        
        # === 阶段 4: 发送给用户（关键！） ===
        # 策略：先送简单文本，再尝试复杂的
        try:
            # --- 4.1 先发送一个简单的成功/失败消息 ---
            if result and result.success:
                base_message = "✅ 图像生成成功！"
            elif result and not result.success:
                base_message = f"❌ 图像生成失败: {result.error}"
            else:
                base_message = final_message
            
            # 先发送这个基础消息！（确保用户能收到）
            await self._safe_reply(event, base_message)
            logger.info("[ImageProducer] 基础消息已发送")
            
            # --- 4.2 尝试发送更丰富的消息 ---
            if result and result.success:
                # 尝试1: 发送图片
                if image_b64_data:
                    try:
                        msg_chain: list[BaseMessageComponent] = [
                            Comp.Reply(id=event.message_obj.message_id) if hasattr(event.message_obj, 'message_id') else None,
                            Comp.Image.fromBase64(image_b64_data)
                        ]
                        msg_chain = [c for c in msg_chain if c is not None]
                        await event.send(MessageChain(chain=msg_chain))
                        logger.info("[ImageProducer] 图片发送成功")
                        
                        # 成功发送图片后，再发送保存路径
                        if save_path:
                            await self._safe_reply(event, f"📁 已保存到: {save_path}")
                    except Exception as pic_err:
                        logger.error(f"[ImageProducer] 发送图片失败: {pic_err}")
                        # 发送图片失败，发送URL
                        if result.image_url:
                            await self._safe_reply(event, f"🎨 图片链接:\n{result.image_url}")
                # 尝试2: 只发送URL
                elif result.image_url:
                    await self._safe_reply(event, f"🎨 图片链接:\n{result.image_url}")
                
                # 保存路径提示（如果还没发送过）
                if save_path and not image_b64_data:
                    await self._safe_reply(event, f"📁 已保存到: {save_path}")
        
        except Exception as send_err:
            logger.error(f"[ImageProducer] 阶段4消息发送异常: {send_err}", exc_info=True)
        
        # === 阶段 5: 最后的保险！ ===
        # 我们不管怎么样，一定要再发送一个确认消息
        try:
            # 延迟一小会儿，确保前面的消息有机会发送
            await asyncio.sleep(0.1)
        except Exception:
            pass
        
        # === 完成 ===
        logger.info("[ImageProducer] 整个流程结束")

    async def generate_image_internal(
        self,
        platform: str,
        prompt: str,
        size: str,
        quality: str,
        style: str,
        model: str = "",
        image_b64_list: list = None
    ) -> ImageResult:
        provider = self.provider_map.get(platform)
        if not provider:
            return ImageResult(success=False, error=f"未找到平台: {platform}")

        last_error = None

        # 获取所有备用Provider（provider_map中除了主provider外的都是备用）
        fallback_providers = []
        for pname, p in self.provider_map.items():
            if pname != platform:
                fallback_providers.append((pname, p))

        # 构建完整的Provider列表（主Provider在前，备用Provider在后）
        all_providers = [(platform, provider)] + fallback_providers

        for provider_name, current_provider in all_providers:
            for attempt in range(self.max_retry):
                try:
                    result = await current_provider.generate_image(
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        style=style,
                        model=model,
                        image_b64_list=image_b64_list,
                    )
                    if result.success:
                        if provider_name != platform:
                            logger.info(f"[ImageProducer] 主Provider失败，备用Provider {provider_name} 成功生成图像")
                        return result
                    last_error = result.error
                    logger.warning(f"[ImageProducer] Provider {provider_name} 生成失败: {last_error}")
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[ImageProducer] Provider {provider_name} 生成异常: {e}", exc_info=True)

        return ImageResult(success=False, error=last_error or "所有Provider生成图像失败")

    async def _save_image(self, image_data: bytes, prompt: str) -> Optional[str]:
        if not self.save_images:
            return None
        
        try:
            import re
            clean_prompt = re.sub(r'[\\/*?:"<>|]', '', prompt[:50])
            timestamp = asyncio.get_event_loop().time()
            filename = f"{clean_prompt}_{int(timestamp)}.jpg"
            save_path = self.save_dir / filename
            with open(save_path, 'wb') as f:
                f.write(image_data)
            return str(save_path)
        except Exception as e:
            logger.error(f"[ImageProducer] 保存图片失败: {e}", exc_info=True)
            return None

    async def _save_image_to_ai_images(self, image_data: bytes, prompt: str) -> Optional[str]:
        """保存图片到 ai_images 目录"""
        try:
            import re
            clean_prompt = re.sub(r'[\\/*?:"<>|]', '', prompt[:50])
            timestamp = asyncio.get_event_loop().time()
            filename = f"{clean_prompt}_{int(timestamp)}.jpg"
            save_path = self.ai_images_dir / filename
            with open(save_path, 'wb') as f:
                f.write(image_data)
            return str(save_path)
        except Exception as e:
            logger.error(f"[ImageProducer] 保存图片到 ai_images 失败: {e}", exc_info=True)
            return None

    async def _download_and_save_to_ai_images(self, url: str, prompt: str) -> Optional[str]:
        """下载图片并保存到 ai_images 目录"""
        try:
            if not self.session:
                return None
            async with self.session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return await self._save_image_to_ai_images(image_data, prompt)
                else:
                    logger.warning(f"[ImageProducer] 下载图片失败，状态码: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"[ImageProducer] 下载并保存图片失败: {e}", exc_info=True)
            return None

    @filter.command("img列表", alias={"图片列表", "预设列表"})
    async def list_presets_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用此功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用此功能")
            return

        if not self.preset_prompt_dict:
            await self._safe_reply(event, "📋 暂无预设提示词")
            return

        presets_list = "\n".join([f"• {trigger}" for trigger in self.preset_prompt_dict.keys()])
        await self._safe_reply(event, f"📋 可用预设触发词:\n{presets_list}\n\n💡 使用方法: /img <触发词> <描述>")

    @filter.command("img查看", alias={"图片查看", "预设查看"})
    async def view_preset_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用此功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用此功能")
            return

        text = self._get_message_text(event)
        text = text.replace('/img查看', '').replace('/图片查看', '').replace('/预设查看', '').strip()

        if not text:
            await self._safe_reply(event, "💡 请指定要查看的预设触发词\n示例: /img查看 手办化")
            return

        preset_prompt = self.preset_prompt_dict.get(text)
        if preset_prompt:
            await self._safe_reply(event, f"📝 预设「{text}」的内容:\n{preset_prompt}")
        else:
            await self._safe_reply(event, f"❌ 未找到预设触发词: {text}")