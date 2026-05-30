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

        prefix_config = self.conf.get("prefix_config", {})
        self.prefix_enabled: bool = prefix_config.get("prefix_enabled", False)
        prefix_list_str: str = prefix_config.get("prefix_list", "/img,/aimg")
        self.prefix_list: list = [p.strip() for p in prefix_list_str.split(",") if p.strip()] if prefix_list_str else []
        self.coexist_enabled: bool = prefix_config.get("coexist_enabled", False)

        whitelist_config = self.conf.get("whitelist_config", {})
        self.group_whitelist_enabled: bool = whitelist_config.get("whitelist_enabled", False)
        group_whitelist_str: str = whitelist_config.get("whitelist_groups", "")
        self.group_whitelist: list = [g.strip() for g in group_whitelist_str.split(",") if g.strip()] if group_whitelist_str else []
        self.user_whitelist_enabled: bool = whitelist_config.get("user_enabled", False)
        self.user_whitelist: list = []

        self.provider_map: Dict[str, BaseProvider] = {}

        data_dir = StarTools.get_data_dir("astrbot_plugin_imageproductor")
        root_data_dir = data_dir.parent.parent
        self.ai_images_dir = root_data_dir / "ai_images"
        self.refer_images_dir = data_dir / "refer_images"
        self.save_dir = data_dir / "save_images"
        os.makedirs(self.ai_images_dir, exist_ok=True)
        os.makedirs(self.refer_images_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.session: Optional[aiohttp.ClientSession] = None

        common_config = self.conf.get("common_config", {})

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

        gather_mode_config = self.conf.get("gather_mode_config", {})
        self.gather_mode_enabled = gather_mode_config.get("gather_mode_enabled", False)

        llm_tool_settings = self.conf.get("llm_tool_settings", {})
        self.llm_tool_enabled = llm_tool_settings.get("llm_tool_enabled", False)

        prompt_settings = self.conf.get("prompt_settings", {})
        self.allow_chinese_prompt = prompt_settings.get("allow_chinese_prompt", False)

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

        logger.info(f"[ImageProducer] 开始初始化提供商，配置键: {provider_keys}")
        logger.info(f"[ImageProducer] 当前配置内容: {dict(self.conf)}")

        for key in provider_keys:
            provider_obj = self.conf.get(key, {})
            logger.info(f"[ImageProducer] 检查提供商 {key}: enabled={provider_obj.get('enabled', False)}, api_type={provider_obj.get('api_type', '')}")
            if isinstance(provider_obj, dict) and provider_obj.get("enabled", False):
                self._init_single_provider(provider_obj, is_main=is_main)
            is_main = False

        if not self.provider_map:
            logger.warning(f"[ImageProducer] 未加载任何提供商")
        elif self.default_platform not in self.provider_map:
            first_platform = list(self.provider_map.keys())[0]
            self.default_platform = first_platform
            logger.info(f"[ImageProducer] 主提供商未启用，使用第一个已加载的提供商作为默认平台: {first_platform}")

        logger.info(f"[ImageProducer] 提供商初始化完成，已加载: {list(self.provider_map.keys())}, 默认平台: {self.default_platform}")

    def _init_single_provider(self, provider_obj: Dict[str, Any], is_main: bool = True):
        """初始化单个提供商"""
        api_type = provider_obj.get("api_type", "")
        provider_name = API_TYPE_TO_PROVIDER.get(api_type)
        api_name = provider_obj.get("api_name", "未命名")

        if not provider_name:
            logger.warning(f"[ImageProducer] {api_name} 不支持的API类型: {api_type}")
            return

        api_key_raw = provider_obj.get("api_key", [])
        if isinstance(api_key_raw, str):
            api_keys = [k.strip() for k in api_key_raw.split(",") if k.strip()]
        elif isinstance(api_key_raw, list):
            api_keys = [k.strip() for k in api_key_raw if k.strip()]
        else:
            api_keys = []

        if not api_keys:
            logger.warning(f"[ImageProducer] {api_name} 未配置API密钥")
            return

        provider_class = PROVIDER_CLASS_MAP.get(provider_name)
        if not provider_class:
            logger.warning(f"[ImageProducer] {api_name} 未找到Provider类: {provider_name}")
            return

        multimodal_models = self.get_multimodal_models()
        
        provider_config = {
            "enabled": True,
            "model": provider_obj.get("model", ""),
            "vision_model": provider_obj.get("vision_model", ""),
            "main_api_keys": api_keys,
            "main_api_key": api_keys[0] if api_keys else "",
            "main_api_url": provider_obj.get("api_url", ""),
            "backup_api_key": provider_obj.get("vision_api_key", ""),
            "backup_api_url": provider_obj.get("vision_api_url", ""),
            "api_name": api_name,
            "auto_switch": provider_obj.get("auto_switch", True),
            "api_key_index": 0,
            "multimodal_models": multimodal_models,
        }

        if self.session:
            provider_instance = provider_class(provider_config, self.session)
            self.provider_map[provider_name] = provider_instance
            self.provider_configs[provider_name] = provider_config
            logger.info(f"[ImageProducer] 已加载 {'主' if is_main else '备用'} provider: {provider_name} ({api_name}), API Keys: {len(api_keys)}个")

            if is_main:
                self.default_platform = provider_name
                self.auto_switch_mode = provider_config.get("auto_switch", True)
        else:
            logger.warning(f"[ImageProducer] {api_name} session 未初始化，跳过加载")

    def get_multimodal_models(self) -> list:
        """获取多模态模型列表"""
        models = self.conf.get("multimodal_models", [])
        if not models:
            return []
        if isinstance(models, str):
            return [m.strip().lower() for m in models.split(",") if m.strip()]
        return [m.strip().lower() for m in models if m.strip()]

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
        text = re.sub(r'^[/\\]*img\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^[/\\]*aimg\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^[/\\]*生图\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^[/\\]*ai生图\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bimg\s*', '', text, flags=re.IGNORECASE)
        return text.strip()

    async def _fetch_forward_message(self, event: AstrMessageEvent, forward_id: str) -> List[str]:
        """通过平台 API 获取合并转发消息中的图片 URL"""
        image_urls = []
        try:
            logger.info(f"[ImageProducer] 尝试获取转发消息内容，id={forward_id}")
            
            # 方法1：尝试通过 event.bot 调用平台 API
            if hasattr(event, 'bot') and event.bot:
                bot = event.bot
                logger.info(f"[ImageProducer] bot 类型: {type(bot).__name__}")
                
                try:
                    result = await bot.call_action('get_forward_msg', id=forward_id)
                    logger.info(f"[ImageProducer] get_forward_msg 返回结果类型: {type(result).__name__}")
                    logger.info(f"[ImageProducer] get_forward_msg 返回结果 keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                    
                    if result and isinstance(result, dict):
                        # 尝试不同的字段名
                        messages = result.get('messages') or result.get('message') or result.get('nodes') or []
                        logger.info(f"[ImageProducer] 转发消息包含 {len(messages)} 个节点")
                        
                        for i, node in enumerate(messages):
                            logger.info(f"[ImageProducer] 节点[{i}] 类型: {type(node).__name__}")
                            if isinstance(node, dict):
                                logger.info(f"[ImageProducer] 节点[{i}] keys: {node.keys()}")
                                # 尝试不同的内容字段 - 使用 message 而不是 content
                                content = node.get('message') or node.get('content') or node.get('data', {}).get('content') or []
                                logger.info(f"[ImageProducer] 节点[{i}] content 类型: {type(content).__name__}")
                                
                                if isinstance(content, str):
                                    logger.info(f"[ImageProducer] 节点[{i}] content 是字符串，尝试解析: {content[:100]}...")
                                    # content 可能是 JSON 字符串
                                    import json
                                    try:
                                        content = json.loads(content)
                                        logger.info(f"[ImageProducer] 节点[{i}] 解析后 content 类型: {type(content).__name__}")
                                    except:
                                        pass
                                
                                if isinstance(content, list):
                                    logger.info(f"[ImageProducer] 节点[{i}] content 长度: {len(content)}")
                                    for j, item in enumerate(content):
                                        item_type = type(item).__name__
                                        logger.info(f"[ImageProducer]   节点[{i}].content[{j}] 类型: {item_type}")
                                        if isinstance(item, dict):
                                            logger.info(f"[ImageProducer]   节点[{i}].content[{j}] keys: {item.keys()}")
                                            if item.get('type') == 'image':
                                                img_data = item.get('data', {})
                                                logger.info(f"[ImageProducer]   节点[{i}].content[{j}] image data keys: {img_data.keys()}")
                                                url = img_data.get('url', '')
                                                if url:
                                                    image_urls.append(url)
                                                    logger.info(f"[ImageProducer] 从转发消息中提取图片: {url[:50]}...")
                                                file_path = img_data.get('file', '')
                                                if file_path and file_path.startswith('http'):
                                                    image_urls.append(file_path)
                                                    logger.info(f"[ImageProducer] 从转发消息提取图片(file): {file_path[:50]}...")
                                else:
                                    logger.warning(f"[ImageProducer] 节点[{i}] content 不是列表类型")
                except Exception as api_err:
                    logger.warning(f"[ImageProducer] 调用 get_forward_msg API 失败: {api_err}")
            else:
                logger.warning(f"[ImageProducer] event 没有 bot 属性，无法调用 API")
            
            # 方法2：尝试从 event.message_obj.raw_message 中获取
            if not image_urls and hasattr(event, 'message_obj'):
                raw = event.message_obj.raw_message
                logger.info(f"[ImageProducer] 尝试从 raw_message 获取图片，类型: {type(raw).__name__}")
                
                if isinstance(raw, dict):
                    # 检查 message 字段
                    if 'message' in raw:
                        for msg in raw['message']:
                            if isinstance(msg, dict):
                                msg_type = msg.get('type', '')
                                if msg_type == 'node' or msg_type == 'forward':
                                    content = msg.get('data', {}).get('content', [])
                                    if isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get('type') == 'image':
                                                url = item.get('data', {}).get('url', '')
                                                if url:
                                                    image_urls.append(url)
                                                    logger.info(f"[ImageProducer] 从 raw_message 提取图片: {url[:50]}...")
                    # 检查 message_chain 字段（某些平台）
                    if 'message_chain' in raw:
                        for msg in raw['message_chain']:
                            if isinstance(msg, dict) and msg.get('type') == 'image':
                                url = msg.get('data', {}).get('url', '')
                                if url:
                                    image_urls.append(url)
                
        except Exception as e:
            logger.error(f"[ImageProducer] 获取转发消息失败: {e}", exc_info=True)
        
        return image_urls

    async def _extract_image_urls_from_event_async(self, event: AstrMessageEvent) -> List[str]:
        """从事件中提取所有的图片 URL，包括引用回复和转发消息中的图片（异步版本）"""
        image_urls: List[str] = []
        try:
            if hasattr(event, 'get_messages'):
                messages = event.get_messages()
                logger.info(f"[ImageProducer] 消息组件总数: {len(messages)}")
                for comp in messages:
                    comp_type = type(comp).__name__
                    logger.info(f"[ImageProducer] 消息组件类型: {comp_type}")
                    
                    if isinstance(comp, Comp.Reply):
                        logger.info(f"[ImageProducer] 发现 Reply 组件，id={comp.id}, chain长度={len(comp.chain) if comp.chain else 0}")
                        if hasattr(comp, 'chain') and comp.chain:
                            for i, quote in enumerate(comp.chain):
                                quote_type = type(quote).__name__
                                logger.info(f"[ImageProducer]   Quote[{i}] 类型: {quote_type}")
                                # 递归处理 Reply chain 中的组件
                                await self._extract_image_from_component_async(quote, image_urls, event)
                        else:
                            logger.warning(f"[ImageProducer] Reply 组件没有 chain 属性或 chain 为空")
                    elif isinstance(comp, Comp.Forward):
                        logger.info(f"[ImageProducer] 发现 Forward 组件，id={comp.id}")
                        # Forward 组件只包含 id，需要通过平台 API 获取内容
                        forward_urls = await self._fetch_forward_message(event, str(comp.id))
                        image_urls.extend(forward_urls)
                    elif isinstance(comp, Comp.Nodes):
                        logger.info(f"[ImageProducer] 发现 Nodes 组件")
                        if hasattr(comp, 'nodes') and comp.nodes:
                            for node in comp.nodes:
                                await self._extract_image_from_node_async(node, image_urls)
                    elif isinstance(comp, Comp.Node):
                        logger.info(f"[ImageProducer] 发现 Node 组件")
                        await self._extract_image_from_node_async(comp, image_urls)
                    else:
                        await self._extract_image_from_component_async(comp, image_urls, event)
        except Exception as e:
            logger.warning(f"[ImageProducer] 提取图片 URL 失败: {e}", exc_info=True)
        
        logger.info(f"[ImageProducer] 最终提取到的图片 URL: {image_urls}")
        return list(dict.fromkeys(image_urls))  # 去重

    async def _extract_image_from_node_async(self, node, image_urls: List[str]):
        """从 Node 组件中提取图片（异步版本）"""
        if hasattr(node, 'content') and node.content:
            content = node.content
            logger.info(f"[ImageProducer] Node.content 类型: {type(content).__name__}")
            if isinstance(content, list):
                logger.info(f"[ImageProducer] Node.content 长度: {len(content)}")
                for i, sub_comp in enumerate(content):
                    sub_type = type(sub_comp).__name__
                    logger.info(f"[ImageProducer]   Node.content[{i}] 类型: {sub_type}")
                    await self._extract_image_from_component_async(sub_comp, image_urls, None)
            else:
                logger.warning(f"[ImageProducer] Node.content 不是列表类型")

    async def _extract_image_from_component_async(self, comp, image_urls: List[str], event: AstrMessageEvent = None):
        """从任意消息组件中提取图片 URL（异步版本）"""
        if isinstance(comp, Comp.Image):
            self._extract_image_url_from_component(comp, image_urls)
        elif isinstance(comp, Comp.File):
            self._extract_file_url_from_component(comp, image_urls)
        elif isinstance(comp, Comp.Forward):
            logger.info(f"[ImageProducer] 在组件中发现 Forward，id={getattr(comp, 'id', 'N/A')}")
            if event:
                forward_urls = await self._fetch_forward_message(event, str(comp.id))
                image_urls.extend(forward_urls)
        elif isinstance(comp, Comp.Nodes):
            logger.info(f"[ImageProducer] 在组件中发现 Nodes")
            if hasattr(comp, 'nodes') and comp.nodes:
                for node in comp.nodes:
                    await self._extract_image_from_node_async(node, image_urls)
        elif isinstance(comp, Comp.Node):
            logger.info(f"[ImageProducer] 在组件中发现 Node")
            await self._extract_image_from_node_async(comp, image_urls)

    def _extract_image_urls_from_event(self, event: AstrMessageEvent) -> List[str]:
        """从事件中提取所有的图片 URL（同步版本，已废弃，请使用 _extract_image_urls_from_event_async）"""
        image_urls: List[str] = []
        try:
            if hasattr(event, 'get_messages'):
                messages = event.get_messages()
                logger.info(f"[ImageProducer] 消息组件总数: {len(messages)}")
                for comp in messages:
                    comp_type = type(comp).__name__
                    logger.info(f"[ImageProducer] 消息组件类型: {comp_type}")
                    
                    if isinstance(comp, Comp.Reply):
                        logger.info(f"[ImageProducer] 发现 Reply 组件，id={comp.id}, chain长度={len(comp.chain) if comp.chain else 0}")
                        if hasattr(comp, 'chain') and comp.chain:
                            for i, quote in enumerate(comp.chain):
                                quote_type = type(quote).__name__
                                logger.info(f"[ImageProducer]   Quote[{i}] 类型: {quote_type}")
                                self._extract_image_from_component(quote, image_urls)
                        else:
                            logger.warning(f"[ImageProducer] Reply 组件没有 chain 属性或 chain 为空")
                    elif isinstance(comp, Comp.Forward):
                        logger.info(f"[ImageProducer] 发现 Forward 组件，id={comp.id}")
                        logger.warning(f"[ImageProducer] Forward 组件只包含 ID，请使用异步方法获取内容")
                    elif isinstance(comp, Comp.Nodes):
                        logger.info(f"[ImageProducer] 发现 Nodes 组件")
                        if hasattr(comp, 'nodes') and comp.nodes:
                            for node in comp.nodes:
                                self._extract_image_from_node(node, image_urls)
                    elif isinstance(comp, Comp.Node):
                        logger.info(f"[ImageProducer] 发现 Node 组件")
                        self._extract_image_from_node(comp, image_urls)
                    else:
                        self._extract_image_from_component(comp, image_urls)
        except Exception as e:
            logger.warning(f"[ImageProducer] 提取图片 URL 失败: {e}", exc_info=True)
        
        logger.info(f"[ImageProducer] 最终提取到的图片 URL: {image_urls}")
        return list(dict.fromkeys(image_urls))  # 去重

    def _extract_image_from_node(self, node, image_urls: List[str]):
        """从 Node 组件中提取图片"""
        if hasattr(node, 'content') and node.content:
            content = node.content
            logger.info(f"[ImageProducer] Node.content 类型: {type(content).__name__}")
            if isinstance(content, list):
                logger.info(f"[ImageProducer] Node.content 长度: {len(content)}")
                for i, sub_comp in enumerate(content):
                    sub_type = type(sub_comp).__name__
                    logger.info(f"[ImageProducer]   Node.content[{i}] 类型: {sub_type}")
                    self._extract_image_from_component(sub_comp, image_urls)
            else:
                logger.warning(f"[ImageProducer] Node.content 不是列表类型")

    def _extract_image_from_component(self, comp, image_urls: List[str]):
        """从任意消息组件中提取图片 URL"""
        if isinstance(comp, Comp.Image):
            self._extract_image_url_from_component(comp, image_urls)
        elif isinstance(comp, Comp.File):
            self._extract_file_url_from_component(comp, image_urls)
        elif isinstance(comp, Comp.Forward):
            logger.info(f"[ImageProducer] 在 _extract_image_from_component 中发现 Forward 组件，id={getattr(comp, 'id', 'N/A')}")
            logger.info(f"[ImageProducer] Forward 组件属性: {comp.__dict__.keys()}")
            if hasattr(comp, 'nodes') and comp.nodes:
                logger.info(f"[ImageProducer] Forward.nodes 长度: {len(comp.nodes)}")
                for node in comp.nodes:
                    self._extract_image_from_node(node, image_urls)
            else:
                logger.warning(f"[ImageProducer] Forward 组件没有 nodes 属性或 nodes 为空，尝试获取其他属性")
                # 尝试其他可能的属性名
                for attr in ['nodes', 'content', 'chain', 'message']:
                    if hasattr(comp, attr):
                        val = getattr(comp, attr)
                        logger.info(f"[ImageProducer] Forward.{attr} = {val}")
        elif isinstance(comp, Comp.Nodes):
            logger.info(f"[ImageProducer] 在 _extract_image_from_component 中发现 Nodes 组件")
            if hasattr(comp, 'nodes') and comp.nodes:
                for node in comp.nodes:
                    self._extract_image_from_node(node, image_urls)
        elif isinstance(comp, Comp.Node):
            logger.info(f"[ImageProducer] 在 _extract_image_from_component 中发现 Node 组件")
            self._extract_image_from_node(comp, image_urls)

    def _extract_image_url_from_component(self, image_comp: Comp.Image, image_urls: List[str]):
        """从 Image 组件中提取 URL"""
        # 优先使用 url 属性
        if image_comp.url and image_comp.url.startswith("http"):
            image_urls.append(image_comp.url)
        # 其次使用 file 属性
        elif image_comp.file:
            if image_comp.file.startswith("http"):
                image_urls.append(image_comp.file)
            elif image_comp.file.startswith("file:///"):
                # 本地文件，尝试转换为 URL 或跳过
                logger.debug(f"[ImageProducer] 发现本地文件图片: {image_comp.file}")
            elif image_comp.file.startswith("base64://"):
                # base64 图片，直接使用
                image_urls.append(image_comp.file)

    def _extract_file_url_from_component(self, file_comp: Comp.File, image_urls: List[str]):
        """从 File 组件中提取 URL（如果是图片格式）"""
        if file_comp.url and file_comp.url.startswith("http"):
            if file_comp.url.lower().endswith(SUPPORTED_FILE_FORMATS_WITH_DOT):
                image_urls.append(file_comp.url)
        elif file_comp.file:
            if file_comp.file.startswith("http") and file_comp.file.lower().endswith(SUPPORTED_FILE_FORMATS_WITH_DOT):
                image_urls.append(file_comp.file)

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
            # 处理 base64 格式的图片
            if url.startswith("base64://"):
                b64_data = url.removeprefix("base64://")
                # 解码 base64 数据
                image_bytes = base64.b64decode(b64_data)
                result = await asyncio.to_thread(ImageProducer._handle_image, image_bytes)
                return result, True
            
            # 处理 HTTP URL
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

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def on_message(self, event: AstrMessageEvent):
        """处理 img 相关消息（仅处理非唤醒命令的前缀匹配）"""
        # 获取消息文本
        text = self._get_message_text(event)
        logger.info(f"[ImageProducer] on_message 触发，文本: {text[:50] if text else 'None'}, 前缀列表: {self.prefix_list}, 提供商: {list(self.provider_map.keys())}")
        if not text:
            return

        # 如果是唤醒命令，全部交给 command 装饰器处理，避免重复
        is_wake_command = event.is_at_or_wake_command
        if is_wake_command:
            logger.info(f"[ImageProducer] 唤醒命令，跳过（由 command 装饰器处理）")
            return

        # 仅处理非唤醒命令的前缀匹配
        matched_prefix = None
        for prefix in self.prefix_list:
            if text.startswith(prefix):
                matched_prefix = prefix
                break

        if not matched_prefix:
            return

        # 去除前缀
        process_text = text[len(matched_prefix):].lstrip()

        logger.info(f"[ImageProducer] 前缀匹配: {matched_prefix}, 处理文本: {process_text[:50]}")

        # 白名单检查
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用图像生成功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用图像生成功能")
            return

        # 收集模式处理
        if self.gather_mode_enabled:
            await self._generate_image_gather_mode(event)
        else:
            await self._generate_image_direct_mode(event, override_text=process_text)

    @filter.command("img", alias={"aimg", "生图", "ai生图"})
    async def generate_image_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用图像生成功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用图像生成功能")
            return

        if self.gather_mode_enabled:
            await self._generate_image_gather_mode(event)
        else:
            await self._generate_image_direct_mode(event)

    async def _generate_image_direct_mode(self, event: AstrMessageEvent, override_text: str = None):
        """直接模式：立即生成图片"""
        text = override_text if override_text is not None else self._get_message_text(event)
        prompt = self._extract_prompt(text)

        # 检查是否是子命令
        if prompt:
            sub_command = prompt.strip().lower()
            if sub_command in ["帮助", "help", "设置", "config", "平台", "platform"]:
                await self._handle_img_subcommand(event, sub_command)
                return

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

        image_urls = await self._extract_image_urls_from_event_async(event)
        logger.info(f"[ImageProducer] 提取到的图片 URL: {image_urls}")
        image_b64_list = []
        if image_urls:
            logger.info(f"[ImageProducer] 检测到 {len(image_urls)} 张图片，开始下载...")
            image_b64_list = await self._fetch_images(image_urls)
            if image_b64_list:
                logger.info(f"[ImageProducer] 成功下载 {len(image_b64_list)} 张图片")

        if not prompt and not image_b64_list:
            if self.allow_chinese_prompt:
                prompt = "根据参考图片生成相似风格的图像，保持相同的构图、色彩和艺术手法"
            else:
                prompt = "Generate a similar image based on the reference image, maintaining the same style, composition, and mood"
        elif prompt and image_b64_list:
            if self.allow_chinese_prompt:
                prompt = f"{prompt}，参考图片风格，保持与参考图片相同的艺术手法和构图"
            else:
                prompt = f"{prompt}, reference image style, maintain the artistic approach and composition from the reference"
        elif not prompt:
            await self._safe_reply(event, "💡 请提供图像生成提示词\n示例: /img 一只可爱的猫咪在草地上玩耍\n使用预设: /img 手办化 一只猫")
            return

        await self._generate_and_send(event, prompt, image_b64_list, use_llm_refine=False)

    @filter.command("文生图", alias={"text2img", "t2i"})
    async def text_to_image_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用此功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用此功能")
            return

        text = self._get_message_text(event)
        text = text.replace('/文生图', '').replace('/text2img', '').replace('/t2i', '').strip()

        if not text:
            await self._safe_reply(event, "💡 请提供图像描述，我将直接生成图片\n示例: /文生图 一只可爱的猫咪在草地上玩耍")
            return

        await self._generate_and_send(event, text, None, use_llm_refine=False)

    @filter.command("图生图", alias={"img2img", "i2i", "以图生图", "参考生图"})
    async def image_to_image_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用此功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用此功能")
            return

        text = self._get_message_text(event)
        text = text.replace('/图生图', '').replace('/img2img', '').replace('/i2i', '').replace('/以图生图', '').replace('/参考生图', '').strip()

        image_urls = await self._extract_image_urls_from_event_async(event)
        image_b64_list = []
        if image_urls:
            logger.info(f"[ImageProducer] 图生图检测到 {len(image_urls)} 张图片，开始下载...")
            image_b64_list = await self._fetch_images(image_urls)
            if image_b64_list:
                logger.info(f"[ImageProducer] 图生图成功下载 {len(image_b64_list)} 张图片")

        if not text and not image_b64_list:
            await self._safe_reply(event, "💡 请发送参考图片，可选附带文字提示\n示例: [图片] /图生图 改成动漫风格")
            return

        if not text:
            if self.allow_chinese_prompt:
                text = "根据参考图片生成相似风格的图像，保持相同的构图、色彩和艺术手法"
            else:
                text = "Generate a similar image based on the reference image, maintaining the same style, composition, and mood"

        await self._generate_and_send(event, text, image_b64_list, use_llm_refine=False)

    async def _handle_img_subcommand(self, event: AstrMessageEvent, sub_command: str):
        """处理 img 子命令（帮助、设置、平台等）"""
        if sub_command in ["帮助", "help"]:
            help_text = """📖 AI 图像生成器 - 使用帮助

【直接生成指令（不调用LLM）】
• /img [提示词] - 根据文字直接生成图片
• /img [图片] - 以图生图
• /img [图片] [提示词] - 参考图片+文字直接生成
• /文生图 [提示词] - 纯文字直接生成
• /图生图 [图片] [提示词] - 参考图片直接生成

【LLM优化生成】
• /生成 [描述] - 先调用LLM优化提示词，再生成图片
• /提示词 [描述] - 仅生成优化后的提示词

【预设风格】
• /img列表 - 查看所有预设
• /img查看 [名称] - 查看预设详情

【智能功能】
• 主提供商失败自动切换备用
• 支持多图参考融合

💡 提示：/img、/文生图、/图生图 直接调用图像API，不经过LLM"""
            await self._safe_reply(event, help_text)
            
        elif sub_command in ["设置", "config"]:
            settings_text = """⚙️ 当前配置信息

【主提供商】
"""
            main_provider = self.conf.get("main_provider", {})
            if main_provider:
                api_type = main_provider.get("api_type", "未配置")
                api_name = main_provider.get("api_name", "主提供商")
                model = main_provider.get("model", "未配置")
                vision_model = main_provider.get("vision_model", "未配置")
                enabled = main_provider.get("enabled", False)
                settings_text += f"""• 名称：{api_name}
• API类型：{api_type}
• 状态：{"✅ 已启用" if enabled else "❌ 已禁用"}
• 生成模型：{model}
• 视觉模型：{vision_model}
"""
            else:
                settings_text += "• 未配置\n"
            
            settings_text += "\n【备用提供商】\n"
            back_providers = ["back_provider", "back_provider2", "back_provider3", "back_provider4", "back_provider5"]
            for i, bp_key in enumerate(back_providers, 1):
                bp = self.conf.get(bp_key, {})
                if bp and bp.get("enabled", False):
                    api_type = bp.get("api_type", "未配置")
                    api_name = bp.get("api_name", f"备用{i}")
                    model = bp.get("model", "未配置")
                    settings_text += f"• {api_name}（{api_type}）- {model}\n"
            
            settings_text += f"""
【默认设置】
• 图像尺寸：{self.default_size}
• 图像质量：{self.default_quality}
• 图像风格：{self.default_style}
• 并发任务数：{self.max_concurrent_jobs}
• 超时时间：{self.timeout}秒

💡 请在插件设置页面修改配置"""
            await self._safe_reply(event, settings_text)
            
        elif sub_command in ["平台", "platform"]:
            platform_text = """🔄 提供商状态

【当前默认平台】：{}""".format(self.default_platform)

            platform_text += "\n\n【已加载的提供商】\n"
            for pname, pconfig in self.provider_configs.items():
                api_name = pconfig.get("api_name", pname)
                model = pconfig.get("model", "未配置")
                vision_model = pconfig.get("vision_model", "未配置")
                is_default = "⭐" if pname == self.default_platform else "  "
                platform_text += f"{is_default}{api_name}：{pname}\n"
                platform_text += f"   生成模型：{model}\n"
                platform_text += f"   视觉模型：{vision_model}\n"
            
            platform_text += """
【降级机制】
• 主提供商失败后，依次尝试备用提供商
• 最多重试 {} 次
• 支持 6 个提供商配置

💡 请在插件设置页面配置多个提供商""".format(self.max_retry)
            await self._safe_reply(event, platform_text)

    async def _generate_image_gather_mode(self, event: AstrMessageEvent):
        """收集模式：收集多张图片和文本后生成"""
        text = self._get_message_text(event)
        prompt = self._extract_prompt(text)

        image_urls = await self._extract_image_urls_from_event_async(event)
        image_b64_list = []
        if image_urls:
            logger.info(f"[ImageProducer] 收集模式：检测到 {len(image_urls)} 张图片...")
            image_b64_list = await self._fetch_images(image_urls)

        if prompt:
            prompt = f"{prompt}, reference image style, maintain the artistic approach and composition from the reference"

        operator_id = event.get_sender_id()
        is_cancel = False

        await self._safe_reply(event, f"""📝 绘图收集模式已启用
提示词：{prompt or "（待输入）"}
图片：{len(image_b64_list)} 张

💡 操作说明：
• 发送图片可追加参考图
• 发送文字可追加到提示词
• 发送「开始」立即生成
• 发送「取消」退出操作
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

            sub_image_urls = await self._extract_image_urls_from_event_async(sub_event)
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
                await self._safe_reply(sub_event, f"""📝 已收集：
提示词：{prompt or "（待输入）"}
图片：{len(image_b64_list)} 张

💡 继续发送内容，或发送「开始」生成""")

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

        prompt = await self._generate_prompt(text, event)
        if prompt:
            await self._safe_reply(event, f"✨ 生成的提示词:\n{prompt}")
        else:
            await self._safe_reply(event, "❌ 提示词生成失败，请稍后重试")

    async def _generate_prompt(self, user_description: str, event: AstrMessageEvent = None) -> Optional[str]:
        try:
            logger.info(f"[ImageProducer] 开始生成提示词，描述: {user_description[:50]}...")
            result = await asyncio.wait_for(
                self._generate_prompt_internal(user_description, event),
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
    
    async def _generate_prompt_internal(self, user_description: str, event: AstrMessageEvent = None) -> Optional[str]:
        """
        内部提示词生成 - 尝试调用 AstrBot 的 LLM 能力
        """
        try:
            if self.allow_chinese_prompt:
                system_content = """你是一个专业的AI图像生成提示词工程师，擅长将简单的用户描述转换为详细、高质量的图像生成提示词。

要求：
1. 语言：仅使用中文
2. 详细描述场景、主体、风格、光线、色彩等
3. 添加质量关键词：杰作、最佳质量、超细节、高品质、8k分辨率等
4. 生成的提示词长度应为100-300字
5. 直接返回提示词，不要包含任何额外解释"""
                
                user_content = f"请将以下描述转换为专业的图像生成提示词：{user_description}"
            else:
                system_content = """You are a professional AI image generation prompt engineer. Your task is to convert simple user descriptions into detailed, high-quality image generation prompts. Requirements: English only, detailed description of the scene, subject, style, lighting, colors, etc., add quality keywords like masterpiece, best quality, ultra-detailed, high quality, 8k, etc. The generated prompt should be 100-300 words long. Just return the prompt directly, do not include any additional explanations."""
                
                user_content = f"Please convert this description into a professional image prompt: {user_description}"

            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            try:
                from astrbot.core.agent.message import UserMessageSegment, TextPart
                contexts = [
                    UserMessageSegment(content=[TextPart(text=msg["content"])])
                    for msg in messages
                ]
                
                umo = None
                if event and hasattr(event, 'unified_msg_origin'):
                    umo = event.unified_msg_origin
                
                provider_id = None
                if umo:
                    try:
                        provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                    except Exception as e:
                        logger.warning(f"[ImageProducer] 获取 provider_id 失败: {e}")
                
                if not provider_id:
                    try:
                        provider_id = await self.context.get_current_chat_provider_id()
                    except Exception as e:
                        logger.warning(f"[ImageProducer] 获取默认 provider_id 失败: {e}")
                        return None
                
                logger.info(f"[ImageProducer] 使用 LLM 生成提示词，provider_id: {provider_id}")
                
                result = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    contexts=contexts
                )
                
                if result and hasattr(result, 'completion_text'):
                    content = result.completion_text
                    if content:
                        logger.info(f"[ImageProducer] 提示词生成成功，长度: {len(content)}")
                        return content.strip()
                    
            except Exception as llm_err:
                logger.error(f"[ImageProducer] LLM调用失败: {llm_err}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"[ImageProducer] 提示词生成内部异常: {e}", exc_info=True)
            return None
        return None

    async def _refine_prompt_with_llm(self, image_description: str, user_prompt: str, event: AstrMessageEvent = None) -> str:
        """
        使用 AstrBot LLM 修饰图片描述和用户描述，生成优质图像提示词
        
        Args:
            image_description: 视觉模型返回的图片描述
            user_prompt: 用户需求描述
            event: 消息事件，用于获取会话上下文
            
        Returns:
            str: 优化后的图像生成提示词（根据配置决定中文或英文）
        """
        try:
            if self.allow_chinese_prompt:
                system_content = """你是一个专业的AI图像生成提示词工程师，擅长创建详细、高质量的中文提示词。你的任务是将参考图片描述和用户需求结合成优秀的中文图像生成提示词。

核心要求：
1. 语言：仅使用中文
2. 长度：150-350字
3. 结构：主体 + 环境 + 风格 + 光线 + 色彩 + 构图 + 质量

需要包含的风格元素：
- 艺术风格（写实、动漫、油画、水彩、数字艺术等）
- 光线条件（黄金时刻、工作室灯光、戏剧性阴影等）
- 色彩搭配和氛围（暖色调、冷色调、鲜艳、柔和等）
- 构图规则（三分法、引导线、景深等）
- 相机角度和视角（特写、广角、低角度等）

质量关键词（选择合适的）：
杰作、最佳质量、超细节、高品质、8k分辨率、照片级真实感、电影级灯光、专业摄影、获奖构图

参考图片整合：
- 分析并保留参考图片的艺术风格
- 保持与参考构图的视觉一致性
- 将用户需求无缝融入参考美学
- 当提供多个参考时，融合它们的风格

输出：仅返回最终提示词，不要解释或格式化。"""
                
                user_content = f"""请结合以下内容创建专业的中文AI图像生成提示词：

参考图片描述：
{image_description}

用户需求：
{user_prompt}

生成一个详细、高质量的中文提示词，保留参考图片的艺术风格同时融入所有用户需求。仅返回提示词文本。"""
            else:
                system_content = """You are an elite AI image generation prompt engineer specializing in creating professional, detailed prompts for modern AI image models. Your task is to combine reference image descriptions and user requirements into exceptional English prompts.

CORE REQUIREMENTS:
1. Language: English ONLY
2. Length: 150-350 words
3. Structure: Subject + Environment + Style + Lighting + Color + Composition + Quality

STYLE ELEMENTS TO INCLUDE:
- Art style (realistic, anime, oil painting, watercolor, digital art, etc.)
- Lighting conditions (golden hour, studio lighting, dramatic shadows, etc.)
- Color palette and mood (warm tones, cool atmosphere, vibrant, muted, etc.)
- Composition rules (rule of thirds, leading lines, depth of field, etc.)
- Camera angle and perspective (close-up, wide shot, low angle, etc.)

QUALITY KEYWORDS (select appropriate ones):
masterpiece, best quality, ultra-detailed, high quality, 8k resolution, photorealistic, cinematic lighting, professional photography, award-winning composition

REFERENCE IMAGE INTEGRATION:
- Analyze and preserve the artistic style from reference images
- Maintain visual consistency with reference compositions
- Blend user requirements seamlessly with reference aesthetics
- When multiple references are provided, harmonize their styles

OUTPUT: Return ONLY the final prompt, no explanations or formatting."""
                
                user_content = f"""Create a professional AI image generation prompt by combining:

Reference Image Description:
{image_description}

User Requirements:
{user_prompt}

Generate a detailed, high-quality English prompt that preserves the reference image's artistic style while incorporating all user requirements. Return ONLY the prompt text."""

            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            try:
                from astrbot.core.agent.message import UserMessageSegment, TextPart
                contexts = [
                    UserMessageSegment(content=[TextPart(text=msg["content"])])
                    for msg in messages
                ]
                
                umo = None
                if event and hasattr(event, 'unified_msg_origin'):
                    umo = event.unified_msg_origin
                
                provider_id = None
                if umo:
                    try:
                        provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                    except Exception as e:
                        logger.warning(f"[ImageProducer] 获取 provider_id 失败: {e}")
                
                if not provider_id:
                    try:
                        provider_id = await self.context.get_current_chat_provider_id()
                    except Exception as e:
                        logger.warning(f"[ImageProducer] 获取默认 provider_id 失败: {e}")
                        return None
                
                logger.info(f"[ImageProducer] 使用 LLM 修饰提示词，provider_id: {provider_id}")
                
                result = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    contexts=contexts
                )
                
                if result and hasattr(result, 'completion_text'):
                    content = result.completion_text
                    if content:
                        import re
                        content = re.sub(r'^```.*?\n', '', content)
                        content = re.sub(r'\n```$', '', content)
                        logger.info(f"[ImageProducer] LLM修饰完成，提示词长度: {len(content)}")
                        return content.strip()
                    
            except Exception as llm_err:
                logger.error(f"[ImageProducer] LLM修饰失败: {llm_err}", exc_info=True)
                return None
                
        except Exception as e:
            logger.error(f"[ImageProducer] LLM修饰内部异常: {e}", exc_info=True)
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
        image_urls = await self._extract_image_urls_from_event_async(event)
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

        prompt = await self._generate_prompt(text or "基于参考图片生成图像", event)
        if not prompt:
            await self._safe_reply(event, "❌ 提示词生成失败，尝试直接生成图像...")
            prompt = text or "根据参考图片生成图像"

        await self._safe_reply(event, f"✨ 提示词生成完成\n正在生成图像...")
        await self._generate_and_send(event, prompt, image_b64_list)

    async def _generate_and_send(self, event: AstrMessageEvent, prompt: str, image_b64_list: list = None, use_llm_refine: bool = True):
        """
        生成图片并发送给用户，确保100%给用户回复
        
        发送策略：
        1. 发送简单的文本消息（正在生成）
        2. 生成完成后发送图片
        
        Args:
            use_llm_refine: 是否使用LLM修饰提示词（/文生图、/图生图设为False，/生成设为True）
        """
        # === 阶段 1: 准备 ===
        try:
            if not self.provider_map:
                await self._safe_reply(event, "❌ 未配置任何图像生成平台，请先在插件设置中配置API")
                return
            logger.info(f"[ImageProducer] 开始生成图像，提示词: {prompt[:50]}...")
            
            # 发送"正在生成"的消息
            await self._safe_reply(event, "🎨 正在生成图像...")
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

        # 智能选择平台：有图片时自动使用视觉模型
        target_platform = self.default_platform
        has_images = bool(image_b64_list and len(image_b64_list) > 0)
        final_prompt = prompt
        vision_processed = False  # 标记是否已在 main.py 完成视觉分析
        
        if has_images and self.auto_switch_mode:
            main_provider_config = self.conf.get("main_provider", {})
            api_type = main_provider_config.get("api_type", "")
            logger.info(f"[ImageProducer] 视觉模型切换检查: api_type={api_type}, provider_map={list(self.provider_map.keys())}, auto_switch_mode={self.auto_switch_mode}")
            vision_provider_name = PLATFORM_TO_VISION_PROVIDER.get(api_type)
            logger.info(f"[ImageProducer] 视觉模型映射: {api_type} -> {vision_provider_name}")
            
            # 先调用视觉模型分析图片，得到图片描述
            provider = self.provider_map.get(self.default_platform)
            if provider:
                vision_api_key = main_provider_config.get("vision_api_key", "")
                vision_api_url = main_provider_config.get("vision_api_url", "")
                if vision_api_key and vision_api_url:
                    logger.info(f"[ImageProducer] 开始视觉模型分析图片...")
                    image_description = await provider._analyze_reference_images(
                        vision_api_url, vision_api_key, image_b64_list, prompt,
                        use_chinese=self.allow_chinese_prompt
                    )
                    logger.info(f"[ImageProducer] 视觉模型分析完成: {image_description[:50]}...")
                    vision_processed = True
                    
                    # 使用 LLM 修饰图片描述和用户描述（仅当 use_llm_refine=True 时）
                    if use_llm_refine:
                        logger.info(f"[ImageProducer] 开始 LLM 修饰...")
                        refined_prompt = await self._refine_prompt_with_llm(image_description, prompt, event)
                        if refined_prompt:
                            final_prompt = refined_prompt
                            logger.info(f"[ImageProducer] LLM 修饰完成，使用新提示词: {final_prompt[:50]}...")
                        else:
                            logger.warning(f"[ImageProducer] LLM 修饰失败，使用视觉模型分析结果")
                            final_prompt = image_description
                    else:
                        # 不使用LLM修饰，将用户提示词与图片描述结合，确保多图参考生效
                        if self.allow_chinese_prompt:
                            if prompt:
                                final_prompt = f"{prompt}。请参考提供的图片的风格、构图、光线、色彩搭配和艺术手法，保持与参考图片的视觉一致性，同时融入描述中的元素。"
                            else:
                                final_prompt = f"根据参考图片生成图像。{image_description}。保持与参考图片相同的风格、构图、光线、色彩搭配和艺术手法。"
                        else:
                            if prompt:
                                final_prompt = f"{prompt}. Reference the provided image(s) for style, composition, lighting, color palette, and artistic approach. Maintain visual consistency with the reference image(s) while incorporating the described elements."
                            else:
                                final_prompt = f"Generate an image based on the reference image(s). {image_description}. Maintain the same style, composition, lighting, color palette, and artistic approach as shown in the reference image(s)."
                        logger.info(f"[ImageProducer] 不使用LLM修饰，结合用户提示词与图片描述")
                else:
                    logger.info(f"[ImageProducer] 未配置视觉模型，使用原始提示词")
            
            if vision_provider_name and vision_provider_name in self.provider_map:
                target_platform = vision_provider_name
                logger.info(f"[ImageProducer] 检测到图片，自动切换到视觉模型: {target_platform}")
            elif self.default_platform and self.default_platform in self.provider_map:
                if vision_api_key and vision_api_url:
                    logger.info(f"[ImageProducer] 检测到图片，使用默认提供商 {self.default_platform} + 视觉模型配置")
                    target_platform = self.default_platform
                else:
                    logger.info(f"[ImageProducer] 检测到图片，但未配置视觉模型，继续使用默认平台: {target_platform}")
            else:
                logger.info(f"[ImageProducer] 检测到图片，但未找到可用提供商，继续使用默认平台: {target_platform}")
        
        try:
            try:
                async with self.semaphore:
                    result = await asyncio.wait_for(
                        self.generate_image_internal(
                            platform=target_platform,
                            prompt=final_prompt,
                            size=self.default_size,
                            quality=self.default_quality,
                            style=self.default_style,
                            image_b64_list=image_b64_list,
                            auto_switch_mode=not vision_processed,
                            vision_processed=vision_processed,
                        ),
                        timeout=180
                    )
            except asyncio.TimeoutError:
                logger.error(f"[ImageProducer] 生成图像超时（3分钟）")
                result = None
            except Exception as e:
                logger.error(f"[ImageProducer] 生成过程异常: {e}", exc_info=True)
                result = None
        except Exception as e:
            logger.error(f"[ImageProducer] 阶段2总异常: {e}", exc_info=True)
        
        # === 阶段 3: 处理结果（保存到本地）===
        download_success = False  # 记录下载是否成功
        try:
            if result and result.success:
                logger.info("[ImageProducer] 图像生成成功，处理数据...")
                try:
                    if result.image_data:
                        save_path = await self._save_image_to_ai_images(result.image_data, prompt)
                        image_b64_data = base64.b64encode(result.image_data).decode('utf-8')
                        download_success = True
                    elif result.b64_json:
                        import re
                        b64_content = result.b64_json.strip()
                        # 检查是否是 Markdown 图片格式 ![image](url)
                        md_match = re.search(r'!\[.*?\]\((https?://[^\s)]+)\)', b64_content)
                        if md_match:
                            # 提取 URL 并下载
                            image_url = md_match.group(1)
                            logger.info(f"[ImageProducer] 检测到 Markdown 图片链接: {image_url}")
                            save_path = await self._download_and_save_to_ai_images(image_url, prompt)
                            if save_path:
                                try:
                                    with open(save_path, 'rb') as f:
                                        image_b64_data = base64.b64encode(f.read()).decode('utf-8')
                                    download_success = True
                                except Exception as read_err:
                                    logger.warning(f"[ImageProducer] 读取保存的图片失败: {read_err}")
                            else:
                                # 下载失败，保留 URL
                                result.image_url = image_url
                        else:
                            # 真正的 base64 数据
                            image_b64_data = b64_content
                            save_path = await self._save_image_to_ai_images(base64.b64decode(b64_content), prompt)
                            download_success = True
                    elif result.image_url:
                        logger.info(f"[ImageProducer] 下载图片: {result.image_url}")
                        save_path = await self._download_and_save_to_ai_images(result.image_url, prompt)
                        if save_path:
                            try:
                                with open(save_path, 'rb') as f:
                                    image_b64_data = base64.b64encode(f.read()).decode('utf-8')
                                download_success = True
                            except Exception as read_err:
                                logger.warning(f"[ImageProducer] 读取保存的图片失败: {read_err}")
                        # 下载失败，保留 URL 用于后续发送
                except Exception as e:
                    logger.error(f"[ImageProducer] 处理图片数据异常: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[ImageProducer] 阶段3总异常: {e}", exc_info=True)
        
        # === 阶段 4: 发送给用户 ===
        try:
            if result and result.success:
                image_sent = False
                
                # 如果下载成功，尝试发送图片
                if download_success:
                    # 1. 优先使用本地文件发送
                    if save_path and os.path.exists(save_path):
                        try:
                            msg_chain: list[BaseMessageComponent] = [
                                Comp.Reply(id=event.message_obj.message_id) if hasattr(event.message_obj, 'message_id') else None,
                                Comp.Image.fromFileSystem(save_path)
                            ]
                            msg_chain = [c for c in msg_chain if c is not None]
                            await event.send(MessageChain(chain=msg_chain))
                            logger.info(f"[ImageProducer] 图片发送成功 (路径: {save_path})")
                            image_sent = True
                        except Exception as pic_err:
                            logger.warning(f"[ImageProducer] 发送本地图片失败: {pic_err}")
                    
                    # 2. 本地发送失败，尝试 base64
                    if not image_sent and image_b64_data:
                        try:
                            msg_chain = [
                                Comp.Reply(id=event.message_obj.message_id) if hasattr(event.message_obj, 'message_id') else None,
                                Comp.Image.fromBase64(image_b64_data)
                            ]
                            msg_chain = [c for c in msg_chain if c is not None]
                            await event.send(MessageChain(chain=msg_chain))
                            logger.info("[ImageProducer] 图片发送成功 (base64)")
                            image_sent = True
                        except Exception as b64_err:
                            logger.warning(f"[ImageProducer] 发送 base64 图片失败: {b64_err}")
                    
                    # 3. base64 也失败，使用 URL
                    if not image_sent and result.image_url:
                        await self._safe_reply(event, f"🎨 图片链接:\n{result.image_url}")
                        image_sent = True
                    
                    if not image_sent:
                        await self._safe_reply(event, "❌ 图片发送失败，请查看日志")
                else:
                    # 下载失败，直接发送 URL
                    if result.image_url:
                        await self._safe_reply(event, f"🎨 图片链接:\n{result.image_url}")
                        image_sent = True
                    else:
                        await self._safe_reply(event, "❌ 图片下载失败，请查看日志")
                    
            elif result and not result.success:
                await self._safe_reply(event, f"❌ 图像生成失败: {result.error}")
            else:
                await self._safe_reply(event, "⏰ 图像生成超时，请稍后重试")
        
        except Exception as send_err:
            logger.error(f"[ImageProducer] 阶段4消息发送异常: {send_err}", exc_info=True)
        
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
        image_b64_list: list = None,
        auto_switch_mode: bool = True,
        vision_processed: bool = False
    ) -> ImageResult:
        provider = self.provider_map.get(platform)
        if not provider:
            return ImageResult(success=False, error=f"未找到平台: {platform}")

        last_error = None

        fallback_providers = []
        for pname, p in self.provider_map.items():
            if pname != platform:
                fallback_providers.append((pname, p))

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
                        auto_switch_mode=auto_switch_mode,
                        vision_processed=vision_processed,
                    )
                    if result.success:
                        if provider_name != platform:
                            logger.info(f"[ImageProducer] 主Provider失败，备用Provider {provider_name} 成功生成图像")
                        return result

                    last_error = result.error
                    logger.warning(f"[ImageProducer] Provider {provider_name} 生成失败: {last_error}")

                    if self._should_not_retry(last_error):
                        logger.info(f"[ImageProducer] 检测到不可重试错误，停止重试: {last_error}")
                        return result

                    if attempt < self.max_retry - 1:
                        logger.info(f"[ImageProducer] Provider {provider_name} 第 {attempt + 1} 次重试")
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[ImageProducer] Provider {provider_name} 生成异常: {e}", exc_info=True)

        return ImageResult(success=False, error=last_error or "所有Provider生成图像失败")

    def _should_not_retry(self, error: str) -> bool:
        """判断是否不应该重试
        
        以下情况不应重试：
        - 模型拒绝生成（内容策略、NSFW过滤等）
        - API Key无效
        - 模型不存在
        - 参数错误
        """
        if not error:
            return False
        
        error_lower = error.lower()
        
        no_retry_keywords = [
            "未返回图片",
            "返回格式异常",
            "api key",
            "密钥",
            "未配置",
            "invalid api key",
            "authentication",
            "unauthorized",
            "model_not_found",
            "模型不存在",
            "invalid parameter",
            "参数错误",
            "content policy",
            "内容策略",
            "nsfw",
            "safety",
            "安全",
            "blocked",
            "拒绝",
            "terminated",
            "终止",
            "rate limit",
            "配额",
            "quota",
        ]
        
        return any(keyword in error_lower for keyword in no_retry_keywords)

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