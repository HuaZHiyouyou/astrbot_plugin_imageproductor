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
)
from .core.llm_tools import ImageProducerPromptTool, ImageProducerGenerateTool


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

PROVIDER_LIST = ["openai", "gemini", "grok", "seed", "zhipu", "qianwen", "baidu", "hunyuan", "stable_diffusion"]

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
}


class ImageProducer(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config

        self.provider_configs: Dict[str, Dict[str, Any]] = {}

        self.prefix_enabled: bool = self.conf.get("prefix_enabled", False)
        prefix_list_str: str = self.conf.get("prefix_list", "")
        self.prefix_list: list = [p.strip() for p in prefix_list_str.split(",") if p.strip()] if prefix_list_str else []
        self.coexist_enabled: bool = self.conf.get("prefix_coexist", False)

        self.group_whitelist_enabled: bool = self.conf.get("whitelist_enabled", False)
        group_whitelist_str: str = self.conf.get("group_whitelist", "")
        self.group_whitelist: list = [g.strip() for g in group_whitelist_str.split(",") if g.strip()] if group_whitelist_str else []
        self.user_whitelist_enabled: bool = self.conf.get("user_whitelist_enabled", False)
        user_whitelist_str: str = self.conf.get("user_whitelist", "")
        self.user_whitelist: list = [u.strip() for u in user_whitelist_str.split(",") if u.strip()] if user_whitelist_str else []

        self.provider_map: Dict[str, BaseProvider] = {}

        data_dir = StarTools.get_data_dir("astrbot_plugin_imageproductor")
        self.ai_images_dir = data_dir / "ai_images"
        self.save_dir = data_dir / "save_images"
        os.makedirs(self.ai_images_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.session: Optional[aiohttp.ClientSession] = None

        self.default_platform = self.conf.get("default_platform", "openai")
        self.default_size = self.conf.get("default_size", "1024x1024")
        self.default_quality = self.conf.get("default_quality", "standard")
        self.default_style = self.conf.get("default_style", "vivid")
        self.max_concurrent_jobs = self.conf.get("max_concurrent_jobs", 5)
        self.enable_nsfw_filter = self.conf.get("enable_nsfw_filter", True)
        self.auto_save_images = self.conf.get("auto_save_images", True)
        self.save_images = self.conf.get("save_images", True)

        self.semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        # 正在运行的任务
        self.running_tasks: Dict[str, asyncio.Task] = {}

    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        )
        self.init_providers()

        # 检查配置是否启用函数调用工具
        if self.conf.get("llm_tool_enabled", False):
            # 注册两个工具
            self.context.add_llm_tools(ImageProducerGenerateTool(plugin=self))
            logger.info("[ImageProducer] 已注册 LLM 工具: img_producer_generate")
            
            self.context.add_llm_tools(ImageProducerPromptTool(plugin=self))
            logger.info("[ImageProducer] 已注册 LLM 工具: img_producer_prompt")

    def init_providers(self):
        for provider_name in PROVIDER_LIST:
            enabled_key = f"{provider_name}_enabled"
            if not self.conf.get(enabled_key, False):
                continue

            provider_config = {
                "enabled": True,
                "model": self.conf.get(f"{provider_name}_model", ""),
                "main_api_key": self.conf.get(f"{provider_name}_main_api_key", ""),
                "main_api_url": self.conf.get(f"{provider_name}_main_api_url", ""),
                "backup_api_key": self.conf.get(f"{provider_name}_backup_api_key", ""),
                "backup_api_url": self.conf.get(f"{provider_name}_backup_api_url", ""),
            }

            if not provider_config["main_api_key"] and not provider_config["backup_api_key"]:
                continue

            provider_class = PROVIDER_CLASS_MAP.get(provider_name)
            if provider_class and self.session:
                self.provider_map[provider_name] = provider_class(provider_config, self.session)
                self.provider_configs[provider_name] = provider_config
                logger.info(f"[ImageGen] 已加载 provider: {provider_name}")

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
        return text.strip()

    def _extract_image_urls_from_event(self, event: AstrMessageEvent) -> List[str]:
        """从事件中提取所有的图片 URL，包括引用回复中的图片"""
        image_urls: List[str] = []
        try:
            if hasattr(event, 'get_messages'):
                for comp in event.get_messages():
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
            logger.info(f"[ImageGen] 消息发送成功 (方法: event.send)")
            return True
        except Exception as e:
            last_error = e
            logger.warning(f"[ImageGen] event.send 失败: {e}")
        
        # 方法2: event.reply
        try:
            tried_methods.append("event.reply")
            await event.reply(MessageChain().message(message))
            logger.info(f"[ImageGen] 消息发送成功 (方法: event.reply)")
            return True
        except Exception as e:
            last_error = e
            logger.warning(f"[ImageGen] event.reply 失败: {e}")
        
        # 方法3: 直接通过 message_obj 发送
        try:
            tried_methods.append("message_obj")
            if hasattr(event, 'message_obj') and hasattr(event.message_obj, 'reply'):
                await event.message_obj.reply(message)
                logger.info(f"[ImageGen] 消息发送成功 (方法: message_obj.reply)")
                return True
        except Exception as e:
            last_error = e
            logger.warning(f"[ImageGen] message_obj.reply 失败: {e}")
        
        # 所有方法都失败了
        logger.error(f"[ImageGen] 所有消息发送方法都失败了! 尝试过: {tried_methods}, 最后错误: {last_error}")
        return False

    @filter.command("img", alias={"aimg", "生图", "ai生图"})
    async def generate_image_command(self, event: AstrMessageEvent):
        if not self.is_group_allowed(event):
            await self._safe_reply(event, "❌ 当前群组不在白名单中，无法使用图像生成功能")
            return

        if not self.is_user_allowed(event):
            await self._safe_reply(event, "❌ 当前用户不在白名单中，无法使用图像生成功能")
            return

        text = self._get_message_text(event)
        prompt = self._extract_prompt(text)

        # 从事件中提取图片 URL 并下载
        image_urls = self._extract_image_urls_from_event(event)
        image_b64_list = []
        if image_urls:
            logger.info(f"[ImageProducer] 检测到 {len(image_urls)} 张图片，开始下载...")
            image_b64_list = await self._fetch_images(image_urls)
            if image_b64_list:
                logger.info(f"[ImageProducer] 成功下载 {len(image_b64_list)} 张图片")

        if not prompt and not image_b64_list:
            # 如果没有文字提示词但有图片，给出默认提示词
            prompt = "根据这张图片生成类似的图像"

        if not prompt:
            await self._safe_reply(event, "💡 请提供图像生成提示词\n示例: /img 一只可爱的猫咪在草地上玩耍")
            return

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
        
        # === 阶段 2: 生成图片 ===
        result = None
        save_path = None
        image_b64_data = None
        final_message = "✅ 图像生成完成！"
        
        try:
            try:
                async with self.semaphore:
                    result = await asyncio.wait_for(
                        self.generate_image_internal(
                            platform=self.default_platform,
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

        try:
            result = await provider.generate_image(
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                model=model,
                image_b64_list=image_b64_list,
            )
            return result
        except Exception as e:
            logger.error(f"[ImageProducer] 生成图像异常: {e}", exc_info=True)
            return ImageResult(success=False, error=str(e))

    async def _save_image(self, image_data: bytes, prompt: str) -> Optional[str]:
        if not self.save_images:
            return None

        try:
            import re
            safe_prompt = re.sub(r'[\\/*?:"<>|]', '_', prompt[:30])
            timestamp = asyncio.get_event_loop().time()
            filename = f"{self.default_platform}_{safe_prompt}_{int(timestamp)}.png"
            save_path = self.save_dir / filename

            with open(save_path, 'wb') as f:
                f.write(image_data)

            return str(save_path)
        except Exception as e:
            logger.error(f"[ImageProducer] 保存图像失败: {e}", exc_info=True)
            return None

    async def _save_image_to_ai_images(self, image_data: bytes, prompt: str) -> Optional[str]:
        try:
            import re
            from datetime import datetime
            safe_prompt = re.sub(r'[\\/*?:"<>|]', '_', prompt[:30])
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d%H%M%S")
            filename = f"img_{timestamp}_{safe_prompt}.png"
            save_path = self.ai_images_dir / filename

            with open(save_path, 'wb') as f:
                f.write(image_data)

            logger.info(f"[ImageProducer] 图像已保存到: {save_path}")
            return str(save_path)
        except Exception as e:
            logger.error(f"[ImageProducer] 保存图像到ai_images失败: {e}", exc_info=True)
            return None

    async def _download_and_save(self, url: str, prompt: str) -> Optional[str]:
        if not self.save_images:
            return None

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return await self._save_image(image_data, prompt)
        except Exception as e:
            logger.error(f"[ImageProducer] 下载图像失败: {e}", exc_info=True)
        return None

    async def _download_and_save_to_ai_images(self, url: str, prompt: str) -> Optional[str]:
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return await self._save_image_to_ai_images(image_data, prompt)
        except Exception as e:
            logger.error(f"[ImageProducer] 下载并保存图像到ai_images失败: {e}", exc_info=True)
        return None

    @filter.command("img帮助", alias={"aimg帮助", "生图帮助", "ai生图帮助"})
    async def help_command(self, event: AstrMessageEvent):
        help_text = """
📷 图像生成插件帮助

基础命令:
- /img <提示词> - 生成图像
- /img帮助 - 显示此帮助信息

🎨 高级命令:
- /提示词 <描述> - AI生成专业提示词
- /生成 <描述> - AI生成提示词并创作图像

支持的平台:
- OpenAI DALL-E
- Google Gemini Imagen
- xAI Grok
- 字节跳动 Seed
- 智谱 AI
- 阿里云 千问
- 百度 文心一言
- 腾讯 混元
- Stable Diffusion

示例:
- /img 一只可爱的猫咪在草地上玩耍
- /提示词 未来城市夜景
- /生成 赛博朋克风格的城市
"""
        await self._safe_reply(event, help_text.strip())

    @filter.command("img平台", alias={"aimg平台", "生图平台", "ai生图平台"})
    async def platform_command(self, event: AstrMessageEvent):
        platforms_info = "📷 支持的AI图像生成平台:\n\n"
        for name in PROVIDER_LIST:
            config = self.provider_configs.get(name)
            if config:
                platforms_info += f"✅ {name.upper()}: 已配置\n"
            else:
                platforms_info += f"❌ {name.upper()}: 未配置\n"
        platforms_info += f"\n默认平台: {self.default_platform}"
        await self._safe_reply(event, platforms_info)

    @filter.command("img设置", alias={"aimg设置", "生图设置", "ai生图设置"})
    async def settings_command(self, event: AstrMessageEvent):
        settings_text = f"""⚙️ 图像生成插件当前设置:

📌 默认配置:
- 平台: {self.default_platform}
- 尺寸: {self.default_size}
- 质量: {self.default_quality}
- 风格: {self.default_style}

📌 功能设置:
- 最大并发: {self.max_concurrent_jobs}
- NSFW过滤: {'开启' if self.enable_nsfw_filter else '关闭'}
- 自动保存: {'开启' if self.auto_save_images else '关闭'}
- LLM工具: {'开启' if self.conf.get('llm_tool_enabled', False) else '关闭'}

📌 白名单:
- 群组白名单: {'开启' if self.group_whitelist_enabled else '关闭'}
- 用户白名单: {'开启' if self.user_whitelist_enabled else '关闭'}

💡 使用 /img帮助 查看更多命令"""
        await self._safe_reply(event, settings_text)

    async def shutdown(self):
        if self.session:
            await self.session.close()
            logger.info("[ImageGen] 已关闭 HTTP 会话")
