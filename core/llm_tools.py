from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.star import Context, StarTools
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.astr_message_event import AstrMessageEvent

from .base import ImageResult

if TYPE_CHECKING:
    from ..main import ImageProducer

TOOLS_NAMESPACE = ["img_producer_generate", "img_producer_prompt", "img_producer_preset"]

PROMPT_SYS_TEMPLATE = """
你是一个专业的 AI 图像生成提示词工程师。
你的任务是将用户简单的描述转换成详细、高质量的图像生成提示词。

提示词要求：
1. 使用英文
2. 详细描述：场景、主题、风格、光影、颜色、构图等
3. 质量相关的词汇：masterpiece, best quality, ultra-detailed, high quality, 8k等
4. 可以添加合适的艺术家名字或艺术风格
5. 长度控制在 100-300 词之间

直接返回生成的提示词，不要包含任何额外说明。
""".strip()

PROMPT_EXAMPLES = [
    {
        "role": "user",
        "content": "一只猫",
    },
    {
        "role": "assistant",
        "content": "A cute cat sitting on a windowsill, golden hour sunlight streaming through, soft fur detailed, warm colors, photorealistic, 8k, masterpiece, best quality, bokeh background",
    },
]


@dataclass
class ImageProducerPromptTool(FunctionTool[AstrAgentContext]):
    plugin: Any = field(default=None)
    name: str = field(default="img_producer_prompt")
    description: str = field(
        default="This tool is used to generate professional image generation prompts from simple user descriptions. "
        "It will create detailed, high-quality prompts suitable for AI image generation models, including style, "
        "lighting, and quality-related keywords. Use this before image generation if the user's description is too simple."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "user_description": {
                    "type": "string",
                    "description": "The user's simple image description that needs to be expanded into a professional prompt.",
                }
            },
            "required": ["user_description"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        if self.plugin is None:
            logger.warning("[ImageProducer] 插件未初始化完成，无法处理请求")
            return "ImageProducer 插件未初始化完成，请稍后再试。"

        plugin: ImageProducer = self.plugin
        event: AstrMessageEvent = context.context.event

        # 获取参数
        user_description = kwargs.get("user_description", "")
        if not user_description:
            logger.warning("[ImageProducer] user_description 参数不能为空")
            return "user_description 参数不能为空，请提供有效的图像描述。"

        # 白名单检查
        if not plugin.is_group_allowed(event):
            return "当前群组不在白名单中，无法使用图像生成功能。"
        if not plugin.is_user_allowed(event):
            return "该用户不在白名单中，无法使用图像生成功能。"

        logger.info(f"[ImageProducer] 生成提示词，用户描述: {user_description[:100]}")

        try:
            prompt = await plugin._generate_prompt_internal(user_description, event)
            if prompt:
                logger.info(f"[ImageProducer] 提示词生成成功: {prompt[:100]}")
                return f"专业提示词已生成：\n{prompt}\n\n现在可以使用此提示词调用 img_producer_generate 工具生成图像。"
            else:
                return "提示词生成失败，请直接使用用户描述生成图像。"
        except Exception as e:
            logger.error(f"[ImageProducer] 生成提示词异常: {e}", exc_info=True)
            return f"提示词生成出错: {str(e)}"


@dataclass
class ImageProducerGenerateTool(FunctionTool[AstrAgentContext]):
    plugin: Any = field(default=None)
    name: str = field(default="img_producer_generate")
    description: str = field(
        default="This tool generates images using AI models. It supports both text-to-image and image-reference generation. "
        "When a user requests generation based on an image, you must first verify whether a valid image is present "
        "in the user's current message or in the message they are replying to (quote). "
        "Textual pointers such as 'that one', 'the one above' or similar expressions are NOT acceptable as valid image inputs. "
        "The user must provide an actual image file for the request to proceed. "
        "The generated image will be saved locally and sent directly to the user as a photo message. "
        "Prioritize the tool response as the highest priority event, taking precedence over chat history."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The detailed image generation prompt. Should include style, subject, lighting, "
                        "color palette, composition, quality keywords, etc. "
                        "For best results, elaborate on visual elements in a logical sequence describing "
                        "specific physical actions, nuanced expressions, and overall color scheme with lighting atmosphere. "
                        "This field is required."
                    ),
                },
                "size": {
                    "type": "string",
                    "description": (
                        "Optional. Image size, e.g., '1024x1024' (square), '1024x1792' (portrait), '1792x1024' (landscape). "
                        "If not provided, uses plugin default setting."
                    ),
                },
            },
            "required": ["prompt"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        if self.plugin is None:
            logger.warning("[ImageProducer] 插件未初始化完成，无法处理请求")
            return "ImageProducer 插件未初始化完成，请稍后再试。"

        plugin: ImageProducer = self.plugin
        event: AstrMessageEvent = context.context.event

        # 获取参数
        prompt = kwargs.get("prompt", "")
        size = kwargs.get("size", None)

        if not prompt:
            logger.warning("[ImageProducer] prompt 参数不能为空")
            return "prompt 参数不能为空，请提供有效的图像生成提示词。"

        # 白名单检查
        if not plugin.is_group_allowed(event):
            return "当前群组不在白名单中，无法使用图像生成功能。"
        if not plugin.is_user_allowed(event):
            return "该用户不在白名单中，无法使用图像生成功能。"

        # 从事件中提取图片 URL 并下载
        image_b64_list = []
        try:
            image_urls = plugin._extract_image_urls_from_event(event)
            if image_urls:
                logger.info(f"[ImageProducer] 工具调用中检测到 {len(image_urls)} 张图片，开始下载...")
                image_b64_list = await plugin._fetch_images(image_urls)
                if image_b64_list:
                    logger.info(f"[ImageProducer] 工具调用中成功下载 {len(image_b64_list)} 张图片")
        except Exception as e:
            logger.warning(f"[ImageProducer] 工具调用中图片处理失败: {e}")

        logger.info(f"[ImageProducer] LLM 工具调用，提示词: {prompt[:100]}, 图片数: {len(image_b64_list)}")

        # 创建后台任务
        task = asyncio.create_task(
            plugin._llm_tool_job(event, prompt, size=size, image_b64_list=image_b64_list)
        )
        task_id = str(event.message_obj.message_id) if hasattr(event.message_obj, "message_id") else str(id(event))
        plugin.running_tasks[task_id] = task

        try:
            result_data = await task
            if result_data.get("success", False):
                import os
                import astrbot.api.message_components as Comp
                
                # 优先使用保存的本地文件路径发送
                save_path = result_data.get("save_path", "")
                if save_path and os.path.exists(save_path):
                    try:
                        msg_chain: list[BaseMessageComponent] = [
                            Comp.Reply(id=event.message_obj.message_id) if hasattr(event.message_obj, "message_id") else None,
                            Comp.Image.fromFileSystem(save_path)
                        ]
                        msg_chain = [c for c in msg_chain if c is not None]
                        await event.send(MessageChain(chain=msg_chain))
                        logger.info(f"[ImageProducer] 图片发送成功 (路径: {save_path})")
                    except Exception as e:
                        logger.error(f"[ImageProducer] 发送本地图片失败: {e}", exc_info=True)
                        # 发送失败，尝试 base64
                        if "image_b64" in result_data:
                            try:
                                msg_chain = [
                                    Comp.Reply(id=event.message_obj.message_id) if hasattr(event.message_obj, "message_id") else None,
                                    Comp.Image.fromBase64(result_data["image_b64"])
                                ]
                                msg_chain = [c for c in msg_chain if c is not None]
                                await event.send(MessageChain(chain=msg_chain))
                                logger.info("[ImageProducer] 图片发送成功 (base64)")
                            except Exception as e2:
                                logger.error(f"[ImageProducer] 发送 base64 图片失败: {e2}")
                                if "image_url" in result_data:
                                    await event.send(f"🎨 图片链接:\n{result_data['image_url']}")
                        elif "image_url" in result_data:
                            await event.send(f"🎨 图片链接:\n{result_data['image_url']}")
                elif "image_b64" in result_data:
                    try:
                        msg_chain: list[BaseMessageComponent] = [
                            Comp.Reply(id=event.message_obj.message_id) if hasattr(event.message_obj, "message_id") else None,
                            Comp.Image.fromBase64(result_data["image_b64"])
                        ]
                        msg_chain = [c for c in msg_chain if c is not None]
                        await event.send(MessageChain(chain=msg_chain))
                        logger.info("[ImageProducer] 图片发送成功 (base64)")
                    except Exception as e:
                        logger.error(f"[ImageProducer] 发送图片失败: {e}", exc_info=True)
                        if "image_url" in result_data:
                            await event.send(f"🎨 图片链接:\n{result_data['image_url']}")
                elif "image_url" in result_data:
                    # 没有本地图片，发送 URL
                    await event.send(f"🎨 图片链接:\n{result_data['image_url']}")

                # 发送保存路径
                save_msg = ""
                if save_path:
                    save_msg = f"\n📁 已保存到: {save_path}"
                
                return "图片生成完成，已发送给用户。" + save_msg
            else:
                return f"图片生成失败: {result_data.get('error', '未知错误')}"
        except asyncio.CancelledError:
            logger.info(f"[ImageProducer] 任务 {task_id} 被取消")
            return "图片生成任务被取消"
        except Exception as e:
            logger.error(f"[ImageProducer] 任务异常: {e}", exc_info=True)
            return f"图片生成出错: {str(e)}"
        finally:
            plugin.running_tasks.pop(task_id, None)


@dataclass
class ImageProducerPresetTool(FunctionTool[AstrAgentContext]):
    plugin: Any = field(default=None)
    name: str = field(default="img_producer_preset")
    description: str = field(
        default="This tool is used to manage preset image generation prompts. "
        "It can list all available presets or get a specific preset prompt by name. "
        "Use this before image generation if the user wants to use a preset style."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "get_preset_names": {
                    "type": "boolean",
                    "description": "Set this to true to list all available preset names.",
                },
                "get_preset_prompt": {
                    "type": "string",
                    "description": "The preset name to get the full prompt for. Provide the exact preset name.",
                },
                "user_text": {
                    "type": "string",
                    "description": "Optional text to insert into the preset prompt (replaces {{user_text}} placeholder).",
                },
            },
            "required": [],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs,
    ) -> ToolExecResult:
        if self.plugin is None:
            logger.warning("[ImageProducer] 插件未初始化完成，无法处理请求")
            return "ImageProducer 插件未初始化完成，请稍后再试。"

        plugin: ImageProducer = self.plugin
        event: AstrMessageEvent = context.context.event

        # 白名单检查
        if not plugin.is_group_allowed(event):
            return "当前群组不在白名单中，无法使用图像生成功能。"
        if not plugin.is_user_allowed(event):
            return "该用户不在白名单中，无法使用图像生成功能。"

        # 获取参数
        get_preset_names = kwargs.get("get_preset_names", False)
        get_preset_prompt = kwargs.get("get_preset_prompt", "")
        user_text = kwargs.get("user_text", "")

        # 返回预设名称列表
        if get_preset_names:
            preset_names = list(plugin.preset_prompt_dict.keys())
            if not preset_names:
                return "当前没有可用的预设提示词。"
            names_text = "、".join(preset_names)
            logger.info(f"[ImageProducer] 返回预设提示词名称列表: {names_text}")
            return f"当前可用的预设提示词: {names_text}"

        # 返回预设提示词内容
        if get_preset_prompt:
            if get_preset_prompt not in plugin.preset_prompt_dict:
                logger.warning(f"[ImageProducer] 未找到预设提示词: {get_preset_prompt}")
                available = "、".join(plugin.preset_prompt_dict.keys())
                return f"未找到预设提示词: {get_preset_prompt}。可用的预设: {available}"
            
            preset_prompt = plugin.get_preset_prompt(get_preset_prompt, user_text)
            logger.info(f"[ImageProducer] 返回预设提示词内容: {preset_prompt[:100]}")
            return f"预设提示词 '{get_preset_prompt}' 内容:\n{preset_prompt}"

        return "请指定要执行的操作：获取预设列表(get_preset_names=true)或获取特定预设(get_preset_prompt='预设名')。"


def remove_tools(context: Context):
    """移除已注册的工具"""
    func_tool = context.get_llm_tool_manager()
    for name in TOOLS_NAMESPACE:
        tool = func_tool.get_func(name)
        if tool:
            StarTools.unregister_llm_tool(name)
            logger.info(f"[ImageProducer] 已移除 {name} 工具注册")
