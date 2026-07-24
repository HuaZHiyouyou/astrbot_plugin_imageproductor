# AstrBot AI图像生成插件 v2.4.5

[![AstrBot](https://img.shields.io/badge/AstrBot-4.16+-blue.svg)](https://github.com/AstrBotDevs/AstrBot)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

这是一个功能强大的多平台AI图像生成插件，专为AstrBot机器人框架设计。支持OpenAI DALL-E、Google Gemini、xAI Grok、Seed/Seedream和Stable Diffusion等多个AI图像生成平台。

## 功能特性

### 多平台支持
- **OpenAI DALL-E** - 高质量的商业级图像生成
- **Google Gemini Imagen** - Google先进的AI图像生成
- **xAI Grok** - 支持Grok-Imagine模型
- **Seed/Seedream** - 字节跳动的AI图像生成平台
- **Stable Diffusion** - 本地部署的开源图像生成

### 核心功能
- 多平台支持 - 5个主流AI图像生成平台
- 主备切换 - 每个平台支持主用/备用双配置
- 命令别名 - 短命令快速访问（/img、/生图）
- 白名单 - 支持群组和用户白名单控制
- 并发控制 - 智能任务队列，支持最多20个并发任务
- LLM工具调用 - 可作为工具被大语言模型调用

### 设计亮点
- 模块化架构 - Provider模式，易于扩展新平台
- 前缀匹配 - 支持自定义前缀
- 分层配置 - common/prefix/whitelist/provider分层管理
- 管理员权限 - 敏感操作需管理员权限

## 安装要求

### 系统要求
- Python 3.8+
- AstrBot 4.16+
- 网络连接（用于API调用）

### 依赖安装
```bash
pip install aiohttp Pillow
```

## 快速开始

### 1. 安装插件
将插件目录放置到 `AstrBot/data/plugins/astrbot_plugin_imageproductor/`
重启AstrBot
插件将自动加载

### 2. 配置API密钥
在插件配置页面配置各平台的API密钥
启用需要使用的平台

### 3. 开始使用
```bash
# 基本图像生成
/img 一只可爱的猫在花园里玩耍

# 使用别名
/aimg 一只可爱的猫在花园里玩耍
/生图 一只可爱的猫在花园里玩耍

# 查看帮助
/img帮助

# 查看配置
/img设置
```

## 命令参考

### 生图指令
| 命令 | 别名 | 说明 | 示例 |
|------|------|------|------|
| `/文生图 [提示词]` | `/txt2img`, `/文字生图` | 纯文字生成图片（忽略参考图片） | `/文生图 一只可爱的猫咪` |
| `/图生图 [图片] [提示词]` | `/img2img`, `/以图生图`, `/参考生图` | 参考图片生成（可选文字提示） | `[图片] /图生图 改成动漫风格` |

### 智能指令
| 命令 | 别名 | 说明 | 示例 |
|------|------|------|------|
| `/img [提示词]` | `/aimg`, `/生图`, `/ai生图` | 文字生图 | `/img 一只猫` |
| `/img [图片]` | - | 以图生图 | `[图片] /img` |
| `/img [图片] [提示词]` | - | 图+文结合 | `[图片] /img 改成水彩风格` |
| `/img [预设名] [内容]` | - | 使用预设风格 | `/img 手办化 一只猫` |

### 提示词工具
| 命令 | 别名 | 说明 | 示例 |
|------|------|------|------|
| `/提示词 [描述]` | `/prompt`, `/生提示词` | AI 优化提示词 | `/提示词 一只猫` |
| `/生成 [描述]` | `/gen`, `/做图` | 两阶段生成（优化提示词→生成图片） | `/生成 一只猫` 或 `[图片] /生成 改成动漫风` |

### 预设管理
| 命令 | 别名 | 说明 |
|------|------|------|
| `/img预设` | `/图片预设`, `/预设`, `/img列表`, `/img查看` | 不带参数显示列表，带参数显示详情 |

### 任务管理
| 命令 | 别名 | 说明 |
|------|------|------|
| `/生图列表` | `/ai生图列表`, `/任务队列` | 查看当前任务队列 |

### 管理指令
| 命令 | 别名 | 说明 |
|------|------|------|
| `/img帮助` | `/imgh`, `/aimg帮助` | 显示帮助信息 |
| `/img设置` | `/imgconfig`, `/aimg设置` | 查看当前配置 |
| `/img平台` | `/imgp`, `/aimg平台` | 查看平台状态 |

### 参数说明
- **平台**：openai, gemini, grok, seed, stable_diffusion
- **尺寸**：512x512, 1024x1024, 1792x1024, 1024x1792
- **质量**：standard, hd, ultra, premium
- **风格**：vivid, natural, realistic, anime, illustration

## 配置说明

### 通用配置 (common_config)
- default_platform: 默认平台
- default_size: 默认尺寸
- default_quality: 默认质量
- default_style: 默认风格
- max_concurrent_jobs: 最大并发数
- enable_nsfw_filter: NSFW过滤
- auto_save_images: 自动保存
- llm_tool_enabled: 启用LLM工具调用

### 前缀配置 (prefix_config)
- prefix_list: 前缀列表
- coexist_enabled: 混合模式

### 白名单配置 (whitelist_config)
- enabled: 启用群组白名单
- whitelist: 群组ID列表
- user_enabled: 启用用户白名单
- user_whitelist: 用户ID列表

## 版本历史

### v2.4.5
- 新增任务独立日志系统：
  - 每个任务生成独立的日志文件 `task_{id}.log`
  - 新增 `TaskLogger` 类，支持分阶段记录耗时和结果
  - 新增 `TaskLoggerManager` 管理所有任务日志器
  - 日志目录：`task_logs/`
- 新增 Base64 解码工具：
  - 新增 `Base64Decoder` 类，统一处理 base64 解码和图片保存
  - 支持多种格式：data URI、纯 base64、HTTP URL 下载
  - 自动检测图片格式（PNG/JPEG/GIF/WebP/BMP）
  - 并发安全：使用 asyncio.Lock 保证文件写入安全
  - 图片统一保存到 `ai_images/` 目录
- 优化命令注册：
  - 新增 `/img帮助`、`/img设置`、`/img平台`、`/生图列表` 命令
  - 合并 `/img列表` 和 `/img查看` 为 `/img预设` 命令
- 修复 Chat API 返回纯 base64 数据时被错误识别为 URL 的问题：
  - 新增多重特征检测：字符集比例、长度验证、解码测试、Magic Number 检测
  - 长度 > 10000 的内容直接尝试作为 base64
  - 移除换行符和空格后再验证
  - 已修复所有提供商（OpenAI、智谱、百度、混元、Grok、千问等）
- 新增多层殿后方案：
  - 方案5 下载失败后发送 URL 殿后消息
  - 方案6 尝试将长文本作为 base64 解码
  - 最终殿后记录并提示可用的 URL 或 base64 数据
- 优化用户体验：
  - 精简消息流程：生成 → 图片发送（移除冗余提示）
  - 面板日志只显示缩略（50字符预览）
  - 本地文件保存全量 base64 数据
- 修复视觉模型分析失败后流程中断的问题：
  - 添加异常捕获，视觉模型失败后继续使用原始提示词生成
- 修复 `Comp.Text` 组件不存在的问题：
  - 改为动态检测组件是否有 `text` 属性
- 修复 LLM 工具调用被中断的问题：
  - 使用 `asyncio.shield()` 保护任务不被外部取消
  - API 调用一旦开始必须完成（除非 API 调用本身失败）
- 优化重试逻辑，避免重复扣费：
  - 新增 `_is_network_error()` 方法，区分网络错误和 API 错误
  - 只有网络错误（超时、连接失败）才重试
  - API 返回错误（内容策略、模型错误等）不重试

### v2.4.4
- 修复消息文本提取失败导致命令无法识别的问题：
  - 增强 `_get_message_text` 方法，支持多种消息链组件提取方式
  - 添加对 `Comp.Plain`、`Comp.Text`、`message_obj.content` 的支持
  - 添加详细的调试日志，便于排查文本提取问题
- 新增图片发送失败时的 URL 殿后机制：
  - 当所有发送方式（本地文件→base64→压缩兜底）都失败时
  - 自动发送原始 URL 让用户手动访问
- 去除所有 emoji 元素，改为可爱的猫猫语录风格：
  - 所有用户提示消息改为猫猫风格
  - 更友好的错误提示和操作引导
- 优化日志策略，防止多人聊天时面板卡死：
  - 新增 `_log(msg, panel=False)` 方法，所有日志始终写入文件日志
  - 仅关键信息（URL、成功/失败状态、总耗时）输出到 AstrBot 面板
  - 详细日志（保存路径、下载进度、重试过程）仅记录到文件
- 修复超时机制，确保任务不被强制中断：
  - 移除 `asyncio.wait_for(timeout=180)` 强制超时
  - 一旦任务开始就必须走完，只在 API 调用失败时中断
  - 让 API 自己控制超时，避免中断正在生成中的任务
- 新增任务数据记录功能：
  - 新增 `task_data` 目录，每个任务生成独立的 JSON 文件
  - 记录全量信息：任务ID、开始/结束时间、提示词、图片数量
  - 分阶段记录：API调用（传入参数、返回结果、耗时）、保存（路径、耗时）、发送（方法、耗时）
  - 记录总耗时，便于性能分析
- 优化 pipeline 速度：
  - 减少重复操作：已有原始数据时直接编码 base64，避免重复读文件
  - 重试间隔从 1 秒优化为 0.5 秒，减少等待时间
  - 所有日志带任务ID前缀，便于追踪

### v2.4.3
- 重构 LLM 工具调用流程，统一走完整生成链路：
  - `_llm_tool_job` 不再直接调用底层 `generate_image_internal`，改为调用 `_generate_and_send`
  - 完整流程：视觉分析 → 提示词修饰 → 生成 → 保存到本地 → 多重发送（本地文件/base64/URL 三级降级，每种重试 3 次）→ 压缩兜底 → URL 兜底
  - 确保 LLM 工具调用与 `/img` 命令行为完全一致
  - `core/llm_tools.py` 简化为仅负责返回结果文本，不再重复处理发送逻辑
- 修复 LLM 工具调用发送图片失败问题：
  - `/img` 命令使用 `_send_image_with_retry` 有多级重试，正常发送
  - LLM 工具调用（AI 生成/`/生成`）之前使用自己的简单发送逻辑，无重试机制，导致 NapCat NT 协议超时失败
  - 现在 LLM 工具调用统一使用 `plugin._send_image_with_retry`，与 `/img` 命令保持一致
  - 支持本地文件 → base64 → URL 三级降级，每种方法重试 3 次

### v2.4.2
- 新增插件数据目录文件日志：
  - 新增 `_setup_file_logger` 方法，在插件数据目录 `data/astrbot_plugin_imageproductor/logs/` 下创建 `imageproducer.log`
  - 新增 `_log` 方法，同时输出到 AstrBot 日志面板和插件文件日志
  - 所有关键步骤（生成URL、下载、保存、发送、失败、超时）均双路记录
- 日志面板同步显示URL：
  - 生成完毕后立即在日志面板显示图片URL或数据信息
  - 无URL时明确标注"无URL，使用inline base64数据"
- 图片存储位置保持不变（`AstrBot/data/ai_images`）
- 修复本地图片无法打开的问题：
  - 新增 `_detect_image_format` 方法，通过文件头魔数检测图片实际格式（PNG/JPG/GIF/WEBP）
  - 修复文件扩展名固定为 `.jpg` 的问题，现在根据实际图片格式保存正确的扩展名
  - 保存文件时记录格式和大小信息到日志
- 修复 Chat API 返回 data URI 格式图片无法正确处理的问题：
  - 新增对 `![image](data:image/png;base64,...)` 格式的支持
  - 正确提取 data URI 中的 base64 数据并解码保存
  - 修复因 data URI 被误判为纯 base64 导致保存无效文件的问题
- 优化图片发送机制，避免一次熔断：
  - 新增 `_send_image_with_retry` 方法，提供多种发送方式（本地文件/base64/URL）
  - 每种方法支持重试（默认2次），失败后才降级到下一个方法
  - 所有方法失败后，自动压缩图片（512KB）重新发送作为兜底
  - 最终兜底：发送图片URL链接
- 确保图片质量不被压缩：
  - `_save_image_to_ai_images` 默认不压缩（compress=False），保持原始质量
  - 仅在发送全部失败时才压缩图片作为兜底
  - 压缩策略：先降低质量（85→20），若仍超限则按比例缩小尺寸
  - 支持动画图片处理（只取第一帧），自动转换 RGBA 等模式为 RGB
- 完善日志输出，确保 URL 全覆盖：
  - 成功时：记录生成图片 URL / 图片数据大小 / Markdown 链接 / data URI 格式
  - 失败时：记录失败原因及可用 URL（如有）
  - 超时时：记录超时提示词信息
  - 发送失败时：记录所有可用 URL 便于排查
- 修复 `/生成` 命令消息错乱问题：
  - 提示词生成失败时不再发送"失败"消息，改为"使用简化提示词生成图像"
  - 避免先显示失败后图片又成功导致的矛盾信息
- 处理安全壁垒导致的 400 错误：
  - 新增 `_is_safety_blocked` 方法，识别安全策略拦截
  - `_should_not_retry` 增加安全策略相关关键词，避免无意义重试
  - 给用户友好提示："提示词可能触发了内容安全策略，请尝试修改描述后重试"
- 修复 LLM 工具调用发送图片失败问题（`core/llm_tools.py`）：
  - 修复 `'str' object has no attribute 'chain'` 错误：URL 兜底发送改用 `MessageChain(chain=[Comp.Text(...)])`，不再直接传纯字符串
  - 修复发送逻辑嵌套过深导致降级遗漏的问题：改用 `sent_successfully` 标志控制发送流程
  - 本地文件 → base64 → URL 三级降级，任一级成功即停止
  - 所有发送失败时返回"图片已生成但发送失败"，不再静默

### v2.4.1
- 修复引用合并消息中多图无法识别的问题：
  - 新增 `_fetch_forward_message` 方法，通过平台 API（`get_forward_msg`）获取合并转发消息的详细内容
  - 新增 `_extract_image_urls_from_event_async` 异步主方法，支持从 `Reply`、`Forward`、`Nodes`、`Node` 组件中递归提取图片
  - 新增 `_extract_image_from_component_async` 通用异步方法，统一处理所有消息组件中的图片
  - 新增 `_extract_image_from_node_async` 方法，专门处理 Node 组件中的 message 列表
  - 支持从 `event.message_obj.raw_message` 中提取转发消息中的图片（备用方案）
  - 修复 `Image` 组件只检查 `url` 属性的问题，现在同时支持 `file` 属性（包括 HTTP URL、本地路径、base64 格式）
  - 新增 `_download_image` 对 base64 格式图片的支持，无需网络下载直接解码
  - 更新所有图片提取调用点，统一使用异步版本方法
- 优化图片提取逻辑：支持所有格式的图片 URL（HTTP、base64、本地文件）
- 增强调试日志：详细输出消息组件结构、API 调用结果、图片提取过程，便于排查问题

### v2.4.0
- 新增中文提示词支持：可在设置中开启 `allow_chinese_prompt` 选项，允许 LLM 生成中文提示词
- 全面支持中文提示词：所有涉及提示词生成的路径均支持中文输出
  - `/生成` - LLM 优化后生成中文提示词
  - `/img` + 图片 - 视觉分析后使用中文提示词组合
  - `/文生图` - 直接文字生成支持中文
  - `/图生图` - 参考图片直接生成支持中文
  - 非 LLM 修饰模式 - 用户提示词与视觉分析结果使用中文结合
- 优化视觉分析提示词：所有 Provider（OpenAI/智谱/千问/百度/混元/Grok）的 `_analyze_reference_images` 方法支持 `use_chinese` 参数
- 优化 LLM 修饰提示词：`_refine_prompt_with_llm` 方法根据配置动态切换中英文 System Prompt
- 优化 `_generate_prompt_internal`：根据配置动态切换中英文提示词生成
- 保持向后兼容：默认关闭中文提示词，确保现有用户不受影响

### v2.3.0
- 修复 Chat API 返回 Markdown 图片格式（`![image](url)`）时的解码错误
- 优化图片发送降级逻辑：下载成功时按「本地文件 → base64 → URL」顺序降级，下载失败直接发送 URL
- 修复图片发送时先发错误消息再发图片的问题，确保只发送一次
- 修复图片重复发送问题，使用 `image_sent` 标志控制发送状态
- 修复指令重复回复问题：`on_message` 唤醒命令直接跳过，仅由 `@filter.command` 装饰器处理
- 移除所有 img 相关指令的 HTML 标签（`<b>`、`<code>` 等），使用纯文本格式更直观
- 修复 `/生成 + 图片 + 描述` 不带图参考生成问题：避免 main.py 与 provider 重复视觉分析，引入 `vision_processed` 标记
- 优化视觉模型分析提示词，支持多图参考详细描述，自动识别图片数量并提取共同风格特征
- 优化 LLM 提示词生成 System Prompt：增加风格元素、光线、色彩、构图、视角等结构化要求，提升生成质量
- 优化非 LLM 修饰模式：用户提示词与视觉模型分析结果结合，确保多图参考风格融入最终提示词
- 所有 Provider（OpenAI/智谱/千问/百度/混元/Grok）同步支持 `vision_processed` 参数，避免重复分析

### v2.2.0
- 配置结构优化：恢复主/备用提供商固定结构，API密钥改为列表类型，支持添加多个Key
- 多模态模型配置化：从配置读取多模态模型列表，预设30+主流多模态模型
- 智能重试机制：调用成功但被模型终止（内容策略、NSFW等）不再重试，防止浪费余额
- 新增 `_should_not_retry()` 方法，识别不可重试错误类型
- 新增 `_is_multimodal_model()` 方法，统一多模态模型检测逻辑
- 新增 `_rotate_api_key()` 方法，API Key失败时自动切换到下一个
- 修复多模态模型API调用错误：多模态模型使用 `/chat/completions` 端点，传统模型使用 `/images/generations` 端点
- 修复所有Provider（OpenAI/智谱/千问/百度/混元/Grok）的多模态模型调用问题

### v2.1.0
- 新增指令体系：`/文生图`、`/图生图`、`/以图生图`、`/参考生图`
- 优化LLM调用逻辑：`/img`、`/文生图`、`/图生图` 直接调用图像API，不经过LLM
- 修复图片丢失bug：视觉模型分析后图片正确传递给图像生成API
- 修复所有Provider（OpenAI/智谱/千问/百度/混元/Grok）的图片传递问题
- 简化帮助文档，明确区分直接生成和LLM优化生成指令

### v2.0.1
- 修复图像生成成功后未自动保存的问题
- 新增 `_download_and_save` 方法，支持URL图像下载保存
- 生成成功后优先保存本地，再发送结果

### v2.0.0
- 架构重构 - 采用Provider模式，模块化设计
- 新增命令别名系统
- 新增前缀匹配机制
- 新增白名单系统
- 新增管理员权限检查
- 改进LLM工具调用描述
- 分层配置结构

### v1.1.0
- 适配AstrBot v4.19.2
- 配置界面优化
- 主备切换支持

### v1.0.0
- 初始版本
