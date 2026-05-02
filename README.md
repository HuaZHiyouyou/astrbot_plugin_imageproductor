# AstrBot AI图像生成插件 v1.0

[![AstrBot](https://img.shields.io/badge/AstrBot-4.16+-blue.svg)](https://github.com/AstrBotDevs/AstrBot)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-仓库-blue?logo=github)](https://github.com/HuaZHiyouyou/astrbot_plugin_imageproductor)

这是一个功能强大的多平台AI图像生成插件，专为AstrBot机器人框架设计。支持OpenAI DALL-E、Google Gemini、xAI Grok、Seed/Seedream和Stable Diffusion等多个AI图像生成平台。

## 功能特性

### 多平台支持
- **OpenAI DALL-E** - 高质量的商业级图像生成，支持GPT-4V图像分析
- **Google Gemini Imagen** - Google先进的AI图像生成，原生支持图像输入
- **xAI Grok** - 支持Grok-Imagine模型，支持图像分析
- **Seed/Seedream** - 字节跳动的AI图像生成平台
- **智谱 AI** - GLM-4V 视觉模型，支持图像分析
- **阿里云 千问** - Qwen-VL 视觉模型
- **百度 文心一言** - ERNIE-VL 视觉模型
- **腾讯 混元** - Hunyuan Vision 视觉模型
- **Stable Diffusion** - 本地部署的开源图像生成，原生支持img2img
- **Claude Vision** - Anthropic Claude 视觉理解
- **DeepSeek Vision** - DeepSeek 视觉模型
- **火山引擎 Vision** - 火山引擎视觉模型
- **阶跃星辰 Vision** - 阶跃星辰视觉模型

### 核心功能
- 多平台支持 - 13个主流AI图像生成平台
- 主备切换 - 每个平台支持主用/备用双配置
- 视觉模型独立配置 - 支持独立的视觉模型 API 配置
- 命令别名 - 短命令快速访问（/img、/生图）
- 白名单 - 支持群组和用户白名单控制
- 并发控制 - 智能任务队列，支持最多20个并发任务
- LLM工具调用 - 可作为工具被大语言模型调用
- 图像 + 文本输入 - 支持同时上传参考图片和文字提示
- AI提示词生成 - 自动生成专业的图像生成提示词
- LLM 提示词修饰 - 视觉模型分析图片 + AstrBot LLM 修饰 → 优质提示词
- 两阶段生成 - 先AI生成提示词，再生成图像
- 预设提示词 - 内置20+风格预设，快速使用
- 收集模式 - 支持分阶段收集多张图片和文字后统一生成
- 重试机制 - 自动重试失败的图像生成请求
- 代理支持 - 支持配置代理访问API
- 本地保存 - 自动保存生成的图片到本地

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

### 基础命令
| 命令 | 别名 | 说明 |
|------|------|------|
| `/img [提示词]` | `/aimg`, `/生图`, `/ai生图` | 使用默认配置生成图像 |
| `/img帮助` | `/aimg帮助`, `/imgh` | 显示帮助信息 |
| `/img设置` | `/aimg设置` | 查看当前配置 |
| `/img平台` | `/aimg平台`, `/imgp` | 查看平台状态（需管理员） |

### 高级命令
| 命令 | 别名 | 说明 |
|------|------|------|
| `/提示词 [描述]` | - | AI生成专业提示词 |
| `/生成 [描述]` | - | AI生成提示词并创作图像 |

### 预设提示词命令
| 命令 | 别名 | 说明 |
|------|------|------|
| `/img列表` | `/aimg列表`, `/生图列表` | 查看所有预设提示词 |
| `/img查看 [触发词]` | `/aimg查看`, `/生图查看` | 查看预设提示词详情 |
| `/img [触发词] [文本]` | - | 使用预设提示词生成（如: /img 手办化 一只猫） |

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

### v1.0.0
- 初始版本发布
- 架构重构 - 采用Provider模式，模块化设计
- 新增命令别名系统
- 新增前缀匹配机制
- 新增白名单系统
- 新增管理员权限检查
- 改进LLM工具调用描述
- 分层配置结构
- 适配AstrBot v4.19.2
- 配置界面优化
- 主备切换支持
- 修复 FunctionTool 验证错误（Pydantic dataclass → 标准 dataclass）
- 修复 img 指令无响应问题（唤醒命令正则表达式优化）
- 修复视觉模型配置不识别问题（所有平台支持独立的视觉模型配置）
- 修复 OCR 模型 + image 模型报错（改为两阶段流程：视觉分析 → 图像生成）
- 修复 img+提示词连续生成2张图片问题（避免 on_message 和 command 重复触发）
- 简化反馈消息（减少冗余消息，提升用户体验）
- 新增 AstrBot LLM 修饰功能（视觉模型分析图片描述 + AstrBot LLM 修饰 → 优质提示词）
- 修复默认平台初始化问题（主提供商未启用时使用第一个已加载的提供商）
- 所有平台视觉模型配置支持（backup_api_key/backup_api_url 用于视觉模型）
- 修复图像生成成功后未自动保存的问题
- 新增 `_download_and_save` 方法，支持URL图像下载保存
- 生成成功后优先保存本地，再发送结果
- 新增智谱 AI、千问、文心一言、混元 4个平台支持
- 新增图像 + 文本输入支持
- 新增AI提示词生成功能
- 新增两阶段生成（提示词+图像）
- 新增LLM工具调用支持（三个工具）
- 新增预设提示词功能（20+风格预设）
- 新增收集模式
- 新增重试机制
- 新增代理支持
- 优化错误处理，确保用户100%收到回复
- 所有平台支持图像输入，视觉模型分析参考图片
- Stable Diffusion 支持原生img2img
- 新增 `/img列表`、`/img查看` 命令
- 配置结构扁平化，解决插件加载问题
- 更新帮助文档，添加新功能说明
