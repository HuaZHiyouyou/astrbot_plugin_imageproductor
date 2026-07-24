"""
Base64 解码工具
支持并发安全的图片解码和保存
"""

import os
import base64
import asyncio
import re
import aiohttp
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import logging


class Base64Decoder:
    """Base64 解码器，支持并发安全的图片解码和保存"""

    def __init__(self, save_dir: Path, session: Optional[aiohttp.ClientSession] = None):
        """
        初始化解码器
        
        Args:
            save_dir: 图片保存目录
            session: aiohttp 会话（可选）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session = session
        self._file_lock = asyncio.Lock()
        self.logger = logging.getLogger("ImageProducerFile")

    def _extract_base64_data(self, data: str) -> Tuple[Optional[str], str]:
        """
        从各种格式中提取 base64 数据
        
        支持格式：
        - data:image/png;base64,xxxx
        - data:image/jpeg;base64,xxxx
        - 纯 base64 字符串
        
        Returns:
            (mime_type, base64_data)
        """
        # data URI 格式
        if data.startswith("data:image/"):
            match = re.match(r"data:image/([^;]+);base64,(.+)", data, re.DOTALL)
            if match:
                mime = f"image/{match.group(1)}"
                b64_data = match.group(2)
                return mime, b64_data
        
        # 纯 base64 字符串，默认为 png
        if data.startswith("/") or data.startswith("iVBOR") or data.startswith("/9j/"):
            return "image/png", data
        
        # HTTP URL
        if data.startswith("http://") or data.startswith("https://"):
            return None, data
        
        return None, data

    def _detect_image_format(self, b64_data: str) -> str:
        """
        通过 magic number 检测图片格式
        
        Args:
            b64_data: base64 编码的图片数据
            
        Returns:
            文件扩展名（如 .png, .jpg）
        """
        try:
            # 解码前几个字节用于格式检测
            sample = base64.b64decode(b64_data[:100])
            
            # PNG: 89 50 4E 47
            if sample[:4] == b'\x89PNG':
                return ".png"
            
            # JPEG: FF D8 FF
            if sample[:3] == b'\xff\xd8\xff':
                return ".jpg"
            
            # GIF: 47 49 46 38
            if sample[:4] == b'GIF8':
                return ".gif"
            
            # WebP: 52 49 46 46 ... 57 45 42 50
            if sample[:4] == b'RIFF' and sample[8:12] == b'WEBP':
                return ".webp"
            
            # BMP: 42 4D
            if sample[:2] == b'BM':
                return ".bmp"
            
        except Exception:
            pass
        
        return ".png"  # 默认 png

    def _generate_filename(self, b64_data: str, ext: str = ".png") -> str:
        """
        根据内容生成唯一文件名
        
        Args:
            b64_data: base64 数据
            ext: 文件扩展名
            
        Returns:
            文件名
        """
        # 使用 base64 的 MD5 作为唯一标识
        content_hash = hashlib.md5(b64_data.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"img_{timestamp}_{content_hash}{ext}"

    async def decode_and_save(self, data: str, task_id: str = "") -> Tuple[bool, Optional[str], str]:
        """
        解码并保存图片（并发安全）
        
        Args:
            data: base64 数据或 data URI 或 URL
            task_id: 任务 ID（用于日志）
            
        Returns:
            (success, file_path, message)
        """
        mime_type, b64_data = self._extract_base64_data(data)
        
        # HTTP URL - 需要下载
        if mime_type is None and (b64_data.startswith("http://") or b64_data.startswith("https://")):
            return await self._download_and_save(b64_data, task_id)
        
        # 纯文本或其他非图片数据
        if mime_type is None:
            return False, None, f"无法识别的数据格式: {b64_data[:50]}..."
        
        try:
            # 检测图片格式
            ext = self._detect_image_format(b64_data)
            
            # 生成文件名
            filename = self._generate_filename(b64_data, ext)
            file_path = self.save_dir / filename
            
            # 解码
            image_bytes = base64.b64decode(b64_data)
            
            # 并发安全写入
            async with self._file_lock:
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
            
            self.logger.info(f"[Base64Decoder] 任务 {task_id} 解码成功: {file_path}")
            return True, str(file_path), f"图片已保存: {filename}"
            
        except Exception as e:
            error_msg = f"解码失败: {e}"
            self.logger.error(f"[Base64Decoder] 任务 {task_id} {error_msg}")
            return False, None, error_msg

    async def _download_and_save(self, url: str, task_id: str = "") -> Tuple[bool, Optional[str], str]:
        """
        从 URL 下载图片并保存
        
        Args:
            url: 图片 URL
            task_id: 任务 ID
            
        Returns:
            (success, file_path, message)
        """
        if not self.session:
            return False, None, "未配置 HTTP 会话，无法下载图片"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return False, None, f"下载失败: HTTP {response.status}"
                
                # 从 Content-Type 获取格式
                content_type = response.headers.get("Content-Type", "")
                if "png" in content_type:
                    ext = ".png"
                elif "jpeg" in content_type or "jpg" in content_type:
                    ext = ".jpg"
                elif "gif" in content_type:
                    ext = ".gif"
                elif "webp" in content_type:
                    ext = ".webp"
                else:
                    ext = ".png"
                
                # 读取内容
                content = await response.read()
                
                # 通过 magic number 二次确认
                if content[:4] == b'\x89PNG':
                    ext = ".png"
                elif content[:3] == b'\xff\xd8\xff':
                    ext = ".jpg"
                elif content[:4] == b'GIF8':
                    ext = ".gif"
                elif content[:4] == b'RIFF' and content[8:12] == b'WEBP':
                    ext = ".webp"
                
                # 生成文件名
                content_hash = hashlib.md5(content).hexdigest()[:12]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"img_{timestamp}_{content_hash}{ext}"
                file_path = self.save_dir / filename
                
                # 并发安全写入
                async with self._file_lock:
                    with open(file_path, "wb") as f:
                        f.write(content)
                
                self.logger.info(f"[Base64Decoder] 任务 {task_id} 下载成功: {file_path}")
                return True, str(file_path), f"图片已保存: {filename}"
                
        except Exception as e:
            error_msg = f"下载失败: {e}"
            self.logger.error(f"[Base64Decoder] 任务 {task_id} {error_msg}")
            return False, None, error_msg

    async def decode_batch(self, data_list: list, task_id: str = "") -> list:
        """
        批量解码图片
        
        Args:
            data_list: base64 数据列表
            task_id: 任务 ID
            
        Returns:
            结果列表 [(success, file_path, message), ...]
        """
        results = []
        for i, data in enumerate(data_list):
            result = await self.decode_and_save(data, f"{task_id}_batch{i}")
            results.append(result)
        return results