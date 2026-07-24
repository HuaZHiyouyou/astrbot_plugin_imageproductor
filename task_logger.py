"""
任务日志工具
每个任务独立日志文件，并发安全
"""

import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class TaskLogger:
    """任务日志器，每个任务独立日志文件"""
    
    def __init__(self, task_id: str, log_dir: Path):
        """
        初始化任务日志器
        
        Args:
            task_id: 任务 ID
            log_dir: 日志目录
        """
        self.task_id = task_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 任务日志文件
        self.log_file = self.log_dir / f"task_{task_id}.log"
        
        # 创建独立 logger
        self.logger = logging.getLogger(f"Task_{task_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加 handler
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # 任务数据
        self.start_time = datetime.now()
        self.stages = {}
        self.status = "running"
        self.result = None
    
    def info(self, msg: str):
        """记录信息日志"""
        self.logger.info(msg)
    
    def debug(self, msg: str):
        """记录调试日志"""
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        """记录警告日志"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """记录错误日志"""
        self.logger.error(msg)
    
    def stage_start(self, stage_name: str):
        """
        记录阶段开始
        
        Args:
            stage_name: 阶段名称
        """
        self.stages[stage_name] = {
            "start": datetime.now().isoformat(),
            "end": None,
            "duration": None
        }
        self.info(f"[阶段开始] {stage_name}")
    
    def stage_end(self, stage_name: str, success: bool = True):
        """
        记录阶段结束
        
        Args:
            stage_name: 阶段名称
            success: 是否成功
        """
        if stage_name in self.stages:
            end_time = datetime.now()
            start_time = datetime.fromisoformat(self.stages[stage_name]["start"])
            duration = (end_time - start_time).total_seconds()
            
            self.stages[stage_name]["end"] = end_time.isoformat()
            self.stages[stage_name]["duration"] = duration
            
            status = "成功" if success else "失败"
            self.info(f"[阶段结束] {stage_name} - {status} (耗时: {duration:.2f}s)")
    
    def complete(self, result: dict):
        """
        标记任务完成
        
        Args:
            result: 任务结果
        """
        self.status = "completed"
        self.result = result
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        self.info(f"[任务完成] 总耗时: {total_duration:.2f}s")
        
        # 保存任务摘要
        self._save_summary(end_time, total_duration)
    
    def fail(self, error: str):
        """
        标记任务失败
        
        Args:
            error: 错误信息
        """
        self.status = "failed"
        self.result = {"error": error}
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        self.error(f"[任务失败] {error}")
        self.info(f"[任务结束] 总耗时: {total_duration:.2f}s")
        
        # 保存任务摘要
        self._save_summary(end_time, total_duration)
    
    def _save_summary(self, end_time: datetime, total_duration: float):
        """保存任务摘要"""
        summary = {
            "task_id": self.task_id,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration": total_duration,
            "stages": self.stages,
            "result": self.result
        }
        
        summary_file = self.log_dir / f"task_{self.task_id}_summary.json"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存任务摘要失败: {e}")
    
    def close(self):
        """关闭日志器"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class TaskLoggerManager:
    """任务日志管理器，管理所有任务的日志器"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir: Path = None):
        if self._initialized:
            return
        
        self.log_dir = log_dir or Path("logs/tasks")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._loggers: dict = {}
        self._initialized = True
    
    async def create_logger(self, task_id: str) -> TaskLogger:
        """
        创建任务日志器
        
        Args:
            task_id: 任务 ID
            
        Returns:
            TaskLogger 实例
        """
        async with self._lock:
            if task_id in self._loggers:
                return self._loggers[task_id]
            
            logger = TaskLogger(task_id, self.log_dir)
            self._loggers[task_id] = logger
            return logger
    
    async def get_logger(self, task_id: str) -> Optional[TaskLogger]:
        """
        获取任务日志器
        
        Args:
            task_id: 任务 ID
            
        Returns:
            TaskLogger 实例或 None
        """
        return self._loggers.get(task_id)
    
    async def close_logger(self, task_id: str):
        """
        关闭任务日志器
        
        Args:
            task_id: 任务 ID
        """
        async with self._lock:
            if task_id in self._loggers:
                self._loggers[task_id].close()
                del self._loggers[task_id]
    
    def list_active_tasks(self) -> list:
        """列出活跃任务"""
        return list(self._loggers.keys())