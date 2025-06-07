from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import time
from datetime import datetime
from typing import Optional, List
import asyncio
from pathlib import Path
import whisper
import torch
import subprocess
import json
import threading
from queue import Queue

app = FastAPI(
    title="视频转文字API服务",
    description="基于Whisper的视频音频转文字API服务",
    version="1.0.0"
)

# 全局配置
class Config:
    MODEL_SIZE = "base"
    OUTPUT_DIR = "output"
    TEMP_DIR = "temp"
    USE_GPU = torch.cuda.is_available()
    WHISPER_MODEL = None
    SERVER_STATUS = "stopped"  # 服务器状态：stopped, running, error
    SERVER_ERROR = None  # 服务器错误信息
    TASK_COUNT = 0  # 当前任务数
    COMPLETED_TASKS = 0  # 已完成任务数
    STATUS_CALLBACK = None  # 状态回调函数

# 确保输出目录存在
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.TEMP_DIR, exist_ok=True)

# 响应模型
class TranscriptionResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TranscriptionResult(BaseModel):
    task_id: str
    status: str
    text: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    file_path: Optional[str] = None

# 任务状态存储
tasks = {}

def update_status(status=None, error=None, task_count=None, completed_tasks=None):
    """更新服务状态并通知GUI"""
    if status is not None:
        Config.SERVER_STATUS = status
    if error is not None:
        Config.SERVER_ERROR = error
    if task_count is not None:
        Config.TASK_COUNT = task_count
    if completed_tasks is not None:
        Config.COMPLETED_TASKS = completed_tasks
    
    if Config.STATUS_CALLBACK:
        Config.STATUS_CALLBACK({
            "status": Config.SERVER_STATUS,
            "error": Config.SERVER_ERROR,
            "task_count": Config.TASK_COUNT,
            "completed_tasks": Config.COMPLETED_TASKS
        })

async def process_video(task_id: str, video_path: str, model_size: str = "base"):
    try:
        start_time = time.time()
        tasks[task_id] = {"status": "processing", "text": None, "error": None}
        update_status(task_count=len(tasks))

        # 确保模型已加载
        if Config.WHISPER_MODEL is None:
            Config.WHISPER_MODEL = whisper.load_model(model_size, device="cuda" if Config.USE_GPU else "cpu")

        # 提取音频
        audio_path = os.path.join(Config.TEMP_DIR, f"{task_id}_audio.wav")
        result = subprocess.run([
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', audio_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"音频提取失败: {result.stderr}")

        # 转写音频
        result = Config.WHISPER_MODEL.transcribe(
            audio_path,
            language='zh',
            task='transcribe',
            fp16=Config.USE_GPU
        )

        # 保存结果
        output_filename = f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])

        # 清理临时文件
        os.remove(audio_path)
        os.remove(video_path)

        # 更新任务状态
        duration = time.time() - start_time
        tasks[task_id] = {
            "status": "completed",
            "text": result["text"],
            "duration": duration,
            "file_path": output_path
        }
        
        # 更新完成任务数
        update_status(completed_tasks=Config.COMPLETED_TASKS + 1)

    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        update_status(error=str(e))

@app.post("/api/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_size: str = "base"
):
    try:
        # 验证文件类型
        allowed_types = [
            'video/mp4', 'video/avi', 'video/x-msvideo',
            'video/quicktime', 'video/x-ms-wmv', 'video/x-flv',
            'video/x-matroska', 'video/webm'
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="不支持的文件类型")

        # 生成任务ID
        task_id = f"task_{int(time.time())}_{os.urandom(4).hex()}"
        
        # 保存上传的文件
        temp_video_path = os.path.join(Config.TEMP_DIR, f"{task_id}_{file.filename}")
        with open(temp_video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 启动后台处理任务
        background_tasks.add_task(
            process_video,
            task_id,
            temp_video_path,
            model_size
        )

        return TranscriptionResponse(
            task_id=task_id,
            status="accepted",
            message="任务已接受，正在处理中"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}", response_model=TranscriptionResult)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    return TranscriptionResult(
        task_id=task_id,
        status=task["status"],
        text=task.get("text"),
        error=task.get("error"),
        duration=task.get("duration"),
        file_path=task.get("file_path")
    )

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": Config.WHISPER_MODEL is not None,
        "tasks": {
            "total": Config.TASK_COUNT,
            "completed": Config.COMPLETED_TASKS
        }
    }

class APIServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.server_thread = None
        self.should_exit = False

    def run_server(self):
        """在单独的线程中运行服务器"""
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="error")
        server = uvicorn.Server(config)
        try:
            update_status(status="running")
            server.run()
        except Exception as e:
            update_status(status="error", error=str(e))

    def start(self):
        """启动服务器"""
        if self.server_thread is None or not self.server_thread.is_alive():
            self.should_exit = False
            self.server_thread = threading.Thread(target=self.run_server)
            self.server_thread.daemon = True  # 设置为守护线程
            self.server_thread.start()
            return True
        return False

    def stop(self):
        """停止服务器"""
        self.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            # 这里可以添加优雅关闭的逻辑
            update_status(status="stopped")
            return True
        return False

def register_status_callback(callback):
    """注册状态更新回调函数"""
    Config.STATUS_CALLBACK = callback

def start_api_server(host="0.0.0.0", port=8000):
    """启动API服务器"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api_server() 