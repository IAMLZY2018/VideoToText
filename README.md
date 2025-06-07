# 视频转文字工具 (GPU加速版)

这是一个基于 Whisper 的视频转文字工具，支持GPU加速，可以快速将视频中的语音转换为文字。

## 功能特点

- 支持多种视频格式（mp4, avi, mov, wmv, flv, mkv等）
- 支持GPU加速（需要NVIDIA显卡）
- 支持批量处理
- 支持中文识别
- 提供API服务模式
- 自动检测和安装依赖

## 系统要求

- Windows 10/11
- NVIDIA显卡（推荐，但不是必需）
- 4GB以上内存（使用GPU时建议8GB以上）

## 使用方法

1. 双击运行 `VideoToText.exe`
2. 选择要转换的视频文件或文件夹
3. 选择输出文件夹
4. 选择合适的模型（根据系统配置自动推荐）
5. 点击"开始转换"

## 注意事项

1. 首次运行时会自动下载必要的模型文件
2. 如果没有找到ffmpeg，程序会自动下载
3. 转换速度取决于视频长度和系统配置
4. GPU模式需要NVIDIA显卡和最新驱动

## API模式

启动API服务：
```bash
VideoToText.exe --mode api --host 0.0.0.0 --port 8000
```

API接口：
- POST /api/v1/transcribe - 提交转换任务
- GET /api/v1/tasks/{task_id} - 查询任务状态
- GET /api/v1/health - 健康检查

## 常见问题

1. 如果提示缺少ffmpeg，请确保ffmpeg.exe在程序同目录下
2. 如果GPU模式不可用，请更新显卡驱动
3. 如果内存不足，请尝试使用较小的模型

## 技术支持

如有问题，请提交Issue或发送邮件。 