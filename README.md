# 视频转文字工具

一个基于 Whisper 的视频音频转文字工具，支持GPU加速，界面简洁，使用方便。

## ✨ 特性

- 🎥 支持多种视频格式（mp4, avi, mov, wmv, flv, mkv等）
- 🔊 支持多种音频格式（mp3, wav, m4a等）
- 🚀 支持GPU加速，大幅提升转换速度
- 💡 智能选择最适合的模型
- 📂 支持批量处理
- 🔍 支持文件夹递归扫描
- 🛠 自动检查和安装依赖
- 🎯 自动下载所需组件
- 📝 输出带有时间戳的文本文件
- 🌏 优化支持中文识别

## 🖥 系统要求

- Windows 10 或更高版本
- 如果要使用GPU加速：
  - NVIDIA显卡
  - 最新版显卡驱动
  - 建议显存≥4GB

## 📦 下载和安装

1. 从 [Releases](https://github.com/your-username/your-repo/releases) 下载最新版本
2. 解压到任意文件夹
3. 双击运行 `视频转文字工具.exe`
4. 首次运行时会自动安装必要的依赖

## 🚀 使用方法

1. 启动程序
2. 选择要转换的视频文件或文件夹
3. 选择输出文件夹
4. 选择合适的模型：
   - ≥10GB 显存：large（最佳质量）
   - ≥8GB 显存：medium（平衡速度和质量）
   - ≥5GB 显存：small（平衡内存和质量）
   - <5GB 显存：base（基本使用）
   - CPU模式：base（适合CPU模式）
5. 点击"开始转换"

## 🔧 技术细节

- 语音识别：OpenAI Whisper
- GUI框架：PyQt5
- 视频处理：FFmpeg
- GPU加速：PyTorch + CUDA
- 自动化部署：PyInstaller

## 📋 功能特点

### 智能模型选择
- 自动检测系统配置
- 推荐最适合的模型
- 防止显存溢出

### 批量处理
- 支持多文件选择
- 支持文件夹导入
- 显示处理进度和预计时间

### GPU加速
- 自动检测GPU
- 支持CUDA加速
- 支持CPU回退模式

### 依赖管理
- 首次运行自动检查依赖
- 自动下载安装必要组件
- 显示详细的安装进度

## 🔍 常见问题

**Q: 程序无法启动？**  
A: 确保已安装最新的Visual C++运行库

**Q: 转换速度很慢？**  
A: 检查是否正确启用了GPU加速，可以点击"GPU诊断"按钮查看详情

**Q: 显存不足？**  
A: 尝试使用更小的模型，或切换到CPU模式

**Q: 如何选择合适的模型？**  
A: 程序会根据您的硬件配置自动推荐最适合的模型

## 🛠 开发相关

### 环境准备
```bash
pip install pyinstaller pillow openai-whisper torch torchvision torchaudio PyQt5
```

### 打包方法
```bash
# 生成图标
python create_icon.py

# 打包程序
pyinstaller --clean --onefile --noconsole --icon=app.ico --name="视频转文字工具" videoToText.py
```

## 📝 更新日志

### v1.0.0
- 首次发布
- 支持视频转文字
- 支持GPU加速
- 自动依赖安装

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- [FFmpeg](https://ffmpeg.org/)
- [PyTorch](https://pytorch.org/) 