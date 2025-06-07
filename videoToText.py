import sys
import os
import time
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QTextEdit, QFileDialog,
                             QProgressBar, QMessageBox, QComboBox, QCheckBox, QToolTip,
                             QTreeView, QListView, QAbstractItemView, QDialog, QScrollArea,
                             QGroupBox, QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPoint
from PyQt5.QtGui import QFont, QCursor, QIntValidator
import whisper
import torch
from importlib.metadata import version, PackageNotFoundError
import requests
import zipfile
import argparse
from api_service import start_api_server
import shutil
# 这里是核心代码
class DependencyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("依赖检查")
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        
        # 状态标签
        self.status_label = QLabel("正在检查依赖...")
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # 日志区域
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        
        self.setLayout(layout)
        
    def log_message(self, message, replace_last=False):
        if replace_last:
            # 删除最后一行
            cursor = self.log.textCursor()
            cursor.movePosition(cursor.End)
            cursor.movePosition(cursor.StartOfLine, cursor.KeepAnchor)
            cursor.removeSelectedText()
            # 如果不是第一行，删除换行符
            if not self.log.toPlainText().endswith('\n') and self.log.toPlainText() != '':
                cursor.deletePreviousChar()
        
        self.log.append(message)
        # 自动滚动到底部
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.End)
        self.log.setTextCursor(cursor)
        QApplication.processEvents()
        
    def set_status(self, status):
        self.status_label.setText(status)
        QApplication.processEvents()
        
    def set_progress(self, value):
        self.progress.setValue(value)
        QApplication.processEvents()

def check_package_version(package_name, min_version):
    """检查包版本是否满足最小要求"""
    try:
        current_version = version(package_name)
        # 将版本号转换为元组进行比较
        current = tuple(map(int, current_version.split('.')))
        required = tuple(map(int, min_version.strip('>=').split('.')))
        return current >= required
    except PackageNotFoundError:
        return False
    except Exception:
        return False  # 如果版本比较失败，返回False

def check_and_install_dependencies():
    try:
        # 确保有一个QApplication实例
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        dialog = DependencyDialog()
        dialog.show()
        QApplication.processEvents()
        
        # 快速检查基本依赖
        dialog.set_status("正在快速检查基本依赖...")
        dialog.set_progress(10)
        
        # 检查CUDA可用性（不检查详细信息）
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            dialog.log_message("✓ CUDA 可用")
        else:
            dialog.log_message("⚠️ CUDA 不可用，将使用CPU模式")
        
        # 快速检查ffmpeg
        dialog.set_status("正在检查ffmpeg...")
        dialog.set_progress(30)
        ffmpeg_found = False
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, 
                                 text=True, 
                                 encoding='utf-8',
                                 errors='ignore')
            if result.returncode == 0:
                dialog.log_message("✓ 系统已安装ffmpeg")
                ffmpeg_found = True
        except:
            pass
        
        if not ffmpeg_found and os.path.exists("ffmpeg.exe"):
            dialog.log_message("✓ 本地已有ffmpeg.exe")
            ffmpeg_found = True
        
        if not ffmpeg_found:
            dialog.log_message("⚠️ 未找到ffmpeg，将在首次使用时下载")
        
        # 快速检查Whisper模型
        dialog.set_status("正在检查Whisper...")
        dialog.set_progress(60)
        try:
            import whisper
            dialog.log_message("✓ Whisper 已安装")
        except ImportError:
            dialog.log_message("⚠️ Whisper 未安装，将在首次使用时安装")
        
        # 完成基本检查
        dialog.set_status("基本检查完成")
        dialog.set_progress(100)
        dialog.log_message("\n✓ 基本检查完成！")
        QApplication.processEvents()
        time.sleep(0.5)  # 减少等待时间
        dialog.accept()
        return True
        
    except Exception as e:
        if 'dialog' in locals():
            dialog.log_message(f"\n错误: {str(e)}")
            dialog.set_status("检查失败")
            dialog.exec_()
        return False

class VideoProcessor(QThread):
    # 信号定义
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, video_files, output_folder, model_size="base", use_gpu=True, ffmpeg_path=""):
        super().__init__()
        self.video_files = video_files
        self.output_folder = output_folder
        self.model_size = model_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.is_running = True
        self.ffmpeg_path = ffmpeg_path
        self.whisper_model = None

    def run(self):
        try:
            # 检查并安装必要的依赖
            if not self.check_and_install_dependencies():
                return

            # 初始化Whisper模型
            self.log_signal.emit("正在加载Whisper模型...")
            device = "cuda" if self.use_gpu else "cpu"
            self.log_signal.emit(f"使用设备: {device}")

            # 禁用不必要的警告
            import warnings
            warnings.filterwarnings("ignore", message="Failed to launch Triton kernels")

            try:
                self.whisper_model = whisper.load_model(self.model_size, device=device)
                self.log_signal.emit(f"Whisper {self.model_size} 模型加载成功")
            except Exception as e:
                self.log_signal.emit(f"模型加载失败: {str(e)}")
                return

            total_files = len(self.video_files)
            self.log_signal.emit(f"开始处理，共发现 {total_files} 个视频文件")

            for i, video_path in enumerate(self.video_files):
                if not self.is_running:
                    break

                start_time = time.time()
                video_name = Path(video_path).stem
                current_time = time.strftime("%H%M%S")  # 获取当前时间（时分秒）

                self.log_signal.emit(f"正在处理: {video_name}")

                try:
                    # 使用ffmpeg提取音频
                    audio_path = self.extract_audio_with_ffmpeg(video_path, video_name)

                    # 使用Whisper转换为文字
                    text_content = self.audio_to_text_with_whisper(audio_path)

                    # 保存文本文件（添加时间戳）
                    txt_filename = f"{video_name}_{current_time}.txt"
                    txt_path = os.path.join(self.output_folder, txt_filename)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)

                    # 清理临时音频文件
                    if os.path.exists(audio_path):
                        os.remove(audio_path)

                    # 计算处理时间和文字数量
                    end_time = time.time()
                    duration = end_time - start_time
                    word_count = len(text_content)
                    remaining = total_files - i - 1

                    self.log_signal.emit(f"完成: {video_name}")
                    self.log_signal.emit(f"耗时: {duration:.2f}秒, 文字数量: {word_count}, 剩余: {remaining}个文件")
                    self.log_signal.emit(f"输出文件: {txt_filename}")
                    self.log_signal.emit(f"输出路径: {txt_path}")
                    self.log_signal.emit("-" * 50)

                except Exception as e:
                    self.log_signal.emit(f"处理失败 {video_name}: {str(e)}")

                # 更新进度
                progress = int((i + 1) / total_files * 100)
                self.progress_signal.emit(progress)

            self.log_signal.emit("所有文件处理完成！")

        except Exception as e:
            self.log_signal.emit(f"处理过程中出现错误: {str(e)}")

        finally:
            self.finished_signal.emit()

    def extract_audio_with_ffmpeg(self, video_path, video_name):
        """使用ffmpeg从视频中提取音频"""
        try:
            # 临时音频文件路径
            temp_audio_path = os.path.join(self.output_folder, f"temp_audio_{video_name}.wav")

            # 使用用户选择的ffmpeg路径
            if self.ffmpeg_path:
                ffmpeg_cmd = self.ffmpeg_path
            else:
                # 尝试多种ffmpeg调用方式作为备选
                ffmpeg_commands = [
                    'ffmpeg',
                    'ffmpeg.exe',
                    r'C:\ffmpeg\bin\ffmpeg.exe',
                    r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
                ]

                ffmpeg_cmd = None
                for cmd in ffmpeg_commands:
                    try:
                        result = subprocess.run([cmd, '-version'], 
                                             capture_output=True, 
                                             text=True, 
                                             encoding='utf-8',  # 指定编码
                                             errors='ignore',   # 忽略无法解码的字符
                                             timeout=5)
                        if result.returncode == 0:
                            ffmpeg_cmd = cmd
                            break
                    except:
                        continue

                if not ffmpeg_cmd:
                    raise Exception("找不到可用的ffmpeg，请手动选择ffmpeg路径")

            # 构建ffmpeg命令
            cmd = [
                ffmpeg_cmd,
                '-i', video_path,
                '-vn',  # 不要视频
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz采样率
                '-ac', '1',  # 单声道
                '-y',  # 覆盖现有文件
                temp_audio_path
            ]

            self.log_signal.emit(f"提取音频: {video_name}")
            self.log_signal.emit(f"使用ffmpeg: {ffmpeg_cmd}")

            # 执行ffmpeg命令
            try:
                result = subprocess.run(cmd, 
                                     capture_output=True, 
                                     text=True, 
                                     encoding='utf-8',  # 指定编码
                                     errors='ignore',   # 忽略无法解码的字符
                                     timeout=300)
            except:
                # 如果直接调用失败，尝试使用shell=True
                cmd_str = ' '.join([f'"{arg}"' if ' ' in arg else arg for arg in cmd])
                result = subprocess.run(cmd_str, 
                                     capture_output=True, 
                                     text=True, 
                                     encoding='utf-8',  # 指定编码
                                     errors='ignore',   # 忽略无法解码的字符
                                     shell=True, 
                                     timeout=300)

            if result.returncode != 0:
                raise Exception(f"ffmpeg错误: {result.stderr}")

            if not os.path.exists(temp_audio_path):
                raise Exception("音频文件提取失败")

            return temp_audio_path

        except Exception as e:
            raise Exception(f"音频提取失败: {str(e)}")

    def audio_to_text_with_whisper(self, audio_path):
        """使用Whisper将音频转换为文字"""
        try:
            self.log_signal.emit("正在进行语音识别...")
            self.log_signal.emit(f"使用音频文件: {audio_path}")

            # 使用Whisper进行转录
            result = self.whisper_model.transcribe(
                audio_path,
                language='zh',           # 指定中文
                task='transcribe',       # 转录任务
                fp16=torch.cuda.is_available(),  # 如果有GPU则使用fp16加速
                initial_prompt="以下是普通话的转录文本，包含标点符号：",  # 提示词以引导输出带标点的文本
                word_timestamps=True,    # 启用词级时间戳，有助于更好的分段
                condition_on_previous_text=True,  # 考虑上下文
                temperature=0.0,         # 降低随机性，使输出更稳定
                best_of=1,              # 只生成一个结果
                no_speech_threshold=0.6  # 调整无语音检测阈值
            )

            # 获取转录文本
            text = result["text"].strip()
            self.log_signal.emit(f"识别完成，文本长度: {len(text)} 字符")

            if not text:
                self.log_signal.emit("警告: 未识别到语音内容")
                return "未识别到语音内容"

            # 如果文本中缺少标点，尝试进行简单的分段处理
            if len(text) > 0 and not any(p in text for p in '，。！？、'):
                self.log_signal.emit("正在优化文本格式...")
                # 使用时间戳信息进行分段
                segments = result.get("segments", [])
                formatted_text = ""
                for segment in segments:
                    segment_text = segment.get("text", "").strip()
                    if segment_text:
                        # 如果分段文本末尾没有标点，添加句号
                        if not segment_text[-1] in '，。！？、':
                            segment_text += '。'
                        formatted_text += segment_text + '\n'
                text = formatted_text.strip()

            return text

        except Exception as e:
            self.log_signal.emit(f"语音转文字失败: {str(e)}")
            raise Exception(f"语音转文字失败: {str(e)}")

    def stop(self):
        self.is_running = False

    def check_and_install_dependencies(self):
        """检查并安装必要的依赖"""
        try:
            # 检查CUDA工具包（如果使用GPU）
            if self.use_gpu:
                try:
                    import triton
                    self.log_signal.emit("✓ Triton CUDA 工具包已安装")
                except ImportError:
                    self.log_signal.emit("⚠️ Triton CUDA 工具包未安装，某些GPU加速功能将不可用")

            # 检查ffmpeg
            if not self.ffmpeg_path:
                try:
                    result = subprocess.run(['ffmpeg', '-version'], 
                                         capture_output=True, 
                                         text=True, 
                                         encoding='utf-8',
                                         errors='ignore')
                    if result.returncode == 0:
                        self.log_signal.emit("✓ 系统已安装ffmpeg")
                        self.ffmpeg_path = 'ffmpeg'
                    else:
                        raise Exception("ffmpeg执行失败")
                except:
                    # 尝试下载ffmpeg
                    self.log_signal.emit("正在下载ffmpeg...")
                    if not self.download_ffmpeg():
                        self.log_signal.emit("❌ ffmpeg下载失败，请手动下载并放置在程序目录")
                        return False

            # 检查whisper
            try:
                import whisper
            except ImportError:
                self.log_signal.emit("正在安装whisper...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
                        "openai-whisper"
                    ])
                except Exception as e:
                    self.log_signal.emit(f"❌ whisper安装失败: {str(e)}")
                    return False

            return True

        except Exception as e:
            self.log_signal.emit(f"依赖检查失败: {str(e)}")
            return False

    def download_ffmpeg(self):
        """下载ffmpeg"""
        try:
            url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
            
            # 使用requests下载
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1KB
            
            with open("ffmpeg.zip", "wb") as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    progress = (downloaded / total_size) * 100
                    self.log_signal.emit(f"下载进度: {progress:.1f}%")
            
            self.log_signal.emit("正在解压ffmpeg...")
            with zipfile.ZipFile("ffmpeg.zip", "r") as zip_ref:
                zip_ref.extractall("ffmpeg_temp")
            
            # 移动ffmpeg.exe
            ffmpeg_exe = next(Path("ffmpeg_temp").rglob("ffmpeg.exe"))
            if os.path.exists("ffmpeg.exe"):
                os.remove("ffmpeg.exe")
            os.rename(ffmpeg_exe, "ffmpeg.exe")
            
            # 清理临时文件
            shutil.rmtree("ffmpeg_temp")
            os.remove("ffmpeg.zip")
            
            self.ffmpeg_path = "ffmpeg.exe"
            self.log_signal.emit("✓ ffmpeg 安装完成")
            return True
            
        except Exception as e:
            self.log_signal.emit(f"❌ ffmpeg 下载失败: {str(e)}")
            return False

class HelpButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("?", parent)
        self.setFixedSize(20, 20)
        self.setStyleSheet("""
            QPushButton {
                border: 1px solid #999;
                border-radius: 10px;
                background-color: #f0f0f0;
                color: #666;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.help_text = ("模型选择指南：\n"
                         "• ≥10GB 显存：large（适合最佳质量）\n"
                         "• ≥8GB 显存：medium（平衡速度和质量）\n"
                         "• ≥5GB 显存：small（平衡内存和质量）\n"
                         "• <5GB 显存：base（适合基本使用）\n"
                         "• CPU 模式：base（适合CPU模式）")
        
        # 设置工具提示样式
        QToolTip.setFont(QFont('Microsoft YaHei', 9))
        
        # 连接点击事件
        self.clicked.connect(self.show_tooltip)
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        
    def enterEvent(self, event):
        # 鼠标进入时显示提示
        QToolTip.showText(QCursor.pos(), self.help_text, self)
        
    def leaveEvent(self, event):
        # 鼠标离开时隐藏提示
        QToolTip.hideText()
        
    def show_tooltip(self):
        # 点击时显示提示
        pos = self.mapToGlobal(QPoint(self.width(), 0))
        QToolTip.showText(pos, self.help_text, self)


class ConfirmDialog(QDialog):
    def __init__(self, files, parent=None):
        super().__init__(parent)
        self.files = files
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("转换确认")
        self.setMinimumWidth(500)
        layout = QVBoxLayout()

        # 添加文件数量信息
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"待转换文件数量: {len(self.files)}"))
        
        # 估算总时间（假设每分钟视频需要20秒处理）
        total_duration = 0
        for file in self.files:
            try:
                # 使用ffprobe获取视频时长
                cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                      '-of', 'default=noprint_wrappers=1:nokey=1', file]
                result = subprocess.run(cmd, capture_output=True, text=True)
                duration = float(result.stdout.strip())
                total_duration += duration
            except:
                # 如果无法获取时长，假设是5分钟
                total_duration += 300

        # 估算处理时间（假设处理速度是实际时间的1/3）
        estimated_time = total_duration / 3
        hours = int(estimated_time // 3600)
        minutes = int((estimated_time % 3600) // 60)
        seconds = int(estimated_time % 60)
        
        time_label = QLabel(f"预计处理时间: {hours}小时{minutes}分钟{seconds}秒")
        info_layout.addWidget(time_label)
        layout.addLayout(info_layout)

        # 添加文件列表
        list_label = QLabel("文件列表:")
        layout.addWidget(list_label)

        # 创建文件列表显示区域（带滚动条）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # 添加每个文件的信息
        for i, file in enumerate(self.files, 1):
            file_name = Path(file).name
            file_size = os.path.getsize(file) / (1024 * 1024)  # 转换为MB
            file_label = QLabel(f"{i}. {file_name} ({file_size:.1f}MB)")
            file_label.setToolTip(file)  # 鼠标悬停显示完整路径
            content_layout.addWidget(file_label)

        content_layout.addStretch()
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # 添加按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("开始转换")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)


class VideoAudioExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_files = []
        self.output_folder = ""
        self.processor_thread = None
        self.ffmpeg_path = ""
        self.api_server = None
        self.init_ui()
        self.check_dependencies()

    def get_recommended_model(self):
        """获取推荐的模型大小"""
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if gpu_memory_gb >= 10:
                return "large", "适合最佳质量"
            elif gpu_memory_gb >= 8:
                return "medium", "平衡速度和质量"
            elif gpu_memory_gb >= 5:
                return "small", "平衡内存和质量"
            else:
                return "base", "适合基本使用"
        else:
            # CPU模式下推荐使用较小的模型
            return "base", "适合CPU模式"

    def init_ui(self):
        self.setWindowTitle("视频音频转文字工具 (GPU加速版)")
        self.setGeometry(100, 100, 900, 700)

        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 设置字体
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)

        # 标题
        title_label = QLabel("视频音频转文字提取工具 (GPU加速版)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        main_layout.addWidget(title_label)

        # 添加ffmpeg路径选择
        ffmpeg_layout = QHBoxLayout()
        self.ffmpeg_label = QLabel("未选择ffmpeg路径")
        self.ffmpeg_btn = QPushButton("选择ffmpeg")
        self.ffmpeg_btn.clicked.connect(self.select_ffmpeg_path)
        ffmpeg_layout.addWidget(QLabel("ffmpeg路径:"))
        ffmpeg_layout.addWidget(self.ffmpeg_label, 1)
        ffmpeg_layout.addWidget(self.ffmpeg_btn)
        main_layout.addLayout(ffmpeg_layout)

        # 模型设置部分
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Whisper模型:"))

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        # 设置推荐的模型大小
        recommended_model, reason = self.get_recommended_model()
        self.model_combo.setCurrentText(recommended_model)
        model_layout.addWidget(self.model_combo)

        # 添加帮助按钮
        help_btn = HelpButton(self)
        model_layout.addWidget(help_btn)

        # 添加推荐标记
        self.model_recommended_label = QLabel(f"（推荐：{reason}）")
        self.model_recommended_label.setStyleSheet("color: green;")
        model_layout.addWidget(self.model_recommended_label)

        self.gpu_checkbox = QCheckBox("使用GPU加速")
        self.gpu_checkbox.setChecked(torch.cuda.is_available())
        self.gpu_checkbox.setEnabled(torch.cuda.is_available())
        model_layout.addWidget(self.gpu_checkbox)

        model_layout.addStretch()
        main_layout.addLayout(model_layout)

        # 当模型选择改变时更新推荐标记
        self.model_combo.currentTextChanged.connect(self.update_model_recommendation)

        # 选择视频文件/文件夹部分
        video_layout = QHBoxLayout()
        self.video_label = QLabel("未选择视频文件")
        self.video_select_btn = QPushButton("选择视频")
        self.video_select_btn.clicked.connect(self.select_videos)
        
        video_layout.addWidget(QLabel("视频选择:"))
        video_layout.addWidget(self.video_label, 1)
        video_layout.addWidget(self.video_select_btn)
        main_layout.addLayout(video_layout)

        # 选择输出文件夹部分
        output_layout = QHBoxLayout()
        self.output_label = QLabel("未选择输出文件夹")
        self.output_btn = QPushButton("选择输出文件夹")
        self.output_btn.clicked.connect(self.select_output_folder)

        output_layout.addWidget(QLabel("输出文件夹:"))
        output_layout.addWidget(self.output_label, 1)
        output_layout.addWidget(self.output_btn)
        main_layout.addLayout(output_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始转换")
        self.stop_btn = QPushButton("停止转换")
        self.clear_log_btn = QPushButton("清空日志")

        self.start_btn.clicked.connect(self.start_conversion)
        self.stop_btn.clicked.connect(self.stop_conversion)
        self.clear_log_btn.clicked.connect(self.clear_log)

        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.clear_log_btn)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # 日志显示区域
        main_layout.addWidget(QLabel("处理日志:"))
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(350)
        main_layout.addWidget(self.log_text)

        # 状态提示
        gpu_status = "GPU可用" if torch.cuda.is_available() else "GPU不可用，将使用CPU"
        status_label = QLabel(f"状态: {gpu_status} | 依赖: ffmpeg, openai-whisper, torch")
        status_label.setStyleSheet("color: #666; font-size: 9px;")
        main_layout.addWidget(status_label)

        # API服务控制部分
        api_group = QGroupBox("API服务控制")
        api_layout = QVBoxLayout()

        # API服务控制按钮和状态显示
        api_controls = QHBoxLayout()
        self.api_start_btn = QPushButton("启动API服务")
        self.api_stop_btn = QPushButton("停止API服务")
        self.api_status_label = QLabel("服务状态: 未启动")
        
        self.api_start_btn.clicked.connect(self.start_api_service)
        self.api_stop_btn.clicked.connect(self.stop_api_service)
        self.api_stop_btn.setEnabled(False)

        api_controls.addWidget(self.api_start_btn)
        api_controls.addWidget(self.api_stop_btn)
        api_controls.addWidget(self.api_status_label)
        api_controls.addStretch()

        # API服务配置
        api_config = QHBoxLayout()
        self.api_host_input = QLineEdit("0.0.0.0")
        self.api_port_input = QLineEdit("8000")
        self.api_port_input.setValidator(QIntValidator(1, 65535))
        
        api_config.addWidget(QLabel("主机:"))
        api_config.addWidget(self.api_host_input)
        api_config.addWidget(QLabel("端口:"))
        api_config.addWidget(self.api_port_input)
        api_config.addStretch()

        # API服务统计信息
        self.api_stats = QLabel("任务统计: 总数 0 | 已完成 0")

        api_layout.addLayout(api_controls)
        api_layout.addLayout(api_config)
        api_layout.addWidget(self.api_stats)

        api_group.setLayout(api_layout)
        main_layout.addWidget(api_group)

        # GPU诊断按钮
        diag_layout = QHBoxLayout()
        self.gpu_diag_btn = QPushButton("GPU诊断")
        self.gpu_diag_btn.clicked.connect(self.show_gpu_diagnostic)
        diag_layout.addWidget(self.gpu_diag_btn)
        diag_layout.addStretch()
        main_layout.addLayout(diag_layout)

    def check_dependencies(self):
        """检查依赖项"""
        # 检查ffmpeg
        ffmpeg_found = False
        try:
            # 尝试多种方式查找ffmpeg
            result = subprocess.run(['ffmpeg', '-version'], 
                                 capture_output=True, 
                                 text=True, 
                                 encoding='utf-8',
                                 errors='ignore',
                                 shell=True)
            if result.returncode == 0:
                # 提取版本信息
                version_line = result.stdout.split('\n')[0]
                self.log_message(f"✓ {version_line}")
                # 获取ffmpeg路径
                where_result = subprocess.run('where ffmpeg', 
                                           capture_output=True, 
                                           text=True, 
                                           encoding='utf-8',
                                           errors='ignore',
                                           shell=True)
                if where_result.returncode == 0:
                    ffmpeg_path = where_result.stdout.strip().split('\n')[0]
                    self.ffmpeg_path = ffmpeg_path
                    self.ffmpeg_label.setText(f"已自动检测: {ffmpeg_path}")
                    self.ffmpeg_btn.setEnabled(False)  # 禁用选择按钮
                    ffmpeg_found = True
                else:
                    self.ffmpeg_path = 'ffmpeg'  # 使用命令名作为默认值
                    self.ffmpeg_label.setText("已在系统PATH中找到ffmpeg")
                    self.ffmpeg_btn.setEnabled(False)  # 禁用选择按钮
                    ffmpeg_found = True
            else:
                self.log_message("✗ ffmpeg 执行失败")
        except Exception as e:
            self.log_message(f"✗ ffmpeg 检测异常: {str(e)}")

        # 如果第一次检测失败，尝试其他路径
        if not ffmpeg_found:
            common_paths = [
                'ffmpeg.exe',
                r'C:\ffmpeg\bin\ffmpeg.exe',
                r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
            ]

            for path in common_paths:
                try:
                    result = subprocess.run([path, '-version'], 
                                         capture_output=True, 
                                         text=True,
                                         encoding='utf-8',
                                         errors='ignore')
                    if result.returncode == 0:
                        self.log_message(f"✓ ffmpeg 找到: {path}")
                        self.ffmpeg_path = path
                        self.ffmpeg_label.setText(f"已自动检测: {path}")
                        self.ffmpeg_btn.setEnabled(False)  # 禁用选择按钮
                        ffmpeg_found = True
                        break
                except:
                    continue

            if not ffmpeg_found:
                self.log_message("✗ ffmpeg 未找到")
                self.log_message("💡 请检查:")
                self.log_message("   1. ffmpeg是否正确安装")
                self.log_message("   2. PATH环境变量是否包含ffmpeg路径")
                self.log_message("   3. 重启程序或重启电脑")
                self.log_message("   4. 或者手动选择ffmpeg.exe")
                self.ffmpeg_btn.setEnabled(True)  # 启用选择按钮

        # 详细检查GPU状态
        self.log_message("=" * 40)
        self.log_message("GPU 检测报告:")

        # 检查PyTorch
        self.log_message(f"PyTorch版本: {torch.__version__}")

        # 检查CUDA
        if torch.cuda.is_available():
            self.log_message("✓ CUDA 可用")
            self.log_message(f"CUDA版本: {torch.version.cuda}")
            self.log_message(f"GPU数量: {torch.cuda.device_count()}")

            # 检查每个GPU
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                self.log_message(f"GPU {i}: {gpu_name} ({gpu_mem:.1f}GB)")

            # 获取推荐模型和原因
            recommended_model, reason = self.get_recommended_model()
            self.log_message(f"💡 推荐使用 {recommended_model} 模型（{reason}）")

            # 检查CUDA工具包
            try:
                import triton
                self.log_message("✓ Triton CUDA 工具包已安装")
            except ImportError:
                self.log_message("⚠️ Triton CUDA 工具包未安装")
                self.log_message("💡 建议运行: pip install triton")
                self.log_message("  这将启用额外的GPU加速功能")

        else:
            self.log_message("✗ CUDA 不可用")
            self.log_message("可能原因:")
            self.log_message("  1. 没有NVIDIA GPU")
            self.log_message("  2. 显卡驱动未安装")
            self.log_message("  3. PyTorch版本不支持CUDA")
            self.log_message("  4. CUDA工具包未安装")

            # 检查是否有其他GPU
            try:
                import platform
                if platform.system() == "Darwin":  # macOS
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        self.log_message("✓ 检测到 Apple Silicon GPU (MPS)")
                        self.log_message("注意: Whisper暂不支持MPS，将使用CPU")
            except:
                pass

            # 获取推荐模型和原因
            recommended_model, reason = self.get_recommended_model()
            self.log_message(f"💡 推荐使用 {recommended_model} 模型（{reason}）")

    def select_videos(self):
        """选择视频文件和文件夹"""
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles)  # 允许选择多个文件
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # 使用Qt对话框以支持文件夹选择
        dialog.setNameFilter("视频文件 (*.mp4 *.avi *.mov *.wmv *.flv *.mkv *.webm *.m4v *.3gp);;所有文件 (*)")
        
        # 添加文件夹选择按钮
        tree_view = dialog.findChild(QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        list_view = dialog.findChild(QListView)
        if list_view:
            list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # 添加"选择文件夹"按钮
        folder_btn = QPushButton("选择文件夹", dialog)
        dialog.layout().addWidget(folder_btn)
        
        self.video_files = []  # 清空之前的选择
        
        def handle_folder_selection():
            folder = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
            if folder:
                # 扫描文件夹中的视频文件
                video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v', '.3gp']
                for root, dirs, files in os.walk(folder):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in video_extensions):
                            self.video_files.append(os.path.join(root, file))
                dialog.accept()  # 关闭对话框
        
        folder_btn.clicked.connect(handle_folder_selection)
        
        if dialog.exec_() == QFileDialog.Accepted:
            # 获取选择的文件
            selected_files = dialog.selectedFiles()
            for file in selected_files:
                if os.path.isfile(file):  # 确保是文件而不是目录
                    self.video_files.append(file)
            
            # 更新界面显示
            if self.video_files:
                if len(self.video_files) == 1:
                    self.video_label.setText(f"已选择: {Path(self.video_files[0]).name}")
                else:
                    self.video_label.setText(f"已选择 {len(self.video_files)} 个视频文件")
                self.log_message(f"共选择了 {len(self.video_files)} 个视频文件")
                
                # 显示所有选择的文件路径
                self.log_message("选择的文件:")
                for file in self.video_files:
                    self.log_message(f"  • {file}")
            else:
                self.video_label.setText("未选择视频文件")

    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")

        if folder:
            self.output_folder = folder
            self.output_label.setText(f"输出到: {Path(folder).name}")
            self.log_message(f"设置输出文件夹: {folder}")

    def select_ffmpeg_path(self):
        """选择ffmpeg可执行文件路径"""
        file_filter = "ffmpeg (ffmpeg.exe);;所有文件 (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择ffmpeg可执行文件",
            "",
            file_filter
        )

        if file_path:
            # 验证选择的文件是否是ffmpeg
            try:
                result = subprocess.run([file_path, '-version'], 
                                     capture_output=True, 
                                     text=True, 
                                     timeout=5)
                if result.returncode == 0 and 'ffmpeg version' in result.stdout:
                    self.ffmpeg_path = file_path
                    self.ffmpeg_label.setText(f"已选择: {Path(file_path).name}")
                    self.log_message(f"设置ffmpeg路径: {file_path}")
                else:
                    QMessageBox.warning(self, "警告", "所选文件不是有效的ffmpeg可执行文件")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"验证ffmpeg失败: {str(e)}")

    def start_conversion(self):
        """开始转换"""
        if not self.video_files:
            QMessageBox.warning(self, "警告", "请先选择视频文件或文件夹")
            return

        if not self.output_folder:
            QMessageBox.warning(self, "警告", "请先选择输出文件夹")
            return

        # 显示确认对话框
        dialog = ConfirmDialog(self.video_files, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        # 禁用开始按钮，启用停止按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 获取设置
        model_size = self.model_combo.currentText()
        use_gpu = self.gpu_checkbox.isChecked()

        # 创建并启动处理线程
        self.processor_thread = VideoProcessor(
            self.video_files,
            self.output_folder,
            model_size,
            use_gpu,
            self.ffmpeg_path
        )
        self.processor_thread.log_signal.connect(self.log_message)
        self.processor_thread.progress_signal.connect(self.update_progress)
        self.processor_thread.finished_signal.connect(self.conversion_finished)
        self.processor_thread.start()

    def stop_conversion(self):
        """停止转换"""
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
            self.log_message("正在停止转换...")

    def conversion_finished(self):
        """转换完成"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.log_message("转换任务结束")

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    def clear_log(self):
        """清空日志"""
        self.log_text.clear()

    def show_gpu_diagnostic(self):
        """显示GPU详细诊断"""
        self.log_message("\n🔍 开始完整环境诊断...")

        # 系统信息
        import platform
        self.log_message(f"操作系统: {platform.system()} {platform.release()}")
        self.log_message(f"Python版本: {platform.python_version()}")

        # 环境变量检查
        self.log_message("\n📁 环境变量检查:")
        path_env = os.environ.get('PATH', '')
        ffmpeg_in_path = any('ffmpeg' in p.lower() for p in path_env.split(os.pathsep))
        self.log_message(f"PATH中包含ffmpeg: {'✅' if ffmpeg_in_path else '❌'}")

        # 手动检查ffmpeg
        self.log_message("\n🎬 FFmpeg详细检查:")
        try:
            # 使用where命令查找ffmpeg
            result = subprocess.run('where ffmpeg', shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                ffmpeg_paths = result.stdout.strip().split('\n')
                for path in ffmpeg_paths:
                    if path.strip():
                        self.log_message(f"找到ffmpeg: {path.strip()}")

                        # 测试这个ffmpeg
                        try:
                            test_result = subprocess.run([path.strip(), '-version'],
                                                         capture_output=True, text=True, timeout=5)
                            if test_result.returncode == 0:
                                version_info = test_result.stdout.split('\n')[0]
                                self.log_message(f"✅ {version_info}")
                            else:
                                self.log_message(f"❌ 该ffmpeg无法正常运行")
                        except Exception as e:
                            self.log_message(f"❌ 测试失败: {e}")
            else:
                self.log_message("❌ 系统找不到ffmpeg命令")
        except Exception as e:
            self.log_message(f"❌ where命令执行失败: {e}")

        # PyTorch信息
        self.log_message(f"\n🔥 PyTorch信息:")
        self.log_message(f"PyTorch版本: {torch.__version__}")
        self.log_message(f"PyTorch编译CUDA版本: {torch.version.cuda}")

        # CUDA详细检测
        self.log_message(f"\n⚡ CUDA检测:")
        if torch.cuda.is_available():
            self.log_message("✅ CUDA 完全可用")
            self.log_message(f"CUDA运行时版本: {torch.version.cuda}")
            self.log_message(f"CUDA设备数量: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.log_message(f"GPU {i}: {props.name}")
                self.log_message(f"  显存: {props.total_memory / 1024 ** 3:.1f} GB")
                self.log_message(f"  计算能力: {props.major}.{props.minor}")

                # 测试GPU
                try:
                    test_tensor = torch.randn(100, 100).cuda(i)
                    result = torch.matmul(test_tensor, test_tensor)
                    self.log_message(f"  ✅ GPU {i} 测试通过")
                except Exception as e:
                    self.log_message(f"  ❌ GPU {i} 测试失败: {e}")
        else:
            self.log_message("❌ CUDA 不可用")

            # 检查NVIDIA驱动
            self.log_message("\n🔍 NVIDIA驱动检查:")
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    self.log_message("✅ NVIDIA驱动已安装")
                    # 提取GPU信息
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'NVIDIA-SMI' in line:
                            self.log_message(f"驱动版本: {line}")
                        elif 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                            self.log_message(f"GPU: {line.strip()}")

                    self.log_message("💡 NVIDIA驱动正常，问题可能是PyTorch版本")
                    self.log_message("💡 请重新安装CUDA版本的PyTorch:")
                    self.log_message("   pip uninstall torch torchvision torchaudio")
                    self.log_message(
                        "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                else:
                    self.log_message("❌ nvidia-smi 执行失败")
            except Exception as e:
                self.log_message(f"❌ nvidia-smi 命令不存在: {e}")
                self.log_message("💡 请先安装NVIDIA显卡驱动")

        # Whisper检查
        self.log_message(f"\n🎙️ Whisper检查:")
        try:
            import whisper
            self.log_message("✅ Whisper 已安装")

            # 显示推荐配置
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                if gpu_memory_gb >= 10:
                    self.log_message("💡 推荐使用 large 模型获得最佳效果")
                elif gpu_memory_gb >= 5:
                    self.log_message("💡 推荐使用 medium 模型平衡速度和质量")
                else:
                    self.log_message("💡 推荐使用 base 模型")
            else:
                self.log_message("💡 CPU模式推荐使用 tiny 或 base 模型")

        except ImportError:
            self.log_message("❌ Whisper 未安装")
            self.log_message("💡 请运行: pip install openai-whisper")

        # 解决方案汇总
        self.log_message(f"\n🛠️ 问题解决方案:")

        if not ffmpeg_in_path:
            self.log_message("FFmpeg问题:")
            self.log_message("1. 确认ffmpeg已下载并解压")
            self.log_message("2. 将ffmpeg/bin目录添加到系统PATH")
            self.log_message("3. 重启命令提示符和程序")
            self.log_message("4. 或将ffmpeg.exe复制到程序目录")

        if not torch.cuda.is_available():
            self.log_message("CUDA问题:")
            self.log_message("1. 安装最新NVIDIA显卡驱动")
            self.log_message("2. 重新安装支持CUDA的PyTorch:")
            self.log_message("   pip uninstall torch torchvision torchaudio")
            self.log_message(
                "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            self.log_message("3. 重启程序验证")

        self.log_message("=" * 60)

    def update_model_recommendation(self, current_model):
        """更新模型推荐标记"""
        recommended_model, reason = self.get_recommended_model()
        if current_model == recommended_model:
            self.model_recommended_label.setText(f"（推荐：{reason}）")
            self.model_recommended_label.setStyleSheet("color: green;")
        elif self.is_model_too_large(current_model):
            self.model_recommended_label.setText("（警告：可能内存不足）")
            self.model_recommended_label.setStyleSheet("color: red;")
        else:
            self.model_recommended_label.setText("")

    def is_model_too_large(self, model_name):
        """检查选择的模型是否可能超出系统资源"""
        if not torch.cuda.is_available():
            # CPU模式下，large和medium模型可能太大
            return model_name in ["large", "medium"]
        else:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if gpu_memory_gb < 4 and model_name in ["large", "medium", "small"]:
                return True
            elif gpu_memory_gb < 6 and model_name in ["large", "medium"]:
                return True
            elif gpu_memory_gb < 8 and model_name == "large":
                return True
            return False

    def update_api_status(self, status_data):
        """更新API服务状态显示"""
        status = status_data["status"]
        error = status_data["error"]
        task_count = status_data["task_count"]
        completed_tasks = status_data["completed_tasks"]

        # 更新状态标签
        status_text = f"服务状态: {status}"
        if error:
            status_text += f" (错误: {error})"
        self.api_status_label.setText(status_text)

        # 更新按钮状态
        self.api_start_btn.setEnabled(status != "running")
        self.api_stop_btn.setEnabled(status == "running")

        # 更新统计信息
        self.api_stats.setText(f"任务统计: 总数 {task_count} | 已完成 {completed_tasks}")

        # 根据状态设置标签颜色
        if status == "running":
            self.api_status_label.setStyleSheet("color: green")
        elif status == "error":
            self.api_status_label.setStyleSheet("color: red")
        else:
            self.api_status_label.setStyleSheet("")

    def start_api_service(self):
        """启动API服务"""
        try:
            from api_service import APIServer, register_status_callback
            
            host = self.api_host_input.text()
            port = int(self.api_port_input.text())

            if self.api_server is None:
                self.api_server = APIServer(host=host, port=port)
                register_status_callback(self.update_api_status)

            if self.api_server.start():
                self.log_message(f"API服务启动成功 - {host}:{port}")
                self.log_message("API接口:")
                self.log_message(f"  POST http://{host}:{port}/api/v1/transcribe")
                self.log_message(f"  GET  http://{host}:{port}/api/v1/tasks/{{task_id}}")
                self.log_message(f"  GET  http://{host}:{port}/api/v1/health")
            else:
                self.log_message("API服务已在运行")

        except Exception as e:
            self.log_message(f"API服务启动失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"API服务启动失败: {str(e)}")

    def stop_api_service(self):
        """停止API服务"""
        if self.api_server and self.api_server.stop():
            self.log_message("API服务已停止")
        else:
            self.log_message("API服务未在运行")

    def closeEvent(self, event):
        """窗口关闭时的处理"""
        # 停止视频处理
        if self.processor_thread and self.processor_thread.isRunning():
            self.stop_conversion()
        
        # 停止API服务
        if self.api_server:
            self.api_server.stop()
        
        event.accept()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='视频转文字工具 - GUI/API模式')
    parser.add_argument('--mode', choices=['gui', 'api'], default='gui',
                      help='运行模式: gui=图形界面模式, api=API服务模式')
    parser.add_argument('--host', default='0.0.0.0',
                      help='API服务主机地址 (仅在api模式下有效)')
    parser.add_argument('--port', type=int, default=8000,
                      help='API服务端口 (仅在api模式下有效)')
    args = parser.parse_args()

    if args.mode == 'api':
        print(f"启动API服务模式 - 监听地址: {args.host}:{args.port}")
        start_api_server(host=args.host, port=args.port)
    else:
        # GUI模式
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        # 检查依赖
        if not check_and_install_dependencies():
            QMessageBox.critical(None, "错误", "依赖检查失败，请查看控制台输出")
            return

        window = VideoAudioExtractorApp()
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        input("按回车键退出...")