import os
import sys
import shutil
import subprocess
from pathlib import Path

def clean_build_folders():
    """清理构建文件夹"""
    folders = ['build', 'dist']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    spec_file = 'videoToText.spec'
    if os.path.exists(spec_file):
        os.remove(spec_file)

def build_executable():
    """构建可执行文件"""
    print("开始构建可执行文件...")
    
    # PyInstaller命令
    cmd = [
        'pyinstaller',
        '--noconfirm',
        '--onefile',
        '--windowed',
        '--icon=app.ico',  # 如果有图标的话
        '--add-data=LICENSE;.',  # 如果有LICENSE文件的话
        '--name=VideoToText',
        '--hidden-import=torch',
        '--hidden-import=whisper',
        '--hidden-import=numpy',
        '--hidden-import=PyQt5',
        '--hidden-import=requests',
        '--hidden-import=pillow',
        'videoToText.py'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("构建成功！")
        print(f"可执行文件位置: {os.path.abspath('dist/VideoToText.exe')}")
    except subprocess.CalledProcessError as e:
        print(f"构建失败: {e}")
        sys.exit(1)

def copy_additional_files():
    """复制额外需要的文件到dist目录"""
    dist_dir = Path('dist')
    if not dist_dir.exists():
        return
    
    # 复制README文件（如果存在）
    if os.path.exists('README.md'):
        shutil.copy2('README.md', dist_dir / 'README.md')
    
    # 复制ffmpeg（如果存在）
    if os.path.exists('ffmpeg.exe'):
        shutil.copy2('ffmpeg.exe', dist_dir / 'ffmpeg.exe')

def create_installer():
    """创建安装包（可选）"""
    # 这里可以添加创建安装包的代码
    # 例如使用NSIS或Inno Setup
    pass

def main():
    # 清理旧的构建文件
    clean_build_folders()
    
    # 构建可执行文件
    build_executable()
    
    # 复制额外文件
    copy_additional_files()
    
    print("\n构建完成！")
    print("请确保在dist目录中包含以下文件：")
    print("1. VideoToText.exe")
    print("2. ffmpeg.exe（如果需要）")
    print("3. README.md（如果存在）")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main() 