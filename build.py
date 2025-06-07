import os
import shutil
import subprocess
from pathlib import Path

def clean_build():
    """清理build和dist目录"""
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    for file in Path('.').glob('*.spec'):
        if file.name != 'videoToText.spec':
            file.unlink()

def download_ffmpeg():
    """下载ffmpeg"""
    import requests
    print("正在下载ffmpeg...")
    url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    r = requests.get(url)
    with open("ffmpeg.zip", "wb") as f:
        f.write(r.content)
    
    # 解压
    import zipfile
    with zipfile.ZipFile("ffmpeg.zip", "r") as zip_ref:
        zip_ref.extractall("ffmpeg_temp")
    
    # 移动ffmpeg.exe
    ffmpeg_path = next(Path("ffmpeg_temp").rglob("ffmpeg.exe"))
    shutil.copy(ffmpeg_path, "dist/ffmpeg.exe")
    
    # 清理
    os.remove("ffmpeg.zip")
    shutil.rmtree("ffmpeg_temp")

def main():
    # 清理旧的构建文件
    clean_build()
    
    # 生成图标
    print("正在生成图标...")
    subprocess.run(["python", "create_icon.py"], check=True)
    
    # 打包程序
    print("正在打包程序...")
    subprocess.run(["pyinstaller", "--clean", "videoToText.spec"], check=True)
    
    # 下载ffmpeg
    download_ffmpeg()
    
    # 复制README
    shutil.copy("README.txt", "dist/使用说明.txt")
    
    # 创建发布zip
    print("正在创建发布包...")
    shutil.make_archive("视频转文字工具", "zip", "dist")
    
    print("打包完成！")
    print("发布文件: 视频转文字工具.zip")

if __name__ == "__main__":
    main() 