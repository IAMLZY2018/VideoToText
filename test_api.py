import requests
import time
import os


# 这里只是一个api接口测试文档
def test_api():
    # API服务地址
    BASE_URL = "http://localhost:8000"
    
    # 1. 检查服务健康状态
    print("1. 检查服务健康状态...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"健康检查结果: {response.json()}")
    except Exception as e:
        print(f"错误: {e}")
        print("请确保API服务已启动")
        return

    # 2. 上传视频文件
    print("\n2. 上传视频文件...")
    video_files = [f for f in os.listdir(".") if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
    
    if not video_files:
        print("当前目录下没有找到视频文件")
        return
    
    video_file = video_files[0]
    print(f"使用视频文件: {video_file}")
    
    try:
        with open(video_file, "rb") as f:
            files = {"file": (video_file, f, "video/mp4")}
            response = requests.post(f"{BASE_URL}/api/v1/transcribe", files=files)
            task_data = response.json()
            task_id = task_data["task_id"]
            print(f"任务ID: {task_id}")
    except Exception as e:
        print(f"上传失败: {e}")
        return

    # 3. 轮询任务状态
    print("\n3. 等待任务完成...")
    while True:
        try:
            response = requests.get(f"{BASE_URL}/api/v1/tasks/{task_id}")
            status_data = response.json()
            print(f"任务状态: {status_data['status']}")
            
            if status_data["status"] == "completed":
                print("\n转写结果:")
                print("-" * 50)
                print(status_data["text"])
                print("-" * 50)
                print(f"输出文件: {status_data['file_path']}")
                print(f"处理时间: {status_data['duration']:.2f}秒")
                break
            elif status_data["status"] == "failed":
                print(f"任务失败: {status_data['error']}")
                break
            
            time.sleep(2)  # 每2秒检查一次
            
        except Exception as e:
            print(f"查询状态失败: {e}")
            break

if __name__ == "__main__":
    test_api() 