from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    # 创建一个512x512的图像
    size = 512
    image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # 绘制圆形背景
    margin = 10
    draw.ellipse([margin, margin, size-margin, size-margin], 
                 fill='#2196F3')  # Material Design Blue
    
    # 绘制文字
    try:
        # 尝试加载微软雅黑字体
        font = ImageFont.truetype("msyh.ttc", 280)
    except:
        # 如果没有找到，使用默认字体
        font = ImageFont.load_default()
    
    # 添加文字
    text = "V"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 30  # 稍微向上偏移
    
    # 绘制白色文字
    draw.text((x, y), text, fill='white', font=font)
    
    # 保存为ICO文件
    image.save("app.ico", format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])

if __name__ == "__main__":
    create_icon() 