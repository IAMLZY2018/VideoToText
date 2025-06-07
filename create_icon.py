from PIL import Image, ImageDraw

# 创建一个200x200的图像，背景为白色
img = Image.new('RGB', (200, 200), 'white')
draw = ImageDraw.Draw(img)

# 绘制一个简单的图标设计
# 外圈
draw.ellipse([20, 20, 180, 180], outline='#2196F3', width=8)

# 内部的播放按钮形状
draw.polygon([(70, 60), (70, 140), (140, 100)], fill='#2196F3')

# 保存为ICO格式
img.save('app.ico', format='ICO') 