from PIL import Image
import os

def convert_to_grayscale(input_dir, output_dir, start=1, end=100):
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历指定的图片文件范围
    for i in range(start, end + 1):
        file_name = f"{i}.jpg"
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        # 读取图片
        try:
            image = Image.open(input_path)
            # 转换为灰度图
            gray_image = image.convert('L')
            # 保存到输出目录
            gray_image.save(output_path)
            print(f"Converted and saved {file_name} to {output_dir}")
        except IOError as e:
            print(f"Could not open or find the file {file_name}. Error: {e}")

# 输入目录
input_dir = r'C:\Users\ZYY\Desktop\CFNet\CFNet-master\input\dataset_test\LLVIP\IR'
# 输出目录
output_dir = r'C:\Users\ZYY\Desktop\CFNet\CFNet-master\input\dataset_test\LLVIP\IR-8'

convert_to_grayscale(input_dir, output_dir)