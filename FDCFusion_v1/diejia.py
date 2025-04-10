from PIL import Image
import os

def convert_images_to_rgb(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有png图像
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, 'RGB_' + file_name)

            # 打开图像
            img = Image.open(input_path)

            # 转换图像为RGB
            rgb_img = img.convert('RGB')

            # 保存转换后的图像
            rgb_img.save(output_path)
            print(f'Converted {file_name} to RGB and saved to {output_path}')

# 输入和输出文件夹的路径
input_folder = 'C:\\Users\\ZYY\\Desktop\\CFNet\\CFNet-master\\input\\dataset_test\\6_Harvard\\FDG'
output_folder = 'C:\\Users\\ZYY\\Desktop\\CFNet\\CFNet-master\\input\\dataset_test\\6_Harvard\\FDG_RGB'

# 调用函数
convert_images_to_rgb(input_folder, output_folder)
