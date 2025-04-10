import os

def rename_images(directory):
    # 获取指定目录中的所有文件
    files = os.listdir(directory)
    # 过滤出png文件
    png_files = [f for f in files if f.endswith('.png')]

    # 按照原文件名排序
    png_files.sort()

    # 遍历所有png文件并重命名
    for i, filename in enumerate(png_files, start=1):
        # 生成新的文件名，如 "1.png", "2.png", "3.png"
        new_name = f"{i}.png"
        # 获取旧文件路径和新文件路径
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        # 重命名文件
        os.rename(old_path, new_path)

    print("重命名完成")


if __name__ == '__main__':
# 调用函数，并传入图片所在的目录路径
    #rename_images('C:/Users/ZYY/Desktop/MSRS-main/MSRS-main/test/ir')
    #rename_images('C:/Users/ZYY/Desktop/MSRS-main/MSRS-main/test/vi')
    rename_images('C:/Users/ZYY/Desktop/CFNet/CFNet-master/MSRS361')