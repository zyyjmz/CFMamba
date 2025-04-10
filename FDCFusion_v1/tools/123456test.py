import torch

# 加载检查点文件
checkpoint = torch.load(r"C:\Users\ZYY\Desktop\CFNet-master2\save_path\1checkpoint_latest.pth.tar")

# 打印检查点内容
print(checkpoint.keys())

print(f"Epoch: {checkpoint['epoch']}")
