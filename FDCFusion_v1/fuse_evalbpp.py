import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM
import warnings
import utils
import torch
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())



def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def pad(x, p):
    h, w = x.size(2), x.size(3)     #获取原图像的高和宽
        #计算新图像参数
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    #对图像进行填充
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    #最后返回一个填充后图像的张量和一个元组
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

def parse_args(argv):
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser(description="Example testing script.")

    # 添加一个参数，是否使用cuda，默认值为True
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    # 添加一个参数，最大梯度裁剪norm，默认值为1.0    梯度裁剪是深度学习中的一种技术，通常用于防止梯度爆炸。当梯度的范数超过指定的阈值时，梯度裁剪会对其进行缩放，使其保持在这个阈值以内。这有助于提高模型训练的稳定性。
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    # 添加一个参数，检查点路径，默认值为'save_path/1checkpoint_latest.pth.tar'
    parser.add_argument("--checkpoint", type=str, default="save_path/1checkpoint_latest.pth.tar", help="Path to a checkpoint")
    # 添加两个参数，数据集路径，默认值分别为'input\MSRS361\vi'和'input\MSRS361\ir'
    parser.add_argument("--data1", type=str, default=r"input\MSRS361\vi", help="Path to dataset")
    parser.add_argument("--data2", type=str, default=r"input\MSRS361\ir", help="Path to dataset")
    # 添加一个参数，输出路径，默认值为'MSRS361'
    parser.add_argument("--output_path", type=str, default="fusion\zyy305", help="Path to output")
    # 添加一个参数，real，默认值为True
    parser.add_argument(
        "--real", action="store_true", default=True
    )

    # 设置默认值为True
    parser.set_defaults(real=True)
    # 解析参数
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    p = 256     #图像填充参数
    path1 = args.data1
    path2 = args.data2

    img_list1 = []
    for file in os.listdir(path1):      #遍历可见光图像到列表中
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list1.append(file)
    img_list2 = []
    for file in os.listdir(path2):      #遍历红外光图像到列表中
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list2.append(file)
    if args.cuda:
    #     device = 'cuda:0'
    # else:
        device = 'cpu'
        #导入cfnet网络模型
    net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=128, M=320)
    net = net.to(device)    #网络模型从CPU转移到GPU上
    net.eval()      #用于将模型从训练模式切换到评估模式。模型的参数不会更新，而是根据输入数据进行前向传播，计算输出结果。

    count = 0
    Bit_rate = 0
    MS_SSIM = 0
    total_time = 0
    dictory = {}

    if not os.path.exists(args.output_path):  # 如果输出路径不存在，则创建
        os.makedirs(args.output_path)

    if args.checkpoint:  # 用于从检查点文件中加载并恢复神经网络模型的状态，
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)    #用于将字典中的参数加载到神经网络模型中。

    net.update()    #用于更新神经网络模型的参数。

    for img_name1,img_name2 in zip(img_list1,img_list2):        #遍历可见光图像和红外光图像
        # 将图像路径拼接
        img_path1 = os.path.join(path1, img_name1)
        img_path2 = os.path.join(path2, img_name2)
        # 将图像转换为张量并移到GPU上
        img_1 = transforms.ToTensor()(Image.open(img_path1)).to(device)
        img_2 = transforms.ToTensor()(Image.open(img_path2)).to(device)
        #增加额外的一维张量
        x1 = img_1.unsqueeze(0)
        x2 = img_2.unsqueeze(0)
        # 填充图像
        x_padded_1, padding_1 = pad(x1, p)
        x_padded_2, padding_2 = pad(x2, p)

        count += 1
        with torch.no_grad():       #不需要计算梯度的情况下，使用PyTorch张量进行计算。（正向传播）
            if args.cuda:
                torch.cuda.synchronize()

            s = time.time()     #计时开始融合图像

            out_enc = net.compress(x_padded_1,x_padded_2)       #将图像压缩
            out_dec = net.decompress(out_enc["strings"], out_enc["shape"])      #将图像解压

            if args.cuda:
                torch.cuda.synchronize()

            e = time.time()
            total_time += (e - s)       # 计算时间差

            out_dec["x_hat"] = crop(out_dec["x_hat"], padding_1)        # 图像裁剪
            imageout_path = os.path.join(args.output_path, img_name1)

            utils.tensor_save_rgbimage(out_dec["x_hat"], imageout_path)     # 将图像保存到指定路径


            num_pixels = x1.size(0) * x1.size(2) * x1.size(3)       # 计算图像的像素数

            print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp') # 计算比特率
            Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels


    #计算平均MS-SSIM，平均比特率和平均时间
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    total_time = total_time / count
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} s')


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])          #表示从第二个元素（即索引为 1 的位置）开始的所有命令行参数，这样就排除了脚本名称，只保留了实际传递给脚本的参数。
