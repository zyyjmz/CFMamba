#上下采样程序测试    SegPIC-for-Image-Compression-main

from compressai.layers import GDN,conv1x1
import torch.nn as nn
import torch
from torchsummary import summary

class Downblock_SAL(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, groups, pool_stride, num_heads):
        super().__init__()
        self.conv_groups = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups
            ),
            GDN(out_c),
        )
        self.scales_bias = nn.Sequential(
            conv1x1(out_c, out_c),
            nn.GELU(),
            conv1x1(out_c, out_c * 2),
        )

    def forward(self, x):
        x = self.conv_groups(x)
        x_q = self.scales_bias(x)
        ch = x_q.shape[1]
        x_scales = x_q[:, :ch // 2, :, :]
        x_bias = x_q[:, ch // 2:, :, :]
        x = x * (1 + x_scales) + x_bias
        return x


class Upblock_SAL(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, groups, num_heads, inverse=False):
        super().__init__()
        self.conv_groups = nn.Sequential(
            nn.ConvTranspose2d(
                in_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                output_padding=stride - 1,
                padding=kernel_size // 2,
                groups=groups,
            ),
            GDN(out_c, inverse=inverse),
        )
        self.scales_bias = nn.Sequential(
            conv1x1(in_c, in_c),
            nn.GELU(),
            conv1x1(in_c, in_c * 2),
        )

    def forward(self, x):
        x_q = self.scales_bias(x)
        ch = x_q.shape[1]
        x_scales = x_q[:, :ch // 2, :, :]
        x_bias = x_q[:, ch // 2:, :, :]
        x = x * (1 + x_scales) + x_bias
        x = self.conv_groups(x)
        return x


if __name__ == '__main__':
    input = torch.randn(8, 4, 256, 256)  # 随机生成一个输入特征图
    SAL = Downblock_SAL(
                in_c=4, out_c=192, kernel_size=5, stride=2, groups=1,
                pool_stride=32, num_heads=3
                        )
    output = SAL(input)  # 将输入特征图通过SAL模块进行处理
    print(output.shape)  # 打印处理后的特征图形状，验证SAL模块的作用


