#频域模块测试   ECCV2024-AdpatICMH-main


import torch
import torch.nn as nn


class SFMA(nn.Module):
    def __init__(self, in_dim=320, middle_dim=64, adapt_factor=1):
        super().__init__()
        self.factor = adapt_factor
        self.s_down1 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_down2 = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.s_dw = nn.Conv2d(middle_dim, middle_dim, 5, 1, 2, groups=middle_dim)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)

        self.f_down = nn.Conv2d(in_dim, middle_dim, 1, 1, 0)
        self.f_relu1 = nn.ReLU(inplace=True)
        self.f_relu2 = nn.ReLU(inplace=True)
        self.f_up = nn.Conv2d(middle_dim, in_dim, 1, 1, 0)
        self.f_dw = nn.Conv2d(middle_dim, middle_dim, 3, 1, 1, groups=middle_dim)
        self.f_inter = nn.Conv2d(middle_dim, middle_dim, 1, 1, 0)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        '''
        input:
        x: intermediate feature
        output:
        x_tilde: adapted feature
        '''
        _, _, H, W = x.shape

        y = torch.fft.rfft2(self.f_down(x), dim=(2, 3), norm='backward')
        y_amp = torch.abs(y)
        y_phs = torch.angle(y)
        # we only modulate the amplitude component for better training stability
        y_amp_modulation = self.f_inter(self.f_relu1(self.f_dw(y_amp)))
        y_amp = y_amp * self.sg(y_amp_modulation)
        y_real = y_amp * torch.cos(y_phs)
        y_img = y_amp * torch.sin(y_phs)
        y = torch.complex(y_real, y_img)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')

        f_modulate = self.f_up(self.f_relu2(y))
        s_modulate = self.s_up(self.s_relu(self.s_dw(self.s_down1(x)) * self.s_down2(x)))
        x_tilde = x + (s_modulate + f_modulate) * self.factor
        return x_tilde

if __name__ == '__main__':
    input = torch.randn(8, 320, 16, 16)  # 随机生成一个输入特征图
    sfma = SFMA(
                in_dim=320, middle_dim=64, adapt_factor=1
                        )
    output = sfma(input)  # 将输入特征图通过SAL模块进行处理
    print(output.shape)  # 打印处理后的特征图形状，验证SAL模块的作用

