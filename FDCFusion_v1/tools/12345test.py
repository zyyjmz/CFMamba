from compressai.layers import (
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers import GDN,conv1x1

from torch import Tensor
import pywt
import torch
import torch.nn as nn
from torch.autograd import Function


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        #self.filters = self.filters.to(dtype=torch.float16)
        self.filters = self.filters.to(dtype=torch.float32)
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        #self.w_ll = self.w_ll.to(dtype=torch.float16)
        #self.w_lh = self.w_lh.to(dtype=torch.float16)
        #self.w_hl = self.w_hl.to(dtype=torch.float16)
        #self.w_hh = self.w_hh.to(dtype=torch.float16)

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class ResidualBlockWithStride_wave(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, wavelet='haar'):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn_low = GDN(out_ch)
        self.gdn_high = GDN(3*out_ch)
        self.dwt = DWT_2D(wave=wavelet)
        self.idwt = IDWT_2D(wave=wavelet)

        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.high_freq_conv = conv3x3(3*out_ch, 3*out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)

        # DWT
        dwt_output = self.dwt(out)

        # Separate low-frequency and high-frequency components
        low_freq = dwt_output[:, :out.size(1), :, :]
        high_freq = dwt_output[:, out.size(1):, :, :]

        # Process low-frequency and high-frequency components separately
        low_freq_processed = self.low_freq_conv(low_freq)
        low_freq_processed = self.gdn_low(low_freq_processed)
        high_freq_processed = self.high_freq_conv(high_freq)
        high_freq_processed = self.gdn_high(high_freq_processed)

        # Reassemble the processed components
        dwt_processed = torch.cat([low_freq_processed, high_freq_processed], dim=1)

        # IDWT
        output = self.idwt(dwt_processed)


        if self.skip is not None:
            identity = self.skip(x)

        output += identity
        return output


class ResidualBlockUpsample_wave(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2, wavelet='haar'):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        # self.conv = conv3x3(out_ch, out_ch)
        self.igdn_low = GDN(out_ch, inverse=True)
        self.igdn_high = GDN(3 * out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.dwt = DWT_2D(wave=wavelet)
        self.idwt = IDWT_2D(wave=wavelet)

        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.high_freq_conv = conv3x3(3 * out_ch, 3 * out_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        # out = self.conv(out)
        # out = self.igdn(out)
        # DWT
        dwt_output = self.dwt(out)

        # Separate low-frequency and high-frequency components
        low_freq = dwt_output[:, :out.size(1), :, :]
        high_freq = dwt_output[:, out.size(1):, :, :]

        # Process low-frequency and high-frequency components separately
        low_freq_processed = self.low_freq_conv(low_freq)
        low_freq_processed = self.igdn_low(low_freq_processed)
        high_freq_processed = self.high_freq_conv(high_freq)
        high_freq_processed = self.igdn_high(high_freq_processed)

        # Reassemble the processed components
        dwt_processed = torch.cat([low_freq_processed, high_freq_processed], dim=1)

        # IDWT
        output = self.idwt(dwt_processed)

        identity = self.upsample(x)
        output += identity
        return output


if __name__ == '__main__':
    input = torch.randn(8, 320, 16, 16)  # 随机生成一个输入特征图
    haar = ResidualBlockWithStride_wave(
                in_ch = 320, out_ch = 128, stride = 2, wavelet='haar'
                        )
    output = haar(input)  # 将输入特征图通过SAL模块进行处理
    print(output.shape)  # 打印处理后的特征图形状，验证SAL模块的作用


