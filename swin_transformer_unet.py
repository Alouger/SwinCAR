import torch.nn as nn
from einops import rearrange

from self_attention_cv.transunet.bottleneck_layer import Bottleneck
from self_attention_cv.transunet.decoder import Up, SignleConv
from self_attention_cv.vit import ViT

import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat


class DoubleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(SignleConv(in_ch, out_ch, norm_layer),
                                  SignleConv(out_ch, out_ch, norm_layer))

    def forward(self, x):
        return self.conv(x)


class final_Up(nn.Module):
    """
    Doubles spatial size with bilinear upsampling
    Skip connections and double convs
    """

    def __init__(self, in_ch, out_ch):
        super(final_Up, self).__init__()
        mode = "bilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: [b,c, h, w]
            x2: [b,c, 2*h,2*w]

        Returns: 2x upsampled double conv reselt
        """
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)

class final_Up_single(nn.Module):
    """
    Doubles spatial size with bilinear upsampling
    Skip connections and double convs
    """

    def __init__(self, in_ch, out_ch):
        super(final_Up_single, self).__init__()
        mode = "bilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = SignleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: [b,c, h, w]
            x2: [b,c, 2*h,2*w]

        Returns: 2x upsampled double conv reselt
        """
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            # print(self.relative_indices.type())
            self.relative_indices = self.relative_indices.type(torch.long)
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, dropout):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)
    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, dropout):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
        #             x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Acon_FReLU(nn.Module):
    r""" ACON activation (activate or not) based on FReLU:
    # eta_a(x) = x, eta_b(x) = dw_conv(x), according to
    # "Funnel Activation for Visual Recognition" <https://arxiv.org/pdf/2007.11824.pdf>.
    """

    def __init__(self, width, stride=1):
        super().__init__()
        self.stride = stride

        # eta_b(x)
        self.conv_frelu = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=width, bias=True)
        self.bn1 = nn.BatchNorm2d(width)

        # eta_a(x)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        if self.stride == 2:
            x1 = self.maxpool(x)
        else:
            x1 = x

        x2 = self.bn1(self.conv_frelu(x))

        return self.bn2((x1 - x2) * self.sigmoid(x1 - x2) + x2)


class TFBlock(nn.Module):
    def __init__(self, inp, stride):
        super(TFBlock, self).__init__()
        self.oup = inp * stride
        self.stride = stride

        branch_main = [
            # pw conv
            nn.Conv2d(inp, inp, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(inp),
            Acon_FReLU(inp),
            # pw conv
            nn.Conv2d(inp, inp, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(inp)
        ]
        self.branch_main = nn.Sequential(*branch_main)

        self.acon = Acon_FReLU(self.oup, stride)

    def forward(self, x):
        x_proj = x
        x = self.branch_main(x)

        if self.stride == 1:
            return self.acon(x_proj + x)

        elif self.stride == 2:
            return self.acon(torch.cat((x_proj, x), 1))


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()  # non-linear

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        branch_main = [
            # pw conv
            nn.Conv2d(inp, mip, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(mip),
            Acon_FReLU(mip),
            # pw conv
            nn.Conv2d(mip, mip, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(mip)
        ]
        self.branch_main = nn.Sequential(*branch_main)

        self.acon = Acon_FReLU(mip, 1)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        # y = self.conv1(y)
        # y = self.bn1(y)
        # y = self.act(y)
        x_proj = y
        y = self.branch_main(y)
        y = self.acon(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True, dropout):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


class SUnet(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, num_classes=14, channels=3, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True, dropout):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding, dropout=dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )
        """
        My reimplementation of TransUnet based on the paper:
        https://arxiv.org/abs/2102.04306
        Badly written, many details missing and significantly differently
        from the authors official implementation (super messy code also :P ).
        My implementation doesnt match 100 the authors code.
        Basically I wanted to see the logic with vit and resnet backbone for
        shaping a unet model with long skip connections.

        Args:
            img_dim: the img dimension
            in_channels: channels of the input
            classes: desired segmentation classes
            vit_blocks: MHSA blocks of ViT
            vit_heads: number of MHSA heads
            vit_dim_linear_mhsa_block: MHSA MLP dimension
        """
        # super().__init__()
        self.inplanes = 128
        vit_channels = self.inplanes * 8

        # Not clear how they used resnet arch. since the first input after conv
        # must be 128 channels and half spat dims.
        # in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                      bias=False)
        # bn1 = nn.BatchNorm2d(self.inplanes)
        # self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)

        # self.img_dim_vit = img_dim // 16
        # self.vit = ViT(img_dim=self.img_dim_vit,
        #                in_channels=vit_channels,  # encoder channels
        #                patch_dim=1,
        #                # dim=vit_channels,  # vit out channels for decoding
        #                dim=768,
        #                blocks=vit_blocks,
        #                heads=vit_heads,
        #                dim_linear_block=vit_dim_linear_mhsa_block,
        #                classification=False,
        #                dropout=0)
        #
        # self.vit_conv = SignleConv(in_ch=768, out_ch=512)

        self.CA_block_4 = CoordAtt(384, 384)
        #self.CA_block_4 = CoordAtt(512,512)
        #         self.CA_block_4 = CoordAtt(768,768)

        self.CA_block_2 = CoordAtt(192, 192)
        #self.CA_block_2 = CoordAtt(256,256)
        #         self.CA_block_2 = CoordAtt(384,384)

        self.CA_block_1 = CoordAtt(96, 96)
        #self.CA_block_1 = CoordAtt(128,128)
        #         self.CA_block_1 = CoordAtt(192,192)

        self.CA_block_x4 = CoordAtt(1024, 1024)
        #         self.CA_block_x4 = CoordAtt(1536,1536)

        self.dec1 = Up(1152, 192)
        #self.dec1 = Up(1536, 256)
        #         self.dec1 = Up(2304, 384)

        self.dec2 = Up(384, 96)
        #self.dec2 = Up(512, 128)
        #         self.dec2 = Up(768, 192)

        self.dec3 = Up(192, 64)
        #self.dec3 = Up(256, 64)
        #         self.dec3 = Up(384, 64)

        self.dec4 = final_Up(64, 32)
        self.conv1x1 = nn.Conv2d(16, num_classes, kernel_size=1)

        self.dec5 = final_Up_single(32, 16)

    def forward(self, x):
        x1 = self.stage1(x)
        x1 = self.CA_block_1(x1)
        #         print("x1:",x1.shape)
        x2 = self.stage2(x1)
        x2 = self.CA_block_2(x2)
        #         print("x2:",x2.shape)
        x3 = self.stage3(x2)
        x3 = self.CA_block_4(x3)
        #         print("x3:",x3.shape)
        x4 = self.stage4(x3)
        #         x4 = self.CA_block_x4(x4)
        #         print("x4:",x4.shape)

        # ResNet 50-like encoder
        # x2 = self.init_conv(x)  # 128,64,64
        # x4 = self.conv1(x2)  # 256,32,32
        # x8 = self.conv2(x4)  # 512,16,16
        # x16 = self.conv3(x8)  # 1024,8,8
        # y = self.vit(x16)
        # y = rearrange(y, 'b (x y) dim -> b dim x y ', x=self.img_dim_vit, y=self.img_dim_vit)
        # y = self.vit_conv(y)
        #         x3 = self.CA_block_4(x3)

        y3 = self.dec1(x4, x3)  # 256,16,16
        #         print("y3:",y3.shape)

        #         x2 = self.CA_block_2(x2)

        y2 = self.dec2(y3, x2)
        #         print("y2:",y2.shape)

        #         x1 = self.CA_block_1(x1)

        y1 = self.dec3(y2, x1)
        #         print("y1:",y1.shape)
        y = self.dec4(y1)
        #         print("y:",y.shape)
        y_224 = self.dec5(y)
        #         print("y_224:",y_224.shape)
        return self.conv1x1(y_224)
