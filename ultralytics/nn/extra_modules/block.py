# Ultralytics-based paper experiment modules, AGPL-3.0.
# Selected from the verified experiment source; retained computations are unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.conv import Conv, RepConv
from ..modules.block import RepC3, C2f
from .MobileUViT import LGLBlock
from .MAC import MAC
from .filc import FAT_Block

__all__ = ('WCSPELAN', 'DKCSPELAN', 'SpatialSepC3', 'C2f_FAT', 'C2f_LGLB', 'C2f_MAC')


class SpatialSep(nn.Module):
    """SP+GL融合: 1×3+3×1空间可分离 + GELU + 恒等skip, 参数↓28%"""
    def __init__(self, c_in, c_out, s=1, gain=1, bias=True):
        super().__init__()
        mid_in, mid_out = int(c_in * gain), int(c_out * gain)

        self.expand = nn.Conv2d(c_in, mid_in, 1, padding=0, bias=bias)
        self.conv_1x3 = nn.Conv2d(mid_in, mid_out, (1, 3), stride=s, padding=0, bias=bias)
        self.conv_3x1 = nn.Conv2d(mid_out, mid_out, (3, 1), stride=1, padding=0, bias=bias)
        self.project = nn.Conv2d(mid_out, c_out, 1, padding=0, bias=bias)

        self.has_sk = not (c_in == c_out and s == 1)
        if self.has_sk:
            self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, bias=bias)

        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()

    def forward(self, x):
        pad = 1
        x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
        out = self.expand(x_pad)
        out = self.conv_1x3(out)
        out = self.conv_3x1(out)
        out = self.project(out)
        out = self.bn(out)
        skip = x if not self.has_sk else self.sk(x)
        return self.act(out + skip)


class SpatialSepC3(RepC3):
    """方案B版: 1×3+3×1 + GELU + 恒等skip"""
    def __init__(self, c1, c2, n=3, e=1):
        super().__init__(c1, c2, n, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[SpatialSep(c_, c_) for _ in range(n)])


class RGCSPELAN(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RGCSPELAN, self).__init__()
        
        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        
        self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)
        
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))


class WCSPELAN(RGCSPELAN):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super().__init__(c1, c2, n, scale, e)
        mid_half = self.mid // 2
        self.cv3_s = RepConv(self.c, mid_half, 3)           
        self.cv3_l = RepConv(self.c, mid_half, 3, d=2)      
        self.cv4 = Conv(self.mid, self.mid, 1)              
        del self.cv3

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))                  
        y_half = self.cv3_s(y[-1])                          
        y_full = self.cv3_l(y[-1])                         
        y[-1] = torch.cat([y_half, y_full], dim=1)         
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))                            # cv4输出
        return self.cv2(torch.cat(y, 1))


DKCSPELAN = WCSPELAN


class ConvolutionalGLU(nn.Module):
    """
    卷积门控线性单元（Convolutional GLU）
    结合了门控机制和深度可分离卷积的特征增强模块
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        """
        初始化函数
        Args:
            in_features: 输入特征通道数
            hidden_features: 隐藏层特征通道数，默认等于输入通道数
            out_features: 输出特征通道数，默认等于输入通道数
            act_layer: 激活函数层，默认使用GELU
            drop: Dropout概率
        """
        super().__init__()
        out_features = out_features or in_features  # 如果未指定输出通道数，则使用输入通道数
        hidden_features = hidden_features or in_features  # 如果未指定隐藏层通道数，则使用输入通道数
        hidden_features = int(2 * hidden_features / 3)  # 隐藏层通道数缩小为原来的2/3，节省计算量

        # 第一个1x1卷积，输出通道数为hidden_features*2，用于生成两部分特征
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        # 深度可分离卷积块：3x3深度卷积 + GELU激活
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1,
                     padding=1, bias=True, groups=hidden_features),  # 深度卷积，每个通道独立卷积
            act_layer()  # 激活函数
        )

        # 第二个1x1卷积，将隐藏层特征映射到输出特征
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

        # Dropout层，用于正则化
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图，形状 [B, C, H, W]
        Returns:
            输出特征图，形状 [B, out_features, H, W]
        """
        x_shortcut = x  # 保存输入用于残差连接

        # 1x1卷积生成两部分特征：x用于主路径，v用于门控
        x, v = self.fc1(x).chunk(2, dim=1)  # 沿通道维度分成两部分

        # 主路径：深度卷积处理 + 门控乘法
        x = self.dwconv(x) * v  # 门控机制：dwconv(x) * v

        x = self.drop(x)  # Dropout正则化

        # 输出投影
        x = self.fc2(x)

        x = self.drop(x)  # Dropout正则化

        return x_shortcut + x  # 残差连接：输入 + 输出


class C2f_FAT(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FAT_Block(self.c) for _ in range(n))


class C2f_LGLB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(LGLBlock(self.c) for _ in range(n))


class C2f_MAC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MAC(self.c, self.c) for _ in range(n))
