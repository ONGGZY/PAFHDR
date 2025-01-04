import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from mmcv.ops import DeformConv2d
from .mamba import Mamba, SS2D,SS4D
from losses import Get_gradient,GradientCosineLoss

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.ln(x)
        x = rearrange(x, 'b (h w) c ->b c h w', h=h, w=w)
        return x



class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class SpatialAttentionModule(nn.Module):

    def __init__(self, out_channels):
        super().__init__()
        self.att1 = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        # self.m=Mamba(out_channels*2)

    def forward(self, x1, x2):
        f_cat = torch.cat([x1, x2], 1)
        # f_cat=self.m(f_cat)
        att_map = self.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class CrossExtractFeatureOrigin(nn.Module):
    def __init__(self, embed_dims, d_model):
        super(CrossExtractFeatureOrigin, self).__init__()
        self.att1 = SpatialAttentionModule(embed_dims)
        self.att2 = SpatialAttentionModule(embed_dims)

        self.out_conv = nn.Conv2d(embed_dims * 3, d_model, 1, 1)

    def forward(self, f1, f2, f3):
        att_12 = self.att1(f1, f2)
        att_32 = self.att2(f3, f2)

        f1 = f1 * att_12
        f3 = f3 * att_32
        f2 = f2 * (att_12 + att_32)

        space_feat = torch.cat([f1, f2, f3], dim=1)

        return self.out_conv(space_feat)

class AlignNet(nn.Module):
    def __init__(self, in_channels, d_model):
        super(AlignNet, self).__init__()
        self.att1 = SpatialAttentionModule(in_channels)
        self.att2 = SpatialAttentionModule(in_channels)

        self.out_conv = nn.Conv2d(in_channels * 3, d_model, 1, 1)
    def forward(self, f1, f2, f3):
        att_12 = self.att1(f1, f2)
        att_32 = self.att2(f3, f2)
        f1 = f1 * att_12
        f3 = f3 * att_32
        space_feat = torch.cat([f1, f2, f3], dim=1)

        return self.out_conv(space_feat)

class Mlp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        hidden_features = in_features * 2
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LocalContextExtractor(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        B, C, _, _ = x.shape
        y = self.avg_pool(x).reshape(B, C)
        y = self.fc(y).reshape(B, C, 1, 1)
        return x * y


class WMam(nn.Module):
    def __init__(self, window_size=8, d_model=64):
        super(WMam, self).__init__()
        self.window_size = window_size
        self.mamba = Mamba(d_model=d_model)
        self.pos = nn.Parameter(torch.zeros(1, d_model, window_size, window_size), requires_grad=True)

    def forward(self, x):
        _, _, H, W = x.size()
        ws = self.window_size
        x = rearrange(x, 'b d (nh hs) (nw ws) -> (b nh nw) d hs ws', hs=ws, ws=ws)  # hs:h_size ws:w_size
        x = x + self.pos
        x = self.mamba(x)
        x = rearrange(x, '(b nh nw) d hs ws -> b d (nh hs) (nw ws)', nh=H // ws, nw=W // ws)
        
        return x   
    
class W2DMam(nn.Module):
    def __init__(self, window_size=8, d_model=64):
        super(W2DMam, self).__init__()
        self.window_size = window_size
        self.mamba = SS2D(d_model=d_model)
        # self.pos = nn.Parameter(torch.zeros(1, d_model, window_size, window_size), requires_grad=True)

    def forward(self, x):
        _, _, H, W = x.size()
        ws = self.window_size
        x = rearrange(x, 'b d (nh hs) (nw ws) -> (b nh nw) d hs ws', hs=ws, ws=ws)  # hs:h_size ws:w_size
        # x = x + self.pos
        x = self.mamba(x)
        x = rearrange(x, '(b nh nw) d hs ws -> b d (nh hs) (nw ws)', nh=H // ws, nw=W // ws)
        return x


class SWM(nn.Module):
    def __init__(self, window_size=8, shift_size=4, d_model=64):
        super(SWM, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.mamba = SS2D(d_model=d_model)
        self.pos = nn.Parameter(torch.zeros(1, d_model, window_size, window_size), requires_grad=True)

    def forward(self, x):
        _, _, H, W = x.size()
        if self.shift_size > 0:
            img_scale = torch.ones((1, 1, H, W))
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            slices = ((slice(0, -self.window_size), slice(-self.shift_size, None)),
                      (slice(-self.shift_size, None), slice(0, -self.window_size)),
                      (slice(-self.shift_size, None), slice(-self.shift_size, None)))
            for h, w in slices:
                img_scale[:, :, h, w] = 1e-4

            x = x * img_scale.to(x.device)
        ws = self.window_size
        x = rearrange(x, 'b d (nh hs) (nw ws) -> (b nh nw) d hs ws', hs=ws, ws=ws)  # hs:h_size ws:w_size
        x = x + self.pos
        x = self.mamba(x)
        x = rearrange(x, '(b nh nw) d hs ws -> b d (nh hs) (nw ws)', nh=H // ws, nw=W // ws)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x


class SampleGate(nn.Module):
    def __init__(self):
        super(SampleGate, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LocalBranch(nn.Module):
    def __init__(self, d_model):
        super(LocalBranch, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(d_model, d_model, 1, 1)
        self.conv2 = nn.Conv2d(d_model, d_model, 1, 1)

    def forward(self, x):
        x = x * self.conv1(self.gap(x))
        return self.conv2(x)


class SS2DLBlock(nn.Module):
    def __init__(self, d_model, window_size=8, shift_size=4, drop_rate=0.):
        super(SS2DLBlock, self).__init__()
        self.ln_in = LayerNorm2d(d_model)
        self.ss2d = SS2D(d_model=d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.lcf = LocalContextExtractor(dim=d_model)

    def forward(self, x):
        # SWM
        short_cut = x
        x = self.ln_in(x)
        x_l = x
        x = self.ss2d(x) + short_cut

        # FFN
        y = rearrange(x, 'b d h w -> b h w d')
        x = self.ln_ffn(y)
        x = self.mlp(x)
        x = self.drop_path(x) + y
        x = rearrange(x, 'b h w d -> b d h w')

        # LC
        x_l = self.lcf(x_l)
        return x + x_l


class SSMLBlock(nn.Module):
    def __init__(self, d_model, window_size=8, shift_size=4, drop_rate=0.):
        super(SSMLBlock, self).__init__()
        self.ln_in = LayerNorm2d(d_model)
        self.mam = Mamba(d_model=d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.lcf = LocalContextExtractor(dim=d_model)

    def forward(self, x):
        # SWM
        short_cut = x
        x = self.ln_in(x)
        x_l = x
        x = self.mam(x) + short_cut

        # FFN
        y = rearrange(x, 'b d h w -> b h w d')
        x = self.ln_ffn(y)
        x = self.mlp(x)
        x = self.drop_path(x) + y
        x = rearrange(x, 'b h w d -> b d h w')

        # LC
        x_l = self.lcf(x_l)
        return x + x_l


class SWMLBlock(nn.Module):
    def __init__(self, d_model, window_size=8, shift_size=4, drop_rate=0.):
        super(SWMLBlock, self).__init__()
        self.ln_in = LayerNorm2d(d_model)
        self.swm = SWM(window_size=window_size, shift_size=shift_size, d_model=d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.lcf = LocalContextExtractor(dim=d_model)

    def forward(self, x):
        # SWM
        short_cut = x
        x = self.ln_in(x)
        x_l = x
        x = self.swm(x) + short_cut

        # FFN
        y = rearrange(x, 'b d h w -> b h w d')
        x = self.ln_ffn(y)
        x = self.mlp(x)
        x = self.drop_path(x) + y
        x = rearrange(x, 'b h w d -> b d h w')

        # LC
        x_l = self.lcf(x_l)
        return x + x_l


class BasicalBlock(nn.Module):
    def __init__(self, d_model, window_size=8, shift_size=4, drop_rate=0.):
        super(BasicalBlock, self).__init__()
        self.ln_in = LayerNorm2d(d_model)
        # self.ss2d = W2DMam(window_size=window_size,d_model=d_model)
        # self.ss2d = SWM(window_size=window_size,shift_size=shift_size,d_model=d_model)
        # self.ss2d = SS2D(d_model=d_model)
        self.ss2d = SS4D(d_model=d_model)
        # self.ss2d = Mamba(d_model = d_model)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.scale1 = nn.Parameter(torch.ones(1, d_model, 1, 1), requires_grad=True)
        self.scale2 = nn.Parameter(torch.ones(1, 1, 1, d_model), requires_grad=True)

    def forward(self, x):
        # SWM
        short_cut = x
        x = self.ln_in(x)
        x = self.ss2d(x) + short_cut*self.scale1

        # FFN
        y = rearrange(x, 'b d h w -> b h w d')
        x = self.ln_ffn(y)
        x = self.mlp(x)
        x = self.drop_path(x) + y*self.scale2
        x = rearrange(x, 'b h w d -> b d h w')

        return x 


class BLBlock(nn.Module):
    def __init__(self, d_model, drop_rate=0.):
        super(BLBlock, self).__init__()
        self.ln_in = LayerNorm2d(d_model)
        self.m = Mamba(d_model=d_model)
        self.beta = nn.Parameter(torch.ones(1, d_model, 1, 1), requires_grad=True)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.mlp = Mlp(in_features=d_model)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        short_cut = x
        x = self.ln_in(x)
        x = self.m(x) * self.beta
        x = x + short_cut
        # FFN
        y = rearrange(x, 'b d h w -> b h w d')
        x = self.ln_ffn(y)
        x = self.mlp(x)
        x = self.drop_path(x) + y
        x = rearrange(x, 'b h w d -> b d h w')
        return x


# multiple SWMLBlocks and CA
class GroupBlock(nn.Module):
    def __init__(self, d_model, depth, window_size, drop_rates):
        super(GroupBlock, self).__init__()
        shift_sizes = [0 if (i % 2 == 0) else (window_size // 2) for i in range(depth)]

        self.layers = nn.Sequential(*[BasicalBlock(d_model=d_model,
                                                    window_size=window_size,
                                                    shift_size=shift_sizes[i],
                                                    drop_rate=drop_rates[i]) for i in range(depth)])

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, 1, 1),
            nn.Conv2d(d_model // 4, d_model // 4, kernel_size=3, padding=2, dilation=2, groups=d_model // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(d_model // 4, d_model // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(d_model // 4, d_model, 1, 1),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model)
        )

    def forward(self, x):
        return self.dilated_conv(self.layers(x)) + x
        # return self.layers(x)+x


class PostProcessing(nn.Module):
    def __init__(self, out_channels):
        super(PostProcessing, self).__init__()
        self.ln1 = LayerNorm2d(out_channels)
        self.ln2 = LayerNorm2d(out_channels)

        self.step1 = nn.Sequential(nn.Conv2d(out_channels, out_channels * 2, 1, 1),
                                   nn.Conv2d(out_channels * 2, out_channels * 2, 3, 1, 1, groups=out_channels))
        self.sg = SampleGate()

        self.sca1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        self.mid_conv = nn.Conv2d(out_channels, out_channels, 1, 1)

        self.conv1 = nn.Conv2d(out_channels, out_channels * 2, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1)

    def forward(self, x):
        inp = x
        x = self.ln1(x)
        x = self.step1(x)
        x = self.sg(x)
        x = self.sca1(x) * x
        y = self.mid_conv(x) + inp

        x = self.ln2(y)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        return x + y
    

class MamHDR(nn.Module):
    def __init__(self, in_channels=64,d_model=64,out_channels=64, num_layers=[6, 6, 6],window_size=16, drop_path_rate=0.05 ):
        super(MamHDR, self).__init__()
        self.window_size = window_size


        self.conv_f1 = nn.Conv2d(6, in_channels, 3, 1, 1)
        self.conv_f2 = nn.Conv2d(6, in_channels, 3, 1, 1)
        self.conv_f3 = nn.Conv2d(6, in_channels, 3, 1, 1)

        self.fuse_layer= AlignNet(in_channels, d_model=d_model)
        
        self.grad_map=Get_gradient()
        self.l1_loss=nn.L1Loss()
        # self.GC_loss = GradientCosineLoss()
        slice_l = [0]
        s = 0
        for j in num_layers:
            s += j
            slice_l.append(s)
        total_blocks = slice_l[-1]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        self.body = nn.Sequential(
            *[GroupBlock(d_model=d_model,
                         depth=num_layers[i],
                         window_size=window_size,
                         drop_rates=dpr[slice_l[i]:slice_l[i + 1]]
                         )
              for i in range(len(num_layers))])
        self.cab_conv=nn.Conv2d(d_model,out_channels,1,1)
        self.conv_after_body = PostProcessing(out_channels)

        self.out_conv1 = nn.Sequential(nn.Conv2d(d_model, in_channels, 1, 1),
                                      nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels))
        self.act_first = nn.Sequential(nn.Conv2d(in_channels, 3, 3, 1, 1),
                                      nn.Sigmoid())

        self.out_conv = nn.Sequential(nn.Conv2d(out_channels, in_channels, 1, 1),
                                      nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels))
        self.act_last = nn.Sequential(nn.Conv2d(in_channels, 3, 3, 1, 1),
                                      nn.Sigmoid())

    def forward(self, f1, f2, f3):
        
        f1, _, _ = self.check_img_size(f1)
        f2, _, _ = self.check_img_size(f2)
        f3, pad_h, pad_w = self.check_img_size(f3)
        _, _, H, W = f1.shape
        ref_f=f2

        # 提取浅层特征
        f1 = self.conv_f1(f1)
        f2 = self.conv_f2(f2)
        f3 = self.conv_f3(f3)

        fused_f = self.fuse_layer(f1, f2, f3)  # (B, d_model, H, W)

        x=self.out_conv1(fused_f)
        result1=self.act_first(x+f2)
        
        if self.training:

            reslut1_map,reslut1_maph,reslut1_mapv=self.grad_map(result1)
            ref_f_map,ref_f_maph,ref_f_mapv=self.grad_map(ref_f)
            aligned_loss = self.l1_loss(ref_f_map,reslut1_map)
        else:
            aligned_loss =0

        x = self.body(fused_f)  # (B, d_model, H, W)
        # x = self.dilated_conv(x)
        x = self.cab_conv(x+fused_f) # (B, out_channels, H, W)
        x = self.conv_after_body(x)  # (B, out_model, H, W)
        x = self.out_conv(x) #(B, in_channels, H, W)
        result = self.act_last(x + f2)

        low, right = H - pad_h, W - pad_w
        return result[:, :, :low, :right],aligned_loss

    def check_img_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x, mod_pad_h, mod_pad_w

