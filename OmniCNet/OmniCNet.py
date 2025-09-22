from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, bn_momentum=0.0003):
        super().__init__()
        inter_channels = in_channels // 4


        self.last_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(inter_channels, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )


        self.classify = nn.Conv2d(in_channels=inter_channels, out_channels= out_channels, kernel_size=1,
                                        stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
       
        x = self.last_conv(x)

        pred = self.classify(x)
        return pred
    
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
    
class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

class DropBlock(nn.Module):
    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()

        self.drop = LinearScheduler(
            DropBlock2D(block_size=size, drop_prob=0.),
            start_value=0,
            stop_value=rate,
            nr_steps=step
        )
    def forward(self, feats: list):
        if self.training:  # 只在训练的时候加上dropblock
            for i, feat in enumerate(feats):
                feat = self.drop(feat)
                feats[i] = feat
        return feats

    def step(self):
        self.drop.step()

class AlignedModulev2PoolingAtten(nn.Module):

    def __init__(self, inplanel,inplaneh, outplane, kernel_size=3):
        super(AlignedModulev2PoolingAtten, self).__init__()
        self.down_h = nn.Conv2d(inplaneh, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplanel, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1,x2):
        low_feature=x1
        h_feature = x2
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        # h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :], flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        h_feature_mean = torch.mean(h_feature, dim=1).unsqueeze(1)
        l_feature_mean = torch.mean(low_feature, dim=1).unsqueeze(1)
        h_feature_max = torch.max(h_feature, dim=1)[0].unsqueeze(1)
        l_feature_max = torch.max(low_feature, dim=1)[0].unsqueeze(1)

        flow_gates = self.flow_gate(torch.cat([h_feature_mean, l_feature_mean, h_feature_max, l_feature_max], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output
    
class SPP(nn.Module):
    def __init__(self, in_channels, sizes=(5, 9, 13)):
        super(SPP, self).__init__()
        size1, size2, size3 = sizes
        self.pool1 = nn.MaxPool2d(kernel_size=size1, stride=1, padding=size1//2)
        self.pool2 = nn.MaxPool2d(kernel_size=size2, stride=1, padding=size2//2)
        self.pool3 = nn.MaxPool2d(kernel_size=size3, stride=1, padding=size3//2)

        self.dim_reduction = Conv3Relu(in_channels * 4, in_channels)

    def forward(self, x):
        feat1 = self.pool1(x)
        feat2 = self.pool1(x)
        feat3 = self.pool1(x)

        out = self.dim_reduction(torch.cat([x, feat1, feat2, feat3], dim=1))
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        rate1, rate2, rate3 = tuple(atrous_rates)

        out_channels = int(in_channels / 2)

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

        # 全局平均池化
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

        self.dim_reduction = Conv3Relu(out_channels * 5, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]

        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)

        feat4 = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)

        out = self.dim_reduction(torch.cat((feat0, feat1, feat2, feat3, feat4), 1))

        return out
    
class PPM(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        size1, size2, size3, size4 = sizes
        out_channels = int(in_channels / 4)
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(size1),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(size2),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.pool3 = nn.Sequential(nn.AdaptiveAvgPool2d(size3),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(size4),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.dim_reduction = Conv3Relu(in_channels + out_channels * 4, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]

        feat1 = F.interpolate(self.pool1(x), (h, w), mode="bilinear", align_corners=True)
        feat2 = F.interpolate(self.pool2(x), (h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(self.pool3(x), (h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(self.pool4(x), (h, w), mode="bilinear", align_corners=True)

        out = self.dim_reduction(torch.cat((x, feat1, feat2, feat3, feat4), 1))
        return out
    
class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x
    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GCAblock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 ):
        super(GCAblock, self).__init__()
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        out = out + out * channel_mul_term
        return out

class LDAblock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1

class HCF(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.conv1_1 = BasicConv2d(embed_dim * 2, embed_dim, 1)
        self.conv1_1_1 = BasicConv2d(input_dim // 2, embed_dim, 1)
        self.local_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.global_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.GlobelBlock = GCAblock(inplanes=embed_dim, ratio=2)
        self.local = LDAblock(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        local = self.local(self.local_11conv(x_0))
        Globel = self.GlobelBlock(self.global_11conv(x_1))
        x = torch.cat([local, Globel], dim=1)
        x = self.conv1_1(x)

        return x

class HCF_fusion(nn.Module):
    def __init__(self,dim):
        super(HCF_fusion,self).__init__()
        self.attention = HCF(dim+dim+dim+dim,dim)
    def forward(self,feature_list):
        for i,feature in enumerate(feature_list[:-1]):
            feature = F.pixel_shuffle(feature,1)
            x = feature if i ==0 else torch.cat([x,feature],dim=1)
        x = torch.cat([x,feature_list[-1]],dim=1)
        x = self.attention(x)
        return x
    
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )
class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()
        inplanes = 128
        self.stage1_Conv1 = Conv3Relu(128 * 1, 128)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)
        
        self.scn41= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn31= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.scn21= AlignedModulev2PoolingAtten(inplanes , inplanes, inplanes)
        self.final_Conv5 = Conv3Relu(inplanes , inplanes)       
        if "+ppm+" in neck_name:
            self.expand_field = PPM(inplanes * 8)
        elif "+aspp+" in neck_name:
            self.expand_field = ASPP(inplanes * 8)
        elif "+spp+" in neck_name:
            self.expand_field = SPP(inplanes * 8)
        else:
            self.expand_field = None

        if "fuse" in neck_name:
            self.stage2_Conv3 = Conv3Relu(inplanes * 2, inplanes)   
            self.stage3_Conv3 = Conv3Relu(inplanes * 4, inplanes)
            self.stage4_Conv3 = Conv3Relu(inplanes * 8, inplanes)
            self.final_Conv = Conv3Relu(inplanes * 4, inplanes)
            self.fuse = True
        else:
            self.fuse = False
        self.fusion = HCF_fusion(128)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4 = ms_feats
        feature1_h, feature1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4] = self.drop([fa1, fa2, fa3, fa4])  
        feature1 = self.stage1_Conv1(torch.cat([fa1], 1))  
        feature2 = self.stage2_Conv1(torch.cat([fa2], 1))  
        feature3 = self.stage3_Conv1(torch.cat([fa3], 1))  
        feature4 = self.stage4_Conv1(torch.cat([fa4], 1)) 
        feature_out = feature4
        if self.expand_field is not None:
            feature4 = self.expand_field(feature4)
        feature3_2 = self.stage4_Conv_after_up(self.up(feature4))
        feature3 = self.stage3_Conv2(torch.cat([feature3, feature3_2], 1))

        feature2_2 = self.stage3_Conv_after_up(self.up(feature3))
        feature2 = self.stage2_Conv2(torch.cat([feature2, feature2_2], 1))

        feature1_2 = self.stage2_Conv_after_up(self.up(feature2))
        feature1 = self.stage1_Conv2(torch.cat([feature1, feature1_2], 1))
    
        if self.fuse:
            feature4=self.scn41(feature1, self.stage4_Conv3(feature4))
            feature3=self.scn31(feature1, self.stage3_Conv3(feature3))
            feature2=self.scn21(feature1, self.stage2_Conv3(feature2))
            feature = self.fusion([feature1,feature2,feature3,feature4])
        else:
            feature = feature1
            feature=self.final_Conv5(feature1)
        return feature


class CAKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1,
                                      padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + \
               attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
    
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class ca_backbone(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, wcmf_channel=1024,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.cak = CAKblock(base_dim*2**3)
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        features[3] = self.cak(features[3])
        return features

def CA_backbone(**kwargs):
    model = ca_backbone(128, [1, 2, 6, 2], **kwargs)
    return model

    
class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x
        
class Seg_Detection(nn.Module):
    def __init__(self,  fusion_method='conv'):
        super().__init__()
        self.inplanes = 128  # 从backbone名称中提取通道数
        self.backbone2 = None  # 确保初始化
        self._create_backbone()
        self._create_neck()
        self._create_heads()
        self.contrast_loss1 = Conv3Relu(1024,2)
        self.contrast_loss2 = Conv3Relu(1024,16)
        # 定义融合方式：'conv' 或 'attention'
        self.fusion_method = fusion_method

        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # Stage 1
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)  # Stage 2
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)  # Stage 3
        self.conv4 = nn.Conv2d(2048, 1024, kernel_size=1)  # Stage 4

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)  # todo:这里预训练初始化和 hrnet主干网络的初始化有冲突，必须要改！

    def forward(self, x):
        _, _, h_input, w_input = x.shape
        f1, f2, f3, f4 = self.backbone(x)
        ms_feats = (f1, f2, f3, f4)
        feature = self.neck(ms_feats)
        out = self.head_forward(feature, out_size=(h_input, w_input))
        return out
    
    def head_forward(self, feature , out_size):
        out = F.interpolate(self.head(feature ), size=out_size, mode='bilinear', align_corners=True)
        return out
  

    def _init_weight(self, pretrain=''):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 只要是卷积都操作，都对weight和bias进行kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  # bn层都权重初始化为1， bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                                if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))

    def _create_backbone(self):
        self.backbone = CA_backbone()
    
    def _create_neck(self):
        self.neck = FPNNeck(128,"fpn+aspp+fuse+drop")

    def _select_head(self):
        return FCNHead(self.inplanes, 2)

    def _create_heads(self):
        self.head = self._select_head()    

if __name__=='__main__':
    
    img=torch.randn(4,3,256,256)
    
    import argparse
    parser = argparse.ArgumentParser('Seg Detection train')
    parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str,
                        default=" ")  # 预训练权重路径
    parser.add_argument("--input-size", type=int, default=256)
    
    opt = parser.parse_args()
    model=Seg_Detection()
    pred=model(img)
    print('pred_shape',pred.shape)
    from thop import profile
    flops, params = profile(model, inputs=(img, ))
    
    print(f"Number of FLOPs: {flops / 1e9:.2f}G")  # Convert to billions of FLOPs
    print(f"Number of parameters: {params / 1e6:.2f}M")  # Convert to millions of parameters
   
