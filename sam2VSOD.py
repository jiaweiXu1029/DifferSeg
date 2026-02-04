import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from thop import profile, clever_format
import time
# from mmcv.runner import BaseModule
from mmengine.model import BaseModule
import numpy as np
from HA import HA
from timm.models.layers import trunc_normal_
import math
from torchvision.ops import deform_conv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class DifferentialPerceptionFusion(nn.Module):
    def __init__(self, channels):
        super(DifferentialPerceptionFusion, self).__init__()
        self.channels = channels

        self.learnable_diff_x = nn.Parameter(torch.randn(1, 1, 3, 3) * 0.1)
        self.learnable_diff_y = nn.Parameter(torch.randn(1, 1, 3, 3) * 0.1)
        self.learnable_laplacian = nn.Parameter(torch.randn(1, 1, 3, 3) * 0.1)

        self._init_differential_kernels()

        self.rgb_proj = nn.Conv2d(channels, channels, 1)
        self.depth_proj = nn.Conv2d(channels, channels, 1)

        self.complementarity_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )

        self.fusion_net = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.integration_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

        self.low_freq_pool = nn.AvgPool2d(3, stride=1, padding=1)

    def _init_differential_kernels(self):
        with torch.no_grad():
            # 初始化为Sobel X
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            self.learnable_diff_x.data = sobel_x.view(1, 1, 3, 3)

            # 初始化为Sobel Y
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            self.learnable_diff_y.data = sobel_y.view(1, 1, 3, 3)

            # 初始化为Laplacian
            laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
            self.learnable_laplacian.data = laplacian.view(1, 1, 3, 3)

    def compute_differential_features(self, x):
        """使用学习性算子计算差分特征"""
        B, C, H, W = x.shape
        x_reshaped = x.contiguous().view(B * C, 1, H, W)

        # 使用学习性差分算子
        grad_x = F.conv2d(x_reshaped, self.learnable_diff_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.learnable_diff_y, padding=1)
        laplacian = F.conv2d(x_reshaped, self.learnable_laplacian, padding=1)

        # 恢复到原始形状
        grad_x = grad_x.view(B, C, H, W)
        grad_y = grad_y.view(B, C, H, W)
        laplacian = laplacian.view(B, C, H, W)

        return grad_x, grad_y, laplacian

    def forward(self, rgb_feat, depth_feat):
        B, C, H, W = rgb_feat.shape

        # 投影
        rgb_proj = self.rgb_proj(rgb_feat)
        depth_proj = self.depth_proj(depth_feat)

        # 差分特征
        rgb_grad_x, rgb_grad_y, rgb_laplacian = self.compute_differential_features(rgb_proj)
        depth_grad_x, depth_grad_y, depth_laplacian = self.compute_differential_features(depth_proj)

        # 互补性建模
        combined_feat = torch.cat([rgb_proj, depth_proj], dim=1)
        complementarity_weights = self.complementarity_net(combined_feat)

        rgb_weight = complementarity_weights[:, 0:1, :, :].expand(-1, C, -1, -1)
        depth_weight = complementarity_weights[:, 1:2, :, :].expand(-1, C, -1, -1)

        # 差分特征融合
        fused_grad_x = rgb_weight * rgb_grad_x + depth_weight * depth_grad_x
        fused_grad_y = rgb_weight * rgb_grad_y + depth_weight * depth_grad_y
        fused_laplacian = rgb_weight * rgb_laplacian + depth_weight * depth_laplacian

        fused_diff_features = torch.cat([fused_grad_x, fused_grad_y, fused_laplacian], dim=1)
        fused_features = self.fusion_net(fused_diff_features)

        # --- 低频补偿（改成空间平滑版） ---
        rgb_low_freq = self.low_freq_pool(rgb_proj)
        depth_low_freq = self.low_freq_pool(depth_proj)

        # 对权重做 3x3 空间平滑
        rgb_weight_smooth = F.avg_pool2d(complementarity_weights[:, 0:1, :, :],
                                         kernel_size=3, stride=1, padding=1)
        depth_weight_smooth = F.avg_pool2d(complementarity_weights[:, 1:2, :, :],
                                           kernel_size=3, stride=1, padding=1)

        # 扩展到通道维度
        rgb_weight_smooth = rgb_weight_smooth.expand(-1, C, -1, -1)
        depth_weight_smooth = depth_weight_smooth.expand(-1, C, -1, -1)

        combined_low_freq = rgb_weight_smooth * rgb_low_freq + depth_weight_smooth * depth_low_freq
        # --- 结束 ---

        # 集成
        final_input = torch.cat([fused_features, combined_low_freq], dim=1)
        reconstructed_features = self.integration_net(final_input)

        # 残差式输出
        output = reconstructed_features + rgb_proj + depth_proj

        return output

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
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class FrequencyGuidedDecoder(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64]):
        super().__init__()
        self.channels = channels

        # 保留核心的频域分解模块
        self.freq_decomposers = nn.ModuleList([
            EnhancedFrequencyDecomposer(ch) for ch in channels
        ])

        # 轻量级的跨尺度注意力（内存友好）
        self.scale_attention = LightweightScaleAttention(channels)

        # 频域引导的上采样路径
        self.freq_guided_upsamplers = nn.ModuleList([
            EnhancedFreqUpsampler(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

        # 自适应频域权重调节
        self.freq_balancer = AdaptiveFrequencyBalancer(channels[-1])

    def forward(self, features):
        # 频域分解
        freq_decompositions = []
        for i, feat in enumerate(features):
            low_freq, high_freq = self.freq_decomposers[i](feat)
            freq_decompositions.append((low_freq, high_freq))

        # 轻量级跨尺度注意力
        enhanced_freqs = self.scale_attention(freq_decompositions)

        # 渐进式上采样
        current_low, current_high = enhanced_freqs[0]

        for i in range(len(enhanced_freqs) - 1):
            next_low, next_high = enhanced_freqs[i + 1]
            current_low, current_high = self.freq_guided_upsamplers[i](
                current_low, current_high, next_low, next_high
            )

        # 最终频域平衡
        final_feat = self.freq_balancer(current_low, current_high)

        return final_feat

class EnhancedFrequencyDecomposer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 添加更多尺度的滤波器
        self.low_pass_3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.low_pass_5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.low_pass_7x7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)  # 新增

        # 改进低频融合 - 使用注意力机制
        self.low_freq_attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),  # 3个尺度
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),  # 3个权重
            nn.Softmax(dim=1)
        )

        self.low_freq_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 改进高频增强 - 多分支处理
        self.high_freq_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True)
            ) for k in [1, 3, 5]
        ])

        self.high_freq_fusion = nn.Sequential(
            nn.Conv2d(channels // 4 * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度低频提取
        low_3x3 = self.low_pass_3x3(x)
        low_5x5 = self.low_pass_5x5(x)
        low_7x7 = self.low_pass_7x7(x)

        # 计算注意力权重
        low_combined = torch.cat([low_3x3, low_5x5, low_7x7], dim=1)
        attention_weights = self.low_freq_attention(low_combined)

        # 加权融合低频
        weighted_low = (low_3x3 * attention_weights[:, 0:1] +
                        low_5x5 * attention_weights[:, 1:2] +
                        low_7x7 * attention_weights[:, 2:3])

        # 高频分量
        high_freq_raw = x - weighted_low

        # 多分支高频处理
        high_features = [branch(high_freq_raw) for branch in self.high_freq_branches]
        high_combined = torch.cat(high_features, dim=1)
        enhancement_weight = self.high_freq_fusion(high_combined)

        high_freq = high_freq_raw * enhancement_weight

        return weighted_low, high_freq

class LightweightScaleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 分别处理低频和高频的注意力
        self.low_freq_attention = self._build_attention_branch(channels, 'low')
        self.high_freq_attention = self._build_attention_branch(channels, 'high')

        # 频域交互模块
        self.freq_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch // 8),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 1)
            ) for ch in channels
        ])

    def _build_attention_branch(self, channels, freq_type):
        return nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, ch // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 8, ch, 1),
                nn.Sigmoid()
            ) for ch in channels
        ])

    def forward(self, freq_decompositions):
        enhanced_freqs = []

        for i, (low, high) in enumerate(freq_decompositions):
            # 分别计算低频和高频的注意力
            low_att = self.low_freq_attention[i](low)
            high_att = self.high_freq_attention[i](high)

            # 频域交互
            freq_interaction = self.freq_interaction[i](low * high)

            # 应用注意力和交互
            enhanced_low = low * low_att + freq_interaction * 0.1
            enhanced_high = high * high_att + freq_interaction * 0.1

            enhanced_freqs.append((enhanced_low, enhanced_high))

        return enhanced_freqs

class EnhancedFreqUpsampler(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # 统一使用相同kernel_size避免尺寸不一致
        self.low_upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(3)  # 3个相同的上采样器
        ])

        self.high_upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])

        # 为每个上采样器添加不同的特征变换来增加多样性
        self.low_transforms = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=k // 2, groups=out_ch // 4)
            for k in [1, 3, 5]
        ])

        self.high_transforms = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=k // 2, groups=out_ch // 4)
            for k in [1, 3, 5]
        ])

        # 上采样路径选择
        self.path_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化避免空间尺寸问题
            nn.Conv2d(out_ch * 6, out_ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, 6, 1),
            nn.Softmax(dim=1)
        )

        # 改进细节恢复模块 - 确保尺寸一致
        self.detail_recovery = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),  # 保持相同尺寸
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1)
        )

    def forward(self, prev_low, prev_high, curr_low, curr_high):
        target_size = curr_low.shape[2:]  # 目标尺寸

        # 多路径上采样
        low_ups = []
        high_ups = []

        for i, (low_up, high_up) in enumerate(zip(self.low_upsamplers, self.high_upsamplers)):
            # 上采样
            up_low = low_up(prev_low)
            up_high = high_up(prev_high)

            # 强制对齐到目标尺寸
            if up_low.shape[2:] != target_size:
                up_low = F.interpolate(up_low, size=target_size,
                                       mode='bilinear', align_corners=False)
                up_high = F.interpolate(up_high, size=target_size,
                                        mode='bilinear', align_corners=False)

            # 应用特征变换增加多样性
            up_low = self.low_transforms[i](up_low)
            up_high = self.high_transforms[i](up_high)

            low_ups.append(up_low)
            high_ups.append(up_high)

        # 路径选择权重计算
        all_paths = torch.cat(low_ups + high_ups, dim=1)  # [B, out_ch*6, H, W]
        path_weights = self.path_selector(all_paths)  # [B, 6, 1, 1]

        # 加权融合上采样结果
        weighted_low = sum(low_ups[i] * path_weights[:, i:i + 1] for i in range(3))
        weighted_high = sum(high_ups[i] * path_weights[:, i + 3:i + 4] for i in range(3))

        # 确保尺寸匹配后融合
        assert weighted_low.shape == curr_low.shape, f"Size mismatch: {weighted_low.shape} vs {curr_low.shape}"
        assert weighted_high.shape == curr_high.shape, f"Size mismatch: {weighted_high.shape} vs {curr_high.shape}"

        # 与当前层融合
        fused_low = weighted_low * 0.6 + curr_low * 0.4
        fused_high = weighted_high * 0.6 + curr_high * 0.4

        # 细节恢复 - 确保输出尺寸正确
        detail_input = torch.cat([fused_low, fused_high, curr_low, curr_high], dim=1)
        detail_enhancement = self.detail_recovery(detail_input)

        # 确保detail_enhancement与fused特征尺寸一致
        if detail_enhancement.shape[2:] != fused_low.shape[2:]:
            detail_enhancement = F.interpolate(detail_enhancement, size=fused_low.shape[2:],
                                               mode='bilinear', align_corners=False)

        output_low = fused_low + detail_enhancement * 0.3
        output_high = fused_high + detail_enhancement * 0.3

        return output_low, output_high

class AdaptiveFrequencyBalancer(nn.Module):
    """自适应频域平衡器"""

    def __init__(self, channels):
        super().__init__()

        # 全局上下文提取
        self.global_context = nn.AdaptiveAvgPool2d(1)

        # 频域权重预测网络
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        # 最终融合网络
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, low_freq, high_freq):
        # 全局上下文特征
        combined = torch.cat([low_freq, high_freq], dim=1)
        global_ctx = self.global_context(combined)

        # 预测频域权重
        weights = self.weight_predictor(global_ctx)
        low_weight, high_weight = weights.split(1, dim=1)

        # 应用权重
        weighted_low = low_freq * low_weight
        weighted_high = high_freq * high_weight

        # 最终融合
        final_combined = torch.cat([weighted_low, weighted_high], dim=1)
        final_output = self.final_fusion(final_combined)

        # 残差连接
        residual = (low_freq + high_freq) * 0.5
        return final_output + residual

class SAM2(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(SAM2, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 128)
        self.rfb3 = RFB_modified(576, 256)
        self.rfb4 = RFB_modified(1152, 512)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.predtrans4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
        )
        self.predtrans1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)
        )

        self.dpf4 = DifferentialPerceptionFusion(512)
        self.dpf3 = DifferentialPerceptionFusion(256)
        self.dpf2 = DifferentialPerceptionFusion(128)
        self.dpf1 = DifferentialPerceptionFusion(64)
        self.freq_decoder=FrequencyGuidedDecoder()

    def forward(self, x, d):
        x1, x2, x3, x4 = self.encoder(x)
        d1, d2, d3, d4 = self.encoder(d)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        d1, d2, d3, d4 = self.rfb1(d1), self.rfb2(d2), self.rfb3(d3), self.rfb4(d4)

        f4 = self.dpf4(x4, d4)
        out4 = F.interpolate(self.predtrans4(f4), 512, mode="bilinear", align_corners=True)
        f3 = self.dpf3(x3, d3)
        out3 = F.interpolate(self.predtrans3(f3), 512, mode="bilinear", align_corners=True)
        f2 = self.dpf2(x2, d2)
        out2 = F.interpolate(self.predtrans2(f2), 512, mode="bilinear", align_corners=True)
        f1 = self.dpf1(x1, d1)
        features = [f4, f3, f2, f1]
        main_pred = self.freq_decoder(features)
        out1 = F.interpolate(self.predtrans1(main_pred), 512, mode="bilinear", align_corners=True)


        return  out1, out2, out3,out4

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2().cuda()
        x = torch.randn(1, 3, 512, 512).cuda()
        d = torch.randn([1, 3, 512, 512]).cuda()
        params, flops = profile(model, inputs=(x, d))
        flops, params = clever_format([flops, params], "%.2f")
        print(f"FLOPs: {flops}, Params: {params}")
