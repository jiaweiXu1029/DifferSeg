import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from thop import profile, clever_format
import time
import numpy as np
from HA import HA
from timm.models.layers import trunc_normal_
import math
from torchvision.ops import deform_conv2d
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime

matplotlib.use('Agg')  # 使用非GUI后端


VISUALIZATION_DIR="/root/autodl-tmp/1"

class DifferentialPerceptionFusion(nn.Module):
    def __init__(self, channels, layer_name=''):
        super(DifferentialPerceptionFusion, self).__init__()
        self.channels = channels
        self.layer_name = layer_name
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
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            self.learnable_diff_x.data = sobel_x.view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            self.learnable_diff_y.data = sobel_y.view(1, 1, 3, 3)
            laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
            self.learnable_laplacian.data = laplacian.view(1, 1, 3, 3)

    def compute_differential_features(self, x):
        """使用学习性算子计算差分特征"""
        B, C, H, W = x.shape
        x_reshaped = x.contiguous().view(B * C, 1, H, W)
        grad_x = F.conv2d(x_reshaped, self.learnable_diff_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.learnable_diff_y, padding=1)
        laplacian = F.conv2d(x_reshaped, self.learnable_laplacian, padding=1)
        grad_x = grad_x.view(B, C, H, W)
        grad_y = grad_y.view(B, C, H, W)
        laplacian = laplacian.view(B, C, H, W)
        return grad_x, grad_y, laplacian

    def forward(self, rgb_feat, depth_feat):
        B, C, H, W = rgb_feat.shape
        rgb_proj = self.rgb_proj(rgb_feat)
        depth_proj = self.depth_proj(depth_feat)

        # 计算差分特征
        rgb_grad_x, rgb_grad_y, rgb_laplacian = self.compute_differential_features(rgb_proj)
        depth_grad_x, depth_grad_y, depth_laplacian = self.compute_differential_features(depth_proj)

        # 默认保存可视化
        self.visualize_differential_features(
            rgb_grad_x, rgb_grad_y, rgb_laplacian,
            depth_grad_x, depth_grad_y, depth_laplacian,
            self.layer_name
        )

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

        # 低频补偿
        rgb_low_freq = self.low_freq_pool(rgb_proj)
        depth_low_freq = self.low_freq_pool(depth_proj)
        rgb_weight_smooth = F.avg_pool2d(complementarity_weights[:, 0:1, :, :], kernel_size=3, stride=1, padding=1)
        depth_weight_smooth = F.avg_pool2d(complementarity_weights[:, 1:2, :, :], kernel_size=3, stride=1, padding=1)
        rgb_weight_smooth = rgb_weight_smooth.expand(-1, C, -1, -1)
        depth_weight_smooth = depth_weight_smooth.expand(-1, C, -1, -1)
        combined_low_freq = rgb_weight_smooth * rgb_low_freq + depth_weight_smooth * depth_low_freq

        # 集成
        final_input = torch.cat([fused_features, combined_low_freq], dim=1)
        reconstructed_features = self.integration_net(final_input)
        output = reconstructed_features + rgb_proj + depth_proj

        return output

    def visualize_differential_features(self, rgb_grad_x, rgb_grad_y, rgb_laplacian,
                                        depth_grad_x, depth_grad_y, depth_laplacian,
                                        layer_name):
        """可视化六种差分特征"""
        # 创建保存目录
        save_dir = os.path.join(VISUALIZATION_DIR, layer_name)
        os.makedirs(save_dir, exist_ok=True)

        # 处理张量数据
        def tensor_to_image(tensor):
            """将张量转换为可视化图像"""
            tensor = tensor.detach().cpu()
            if len(tensor.shape) == 4:  # [B, C, H, W]
                tensor = tensor[0].mean(dim=0)  # 取第一个样本，在通道上平均
            elif len(tensor.shape) == 3:  # [C, H, W]
                tensor = tensor.mean(dim=0)
            tensor = tensor.numpy()
            # 归一化到 [0, 1]
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
            return tensor

        # 1. RGB Gradient X
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        rgb_grad_x_img = tensor_to_image(rgb_grad_x)
        im = ax.imshow(rgb_grad_x_img, cmap='viridis')
        ax.set_title(f'{layer_name} - RGB Gradient X (Horizontal Edges)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '01_rgb_gradient_x.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # 2. RGB Gradient Y
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        rgb_grad_y_img = tensor_to_image(rgb_grad_y)
        im = ax.imshow(rgb_grad_y_img, cmap='plasma')
        ax.set_title(f'{layer_name} - RGB Gradient Y (Vertical Edges)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '02_rgb_gradient_y.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # 3. RGB Laplacian
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        rgb_laplacian_img = tensor_to_image(rgb_laplacian)
        im = ax.imshow(rgb_laplacian_img, cmap='hot')
        ax.set_title(f'{layer_name} - RGB Laplacian (Second Derivatives)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '03_rgb_laplacian.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # 4. Depth Gradient X
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        depth_grad_x_img = tensor_to_image(depth_grad_x)
        im = ax.imshow(depth_grad_x_img, cmap='coolwarm')
        ax.set_title(f'{layer_name} - Depth Gradient X (Horizontal Edges)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '04_depth_gradient_x.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # 5. Depth Gradient Y
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        depth_grad_y_img = tensor_to_image(depth_grad_y)
        im = ax.imshow(depth_grad_y_img, cmap='twilight')
        ax.set_title(f'{layer_name} - Depth Gradient Y (Vertical Edges)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '05_depth_gradient_y.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # 6. Depth Laplacian
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        depth_laplacian_img = tensor_to_image(depth_laplacian)
        im = ax.imshow(depth_laplacian_img, cmap='RdYlBu')
        ax.set_title(f'{layer_name} - Depth Laplacian (Second Derivatives)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '06_depth_laplacian.png'), bbox_inches='tight', dpi=100)
        plt.close()

        # 7. 创建汇总对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)

        rgb_grad_x_img = tensor_to_image(rgb_grad_x)
        rgb_grad_y_img = tensor_to_image(rgb_grad_y)
        rgb_laplacian_img = tensor_to_image(rgb_laplacian)
        depth_grad_x_img = tensor_to_image(depth_grad_x)
        depth_grad_y_img = tensor_to_image(depth_grad_y)
        depth_laplacian_img = tensor_to_image(depth_laplacian)

        viz_data = [
            (rgb_grad_x_img, 'RGB Grad X', 'viridis', axes[0, 0]),
            (rgb_grad_y_img, 'RGB Grad Y', 'plasma', axes[0, 1]),
            (rgb_laplacian_img, 'RGB Laplacian', 'hot', axes[0, 2]),
            (depth_grad_x_img, 'Depth Grad X', 'coolwarm', axes[1, 0]),
            (depth_grad_y_img, 'Depth Grad Y', 'twilight', axes[1, 1]),
            (depth_laplacian_img, 'Depth Laplacian', 'RdYlBu', axes[1, 2]),
        ]

        for img, title, cmap, ax in viz_data:
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Intensity')

        fig.suptitle(f'{layer_name} - Six Differential Features', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '00_all_features_combined.png'), bbox_inches='tight', dpi=100)
        plt.close()

        print(f"✓ 已保存 {layer_name} 的可视化图到: {save_dir}")


# ==================== 其他必需的类 ====================

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.low_freq_adapter = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim)
        )
        self.high_freq_adapter = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim)
        )
        self.freq_gate = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=-1)
        freq_split = x_freq.shape[-1] // 2
        low_freq = torch.fft.irfft(
            torch.cat([x_freq[..., :freq_split], torch.zeros_like(x_freq[..., freq_split:])], dim=-1),
            n=x.shape[-1]
        )
        high_freq = x - low_freq
        low_adapted = self.low_freq_adapter(low_freq)
        high_adapted = self.high_freq_adapter(high_freq)
        freq_gate_value = self.freq_gate(low_freq).view(-1, low_freq.size(1), low_freq.size(2))
        freq_gate_value = freq_gate_value.unsqueeze(-1)
        freq_gate_value = freq_gate_value.expand(-1, -1, -1, low_freq.size(3))
        prompt = freq_gate_value * low_adapted + (1 - freq_gate_value) * high_adapted
        return self.block(x + prompt)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
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


class EnhancedFrequencyDecomposer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.low_pass_3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.low_pass_5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.low_pass_7x7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.low_freq_attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Softmax(dim=1)
        )
        self.low_freq_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.high_freq_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True)
            )
            for k in [1, 3, 5]
        ])
        self.high_freq_fusion = nn.Sequential(
            nn.Conv2d(channels // 4 * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_3x3 = self.low_pass_3x3(x)
        low_5x5 = self.low_pass_5x5(x)
        low_7x7 = self.low_pass_7x7(x)
        low_combined = torch.cat([low_3x3, low_5x5, low_7x7], dim=1)
        attention_weights = self.low_freq_attention(low_combined)
        weighted_low = (low_3x3 * attention_weights[:, 0:1] + low_5x5 * attention_weights[:, 1:2] +
                        low_7x7 * attention_weights[:, 2:3])
        high_freq_raw = x - weighted_low
        high_features = [branch(high_freq_raw) for branch in self.high_freq_branches]
        high_combined = torch.cat(high_features, dim=1)
        enhancement_weight = self.high_freq_fusion(high_combined)
        high_freq = high_freq_raw * enhancement_weight
        return weighted_low, high_freq


class LightweightScaleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.low_freq_attention = self._build_attention_branch(channels, 'low')
        self.high_freq_attention = self._build_attention_branch(channels, 'high')
        self.freq_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch // 8),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 1)
            )
            for ch in channels
        ])

    def _build_attention_branch(self, channels, freq_type):
        return nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, ch // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 8, ch, 1),
                nn.Sigmoid()
            )
            for ch in channels
        ])

    def forward(self, freq_decompositions):
        enhanced_freqs = []
        for i, (low, high) in enumerate(freq_decompositions):
            low_att = self.low_freq_attention[i](low)
            high_att = self.high_freq_attention[i](high)
            freq_interaction = self.freq_interaction[i](low * high)
            enhanced_low = low * low_att + freq_interaction * 0.1
            enhanced_high = high * high_att + freq_interaction * 0.1
            enhanced_freqs.append((enhanced_low, enhanced_high))
        return enhanced_freqs


class EnhancedFreqUpsampler(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.low_upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            for _ in range(3)
        ])
        self.high_upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            for _ in range(3)
        ])
        self.low_transforms = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=k // 2, groups=out_ch // 4)
            for k in [1, 3, 5]
        ])
        self.high_transforms = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=k // 2, groups=out_ch // 4)
            for k in [1, 3, 5]
        ])
        self.path_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch * 6, out_ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, 6, 1),
            nn.Softmax(dim=1)
        )
        self.detail_recovery = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1)
        )

    def forward(self, prev_low, prev_high, curr_low, curr_high):
        target_size = curr_low.shape[2:]
        low_ups = []
        high_ups = []
        for i, (low_up, high_up) in enumerate(zip(self.low_upsamplers, self.high_upsamplers)):
            up_low = low_up(prev_low)
            up_high = high_up(prev_high)
            if up_low.shape[2:] != target_size:
                up_low = F.interpolate(up_low, size=target_size, mode='bilinear', align_corners=False)
                up_high = F.interpolate(up_high, size=target_size, mode='bilinear', align_corners=False)
            up_low = self.low_transforms[i](up_low)
            up_high = self.high_transforms[i](up_high)
            low_ups.append(up_low)
            high_ups.append(up_high)

        all_paths = torch.cat(low_ups + high_ups, dim=1)
        path_weights = self.path_selector(all_paths)
        weighted_low = sum(low_ups[i] * path_weights[:, i:i + 1] for i in range(3))
        weighted_high = sum(high_ups[i] * path_weights[:, i + 3:i + 4] for i in range(3))

        fused_low = weighted_low * 0.6 + curr_low * 0.4
        fused_high = weighted_high * 0.6 + curr_high * 0.4

        detail_input = torch.cat([fused_low, fused_high, curr_low, curr_high], dim=1)
        detail_enhancement = self.detail_recovery(detail_input)
        if detail_enhancement.shape[2:] != fused_low.shape[2:]:
            detail_enhancement = F.interpolate(detail_enhancement, size=fused_low.shape[2:],
                                               mode='bilinear', align_corners=False)
        output_low = fused_low + detail_enhancement * 0.3
        output_high = fused_high + detail_enhancement * 0.3
        return output_low, output_high


class AdaptiveFrequencyBalancer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_context = nn.AdaptiveAvgPool2d(1)
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, low_freq, high_freq):
        combined = torch.cat([low_freq, high_freq], dim=1)
        global_ctx = self.global_context(combined)
        weights = self.weight_predictor(global_ctx)
        low_weight, high_weight = weights.split(1, dim=1)
        weighted_low = low_freq * low_weight
        weighted_high = high_freq * high_weight
        final_combined = torch.cat([weighted_low, weighted_high], dim=1)
        final_output = self.final_fusion(final_combined)
        residual = (low_freq + high_freq) * 0.5
        return final_output + residual


class FrequencyGuidedDecoder(nn.Module):
    def __init__(self, channels=[512, 256, 128, 64]):
        super().__init__()
        self.channels = channels
        self.freq_decomposers = nn.ModuleList([
            EnhancedFrequencyDecomposer(ch) for ch in channels
        ])
        self.scale_attention = LightweightScaleAttention(channels)
        self.freq_guided_upsamplers = nn.ModuleList([
            EnhancedFreqUpsampler(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])
        self.freq_balancer = AdaptiveFrequencyBalancer(channels[-1])

    def forward(self, features):
        freq_decompositions = []
        for i, feat in enumerate(features):
            low_freq, high_freq = self.freq_decomposers[i](feat)
            freq_decompositions.append((low_freq, high_freq))

        enhanced_freqs = self.scale_attention(freq_decompositions)
        current_low, current_high = enhanced_freqs[0]
        for i in range(len(enhanced_freqs) - 1):
            next_low, next_high = enhanced_freqs[i + 1]
            current_low, current_high = self.freq_guided_upsamplers[i](
                current_low, current_high, next_low, next_high
            )

        final_feat = self.freq_balancer(current_low, current_high)
        return final_feat



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
            blocks.append(Adapter(block))
        self.encoder.blocks = nn.Sequential(*blocks)

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

        # 关键改动：添加layer_name参数
        self.dpf4 = DifferentialPerceptionFusion(512, layer_name='Layer_4_512ch')
        self.dpf3 = DifferentialPerceptionFusion(256, layer_name='Layer_3_256ch')
        self.dpf2 = DifferentialPerceptionFusion(128, layer_name='Layer_2_128ch')
        self.dpf1 = DifferentialPerceptionFusion(64, layer_name='Layer_1_64ch')

        self.freq_decoder = FrequencyGuidedDecoder()

    def forward(self, x, d):
        d = d.repeat(1, 3, 1, 1)
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

        return out1, out2, out3, out4


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs('differential_features_viz', exist_ok=True)

    print("=" * 60)
    print("初始化模型...")
    print("=" * 60)

    with torch.no_grad():
        model = SAM2().cuda()
        x = torch.randn(1, 3, 512, 512).cuda()
        d = torch.randn([1, 1, 512, 512]).cuda()

        print("\n执行前向传播（会自动保存可视化图）...")
        out1, out2, out3, out4 = model(x, d)

        print("\n" + "=" * 60)
        print("✓ 前向传播完成！")
        print("✓ 可视化图已保存到: ./differential_features_viz/")
        print("=" * 60)

        # 计算模型的参数量和FLOPs
        print("\n计算模型性能指标...")
        params, flops = profile(model, inputs=(x, d))
        flops, params = clever_format([flops, params], "%.2f")
        print(f"FLOPs: {flops}, Params: {params}")
        print("=" * 60)


