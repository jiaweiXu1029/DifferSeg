# DifferSeg

**Towards Diverse Multimodal Binary Segmentation via Differential Perception and Frequency Guidance**

## Authors
Qiangqiang Zhou, Jiawei Xu, Yong Chen, Dandan Zhu, Yugen Yi, Xiaoqi Zhao

## Overview
In many binary segmentation tasks, most multimodal methods rely on fixed feature concatenation for cross-modal interaction and straightforward decoder designs dominated by low-frequency semantics.
However, they ignore two key challenges: one is the lack of an adaptive mechanism to handle modality discrepancies and complementarity, and the other is the absence of an efficient decoding strategy to balance both high- and low-frequency representations.
In this work, we propose a simple yet general multimodal binary segmentation framework, termed DifferSeg, to address both problems simultaneously.
With the help of the differential perception fusion (DPF) module, DifferSeg employs learnable differential operators to adaptively align multimodal features and enhance their complementarity through residual fusion, effectively mitigating modality mismatch and fusion redundancy.
In addition, we design a frequency-guided decoder (FGD) that builds cross-frequency interactions and multi-path upsampling to maintain consistency between detailed high-frequency structures and semantic low-frequency representations, ensuring fine-grained boundary recovery and noise suppression.
Benefiting from these designs, DifferSeg can be easily generalized to diverse binary segmentation tasks, including both natural and medical modalities. Without bells and whistles, it consistently surpasses 67 state-of-the-art methods across 29 public datasets involving 18 downstream tasks, demonstrating superior generalization and segmentation accuracy. Code and pretrained models will be available at the
\href{https://github.com/jiaweiXu1029/DifferSeg}{\textcolor{myPink}{link}}

## Requirements
- Python >= 3.8
- PyTorch >= 1.10
- Other dependencies (see `requirements.txt`)

```bash
pip install -r requirements.txt

Pretrained Models
RGB_SOD.pth
RGB-D_SOD.pth
......
