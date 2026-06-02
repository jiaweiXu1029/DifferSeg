# DifferSeg

**Towards Diverse Multimodal Binary Segmentation via Differential Perception and Frequency Guidance**

## Authors
Qiangqiang Zhou, Jiawei Xu, Yong Chen, Dandan Zhu, Yugen Yi, Xiaoqi Zhao
<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/DifferSeg/TCSVT_figure/DifferSeg_01(2).png" width="800"><br>
  <em>Figure 1. Overall architecture of the proposed TP-Seg framework for unified medical lesion segmentation. Each input image, together with its task embedding, is processed by the task-conditioned routing block (TCRB) for feature extraction, followed by the prototype-guided task decoder (PGTD) for task-aware decoding and final lesion prediction.</em>
</p>
## Overview
In many binary segmentation tasks, most multimodal methods rely on fixed feature concatenation for cross-modal interaction and straightforward decoder designs dominated by low-frequency semantics.
However, they ignore two key challenges: one is the lack of an adaptive mechanism to handle modality discrepancies and complementarity, and the other is the absence of an efficient decoding strategy to balance both high- and low-frequency representations.
In this work, we propose a simple yet general multimodal binary segmentation framework, termed DifferSeg, to address both problems simultaneously.
With the help of the differential perception fusion (DPF) module, DifferSeg employs learnable differential operators to adaptively align multimodal features and enhance their complementarity through residual fusion, effectively mitigating modality mismatch and fusion redundancy.
In addition, we design a frequency-guided decoder (FGD) that builds cross-frequency interactions and multi-path upsampling to maintain consistency between detailed high-frequency structures and semantic low-frequency representations, ensuring fine-grained boundary recovery and noise suppression.
Benefiting from these designs, DifferSeg can be easily generalized to diverse binary segmentation tasks, including both natural and medical modalities. Without bells and whistles, it consistently surpasses 67 state-of-the-art methods across 29 public datasets involving 18 downstream tasks, demonstrating superior generalization and segmentation accuracy.

## Requirements
- Python >= 3.8
- PyTorch >= 1.10
- Other dependencies (see `requirements.txt`)

```bash
pip install -r requirements.txt
## Prediction
Pretrained Models
RGB_SOD.pth
RGB-D_SOD.pth
......
## model.pth
