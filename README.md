# DifferSeg

**Towards Diverse Multimodal Binary Segmentation via Differential Perception and Frequency Guidance(TCSVT 2026)**

## Authors
Qiangqiang Zhou, Jiawei Xu, Yong Chen, Dandan Zhu, Yugen Yi, Xiaoqi Zhao
<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/DifferSeg/main/TCSVT_figure/DifferSeg_01(2).png" width="800"><br>
</p>
## Overview
In many binary segmentation tasks, most multimodal methods rely on fixed feature concatenation for cross-modal interaction and straightforward decoder designs dominated by low-frequency semantics.
However, they ignore two key challenges: one is the lack of an adaptive mechanism to handle modality discrepancies and complementarity, and the other is the absence of an efficient decoding strategy to balance both high- and low-frequency representations.
In this work, we propose a simple yet general multimodal binary segmentation framework, termed DifferSeg, to address both problems simultaneously.
With the help of the differential perception fusion (DPF) module, DifferSeg employs learnable differential operators to adaptively align multimodal features and enhance their complementarity through residual fusion, effectively mitigating modality mismatch and fusion redundancy.
In addition, we design a frequency-guided decoder (FGD) that builds cross-frequency interactions and multi-path upsampling to maintain consistency between detailed high-frequency structures and semantic low-frequency representations, ensuring fine-grained boundary recovery and noise suppression.
Benefiting from these designs, DifferSeg can be easily generalized to diverse binary segmentation tasks, including both natural and medical modalities. Without bells and whistles, it consistently surpasses 67 state-of-the-art methods across 29 public datasets involving 18 downstream tasks, demonstrating superior generalization and segmentation accuracy.
<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/DifferSeg/main/TCSVT_figure/CVPR1.2 (1).png" width="800"><br>
</p>

## Visual Comparison

<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/DifferSeg/main/TCSVT_figure/all (1).png" width="800"><br>
</p>

---
## Dataset
All datasets are publicly available. Please download them as needed for the task.

## Prediction Maps
[link](https://drive.google.com/file/d/1TtJOQhGJlwa7OiQe0Z1ImUXXxJ7HCjKV/view?usp=drive_link).
---

## model.pth
[link](https://drive.google.com/file/d/1XRGXab_Z_6ej-purCmQ0pdmKtLi2dodh/view?usp=drive_link).
---

## Citing DifferSeg

If you find **DifferSeg** useful in your research or work, please consider citing our paper:

```bibtex
@inproceedings{xu2026tp,
  title={TP-Seg: Task-Prototype Framework for Unified Medical Lesion Segmentation},
  author={Xu, Jiawei and Zhou, Qiangqiang and Zhu, Dandan and Chen, Yong and Yi, Yugen and Zhao, Xiaoqi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5452--5462},
  year={2026}
}
