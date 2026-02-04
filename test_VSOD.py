import argparse
import os
import torch
import imageio
import numpy as np
from PIL import Image
import torch.nn.functional as F
from sam2unet_VSOD import SAM2
from dataset_VSOD import TestDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--data_root", type=str, required=True,
                    help="root directory containing all datasets")
parser.add_argument("--save_root", type=str, required=True,
                    help="root directory to save results for all datasets")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = SAM2().to(device)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

# 直接测试整合后的数据集
# 假设data_root下直接包含CAD和MoCA_Mask两个数据集文件夹
datasets = []
for dataset_name in os.listdir(args.data_root):
    dataset_path = os.path.join(args.data_root, dataset_name)
    if os.path.isdir(dataset_path):
        datasets.append(dataset_name)

print(f"找到 {len(datasets)} 个数据集: {', '.join(datasets)}")

# 对每个数据集进行测试
for dataset_name in datasets:
    print(f"\n==== 测试数据集: {dataset_name} ====")

    # 构建当前数据集的路径
    dataset_path = os.path.join(args.data_root, dataset_name)
    save_path = os.path.join(args.save_root, dataset_name)

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 使用修改后的TestDataset加载数据
    test_loader = TestDataset(dataset_path, 512)

    print(f"数据集 {dataset_name} 包含 {test_loader.size} 张图像")

    # 跳过空数据集
    if test_loader.size == 0:
        print(f"警告: 数据集 {dataset_name} 没有有效图像，跳过")
        continue

    # 测试当前数据集中的所有图像
    for i in tqdm(range(test_loader.size), desc=f"处理 {dataset_name}"):
        with torch.no_grad():
            image, flow, gt, name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            image = image.to(device)
            flow = flow.to(device)
            res, _, _, _ = model(image, flow)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu()
            res = res.numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            # 保存预测结果
            base_name = os.path.splitext(name)[0]
            save_file = os.path.join(save_path, base_name + ".png")
            imageio.imsave(save_file, res)

    print(f"完成数据集测试: {dataset_name}")

print("\n所有数据集测试完成!")
