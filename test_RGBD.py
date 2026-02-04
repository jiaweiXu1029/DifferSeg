import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from sam2unet_RGBD import SAM2
from dataset_RGBD import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--data_root", type=str, required=True,
                    help="root directory containing all datasets")
parser.add_argument("--save_root", type=str, required=True,
                    help="root directory to save results for all datasets")
parser.add_argument("--image_folder", type=str, default="image",
                    help="folder name for images in each dataset")
parser.add_argument("--depth_folder", type=str, default="depth",
                    help="folder name for depth maps in each dataset")
parser.add_argument("--mask_folder", type=str, default="mask",
                    help="folder name for ground truth masks in each dataset")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAM2().to(device)
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()
model.cuda()

# 自动发现数据集
datasets = []
for item in os.listdir(args.data_root):
    dataset_path = os.path.join(args.data_root, item)
    # 检查是否是目录
    if os.path.isdir(dataset_path):
        # 直接检查当前目录是否包含必要的子文件夹
        has_images = os.path.exists(os.path.join(dataset_path, args.image_folder))
        has_depths = os.path.exists(os.path.join(dataset_path, args.depth_folder))
        has_masks = os.path.exists(os.path.join(dataset_path, args.mask_folder))

        if has_images and has_depths and has_masks:
            datasets.append(item)

print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")

# 对每个数据集分别测试
for dataset_name in datasets:
    print(f"\n==== Testing on dataset: {dataset_name} ====")

    # 直接使用数据集目录
    test_image_path = os.path.join(args.data_root, dataset_name, args.image_folder) + "/"
    test_depth_path = os.path.join(args.data_root, dataset_name, args.depth_folder) + "/"
    test_gt_path = os.path.join(args.data_root, dataset_name, args.mask_folder) + "/"
    save_path = os.path.join(args.save_root, dataset_name)

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 加载当前数据集
    test_loader = TestDataset(test_image_path, test_depth_path, test_gt_path, 512)

    print(f"Dataset {dataset_name} contains {test_loader.size} images")

    # 测试当前数据集中的所有图像
    for i in range(test_loader.size):
        with torch.no_grad():
            image, depth, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.to(device)
            depth = depth.to(device)
            res, _, _, _ = model(image, depth)


            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
            res = res.sigmoid().data.cpu()
            res = res.numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)
            # If you want to binarize the prediction results, please uncomment the following three lines.
            # Note that this action will affect the calculation of evaluation metrics.
            # lambda = 0.5
            # res[res >= int(255 * lambda)] = 255
            # res[res < int(255 * lambda)] = 0
            print(f"[{dataset_name}] [{i + 1}/{test_loader.size}] Saving {name}")
            imageio.imsave(os.path.join(save_path, name[:-4] + ".png"), res)

    print(f"Completed testing on dataset: {dataset_name}")

print("\nAll datasets testing completed!")
