import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from SAM2UNet import SAM2
from dataset1 import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam2-unet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--datasets_root", type=str, required=True,
                    help="root directory containing multiple datasets")
parser.add_argument("--save_root", type=str, required=True,
                    help="root directory to save prediction results")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型时传入hiera_path参数
model = SAM2(args.hiera_path).to(device)

# 加载checkpoint (支持两种格式)
checkpoint = torch.load(args.checkpoint, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    # 如果是字典格式并包含model_state_dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    # 如果直接是模型的state_dict
    model.load_state_dict(checkpoint, strict=True)
    print("Loaded model state_dict directly")

model.eval()

# Get all dataset folders
dataset_folders = [f for f in os.listdir(args.datasets_root) if os.path.isdir(os.path.join(args.datasets_root, f))]

for dataset in dataset_folders:
    # Assuming each dataset folder has 'images' and 'masks' subfolders
    test_image_path = os.path.join(args.datasets_root, dataset, 'image')
    test_gt_path = os.path.join(args.datasets_root, dataset, 'mask')

    # Skip if either subfolder doesn't exist
    if not (os.path.exists(test_image_path) and os.path.exists(test_gt_path)):
        print(f"Skipping {dataset}: missing image or mask folder")
        continue

    # Create save directory for this dataset
    save_path = os.path.join(args.save_root, dataset)
    os.makedirs(save_path, exist_ok=True)

    print(f"Processing dataset: {dataset}")

    test_loader = TestDataset(test_image_path, test_gt_path, 512)

    for i in range(test_loader.size):
        with torch.no_grad():
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.to(device)
            res, _, _,_ = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu()
            res = res.numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res * 255).astype(np.uint8)

            print(f"Saving {dataset}/{name}")
            imageio.imsave(os.path.join(save_path, name[:-4] + ".png"), res)

    print(f"Completed dataset: {dataset}")

print("All datasets processed")
