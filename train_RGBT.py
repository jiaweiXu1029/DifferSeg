import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_RGBT import FullDataset  # 修改：从 dataset_RGBD 改为 dataset_RGBT
from sam2unet_RGBT import SAM2  # 修改：从 sam2unet_RGBD 改为 sam2unet_RGBT
import time
from tqdm import tqdm
import os
import torch.nn.functional as F
from collections import OrderedDict
import time
from utils.AvgMeter import AvgMeter
import torchvision.transforms as transforms

parser = argparse.ArgumentParser("SAM2")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
# 训练集路径
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the RGB images used to train the model")
parser.add_argument("--train_thermal_path", type=str, required=True,  # 修改：从 depth 改为 thermal
                    help="path to the thermal images used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
# 验证集路径
parser.add_argument("--val_image_path", type=str, default=None,
                    help="path to the RGB images used for validation")
parser.add_argument("--val_thermal_path", type=str, default=None,  # 修改：从 depth 改为 thermal
                    help="path to the thermal images used for validation")
parser.add_argument("--val_mask_path", type=str, default=None,
                    help="path to the mask file for validation")
# 新增验证文件夹路径参数
parser.add_argument("--val_dir", type=str, default=None,
                    help="directory containing multiple validation datasets")

parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--weight_decay", default=1e-3, type=float)
# --- New argument for resuming training ---
parser.add_argument("--resume", type=str, default=None,
                    help="path to the checkpoint to resume training from")
args = parser.parse_args()


def Eval_mae(self):
    print('eval[MAE]:{} dataset with {} method.'.format(
        self.dataset, self.method))
    avg_mae, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:
            if self.cuda:
                pred = trans(pred).cuda()
                gt = trans(gt).cuda()
            else:
                pred = trans(pred)
                gt = trans(gt)
            mea = torch.abs(pred - gt).mean()
            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
        avg_mae /= img_num
        return avg_mae.item()

def structure_loss(pred, mask):
    """
    Structure loss (weighted BCE + weighted IOU) as commonly used in saliency detection.
    Assumes mask is binary (0 or 1).
    """
    # Weighted BCE
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # Weighted IOU
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)  # Adding 1 for numerical stability

    return (wbce + wiou).mean()


def validate(model, val_loader, device, dataset_name="unknown"):
    """
    Performs validation and calculates Mean Absolute Error (MAE).
    """
    model.eval()  # Set model to evaluation mode
    mae_meter = AvgMeter()
    print(f"Starting validation on dataset '{dataset_name}'...")

    with torch.no_grad():  # Disable gradient calculation for validation
        for i, batch in enumerate(tqdm(val_loader, desc=f"Validating {dataset_name}")):
            x = batch['image'].to(device)
            thermal = batch['thermal'].to(device)  # 修改：从 depth 改为 thermal
            target = batch['label'].to(device)  # Target is the ground truth mask (0 or 1)

            # Forward pass (only need the final output for evaluation)
            # Assuming pred0 is the main output prediction with highest resolution
            pred0, _, _, _ = model(x, thermal)  # 修改：传入 thermal 而不是 depth

            # Apply sigmoid to get probabilities (0 to 1)
            pred = torch.sigmoid(pred0)

            # Calculate MAE for the batch
            # MAE = mean(abs(prediction - target))
            mae = torch.mean(torch.abs(pred - target))

            # Update the average meter (value, number of samples)
            mae_meter.update(mae.item(), x.size(0))

    model.train()  # Set model back to training mode
    print(f"Validation on '{dataset_name}' finished. Average MAE: {mae_meter.avg:.4f}")
    return mae_meter.avg


def main(args):
    # --- Dataset and DataLoader Setup ---
    print("Loading datasets...")
    # 训练集
    train_dataset = FullDataset(args.train_image_path, args.train_thermal_path, args.train_mask_path, 512, mode='train')  # 修改：depth改为thermal
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print(f"Train dataset size: {len(train_dataset)}")

    # 验证数据集加载逻辑
    val_dataloaders = {}

    # 检查是否提供了验证目录
    if args.val_dir and os.path.isdir(args.val_dir):
        # 自动扫描验证文件夹中的数据集
        datasets = os.listdir(args.val_dir)
        for dataset in datasets:
            dataset_path = os.path.join(args.val_dir, dataset)
            if os.path.isdir(dataset_path):
                # 直接检查标准文件夹结构
                image_path = os.path.join(dataset_path, 'image')
                if not image_path.endswith('/'):
                    image_path += '/'
                thermal_path = os.path.join(dataset_path, 'thermal')  # 修改：从 depth 改为 thermal
                if not thermal_path.endswith('/'):
                    thermal_path += '/'
                mask_path = os.path.join(dataset_path, 'mask')
                if not mask_path.endswith('/'):
                    mask_path += '/'

                # 检查所有必要的文件夹是否存在
                if os.path.isdir(image_path) and os.path.isdir(thermal_path) and os.path.isdir(mask_path):  # 修改：depth改为thermal
                    try:
                        val_dataset = FullDataset(image_path, thermal_path, mask_path, 512, mode='val')  # 修改：depth改为thermal
                        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=8)
                        val_dataloaders[dataset] = val_dataloader
                        print(f"Added validation dataset '{dataset}' with {len(val_dataset)} samples")
                    except Exception as e:
                        print(f"Error loading dataset '{dataset}': {str(e)}")
                else:
                    print(f"Skipping dataset '{dataset}' - missing required folders. Found: "
                          f"image:{os.path.isdir(image_path)}, thermal:{os.path.isdir(thermal_path)}, mask:{os.path.isdir(mask_path)}")  # 修改：depth改为thermal

    # 如果没有找到验证集或使用传统方式
    if not val_dataloaders and args.val_image_path and args.val_thermal_path and args.val_mask_path:  # 修改：depth改为thermal
        val_dataset = FullDataset(args.val_image_path, args.val_thermal_path, args.val_mask_path, 512, mode='val')  # 修改：depth改为thermal
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        val_dataloaders['default'] = val_dataloader
        print(f"Using default validation dataset with {len(val_dataset)} samples")

    if not val_dataloaders:
        print("Warning: No validation datasets found!")

    # --- Model, Optimizer, Scheduler Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SAM2(args.hiera_path)
    model.to(device)
    print("Model loaded.")

    optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)  # Corrected 'initia_lr' to 'initial_lr'
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    print("Optimizer and scheduler setup.")

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Saving checkpoints to: {args.save_path}")

    # --- Resume Training Logic ---
    start_epoch = 0
    best_val_mae = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_mae = checkpoint['best_val_mae']
            # Reinitialize scheduler to ensure it's in sync with the loaded optimizer state
            # For CosineAnnealingLR, it might be better to re-create it or manually adjust.
            # Here, we re-create it and let it pick up the loaded LR.
            scheduler = CosineAnnealingLR(optim, args.epoch - start_epoch,
                                          eta_min=1.0e-7)  # Adjust T_max for remaining epochs
            print(f"Resumed from epoch {checkpoint['epoch']} with best MAE {best_val_mae:.4f}")
        else:
            print(f"No checkpoint found at '{args.resume}'. Starting training from scratch.")

    # --- Training and Validation Loop ---
    # best_val_mae is now initialized either to inf or from checkpoint
    for epoch in range(start_epoch, args.epoch):
        # --- Training Phase ---
        model.train()  # Set model to training mode
        train_loss_meter = AvgMeter()
        print(f"\nEpoch {epoch + 1}/{args.epoch}")
        print("-" * 20)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            x = batch['image'].to(device)
            thermal = batch['thermal'].to(device)  # 修改：从 depth 改为 thermal
            target = batch['label'].to(device)

            optim.zero_grad()

            # Forward pass for training (gets multiple scale predictions)
            pred0, pred1, pred2, pred3 = model(x, thermal)  # 修改：传入 thermal 而不是 depth

            # Calculate combined loss from multiple scales
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss3 = structure_loss(pred3, target)
            loss = loss0 + loss1 + loss2 + loss3

            loss.backward()
            optim.step()

            train_loss_meter.update(loss.item(), x.size(0))

        scheduler.step()  # Update learning rate

        print(f"Epoch {epoch + 1} Training finished. Average Train Loss: {train_loss_meter.avg:.4f}")

        # --- Validation Phase ---
        if val_dataloaders:
            # 验证阶段 - 对每个数据集分别验证
            all_maes = {}

            for dataset_name, val_loader in val_dataloaders.items():
                val_mae = validate(model, val_loader, device, dataset_name)
                all_maes[dataset_name] = val_mae

            # 计算所有数据集的平均MAE作为总体指标
            avg_mae = sum(all_maes.values()) / len(all_maes)
            print(f"Average validation MAE across all datasets: {avg_mae:.4f}")

            # 保存每个数据集的MAE到日志
            log_path = os.path.join(args.save_path, f'validation_log_epoch_{epoch + 1}.txt')
            with open(log_path, 'w') as f:
                for dataset_name, mae in all_maes.items():
                    f.write(f"{dataset_name}: {mae:.6f}\n")
                f.write(f"Average: {avg_mae:.6f}\n")

            # --- Checkpoint Saving ---
            # Always save the last epoch's model with full state
            checkpoint_path_latest = os.path.join(args.save_path, 'SAM2-UNet-RGBT-latest.pth')  # 修改：RGBD改为RGBT
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'best_val_mae': best_val_mae,
                'dataset_maes': all_maes,
            }, checkpoint_path_latest)
            print('[Saving latest snapshot:]', checkpoint_path_latest)

            # Save model with the best validation MAE
            if avg_mae < best_val_mae:
                best_val_mae = avg_mae
                checkpoint_path_best = os.path.join(args.save_path, 'SAM2-UNet-RGBT-best_mae.pth')  # 修改：RGBD改为RGBT
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'best_val_mae': best_val_mae,
                    'dataset_maes': all_maes,
                }, checkpoint_path_best)
                print(f'[Saving best MAE snapshot:] {checkpoint_path_best} (MAE: {best_val_mae:.4f})')
        else:
            # 如果没有验证集，只保存最新的模型
            checkpoint_path_latest = os.path.join(args.save_path, 'SAM2-UNet-RGBT-latest.pth')  # 修改：RGBD改为RGBT
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'best_val_mae': best_val_mae,
            }, checkpoint_path_latest)
            print('[Saving latest snapshot (no validation):]', checkpoint_path_latest)

    print("\nTraining finished.")
    if val_dataloaders:
        print(f"Best Validation MAE achieved: {best_val_mae:.4f}")


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024) # Uncomment if you need deterministic results
    main(args)
