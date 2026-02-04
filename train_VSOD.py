import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_VSOD import FullDataset
from sam2unet_VSOD import SAM2
import time
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict
from utils.AvgMeter import AvgMeter

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
# 训练集和测试集根目录
parser.add_argument("--train_path", type=str, required=True,
                    help="path to the training datasets directory")
parser.add_argument("--test_path", type=str, required=True,
                    help="path to the testing datasets directory")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")

parser.add_argument("--epoch", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--resume", type=str, default=None,
                    help="path to the checkpoint to resume training from")
args = parser.parse_args()


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
            flow = batch['flow'].to(device)
            target = batch['label'].to(device)  # Target is the ground truth mask (0 or 1)

            # Forward pass (only need the final output for evaluation)
            # Assuming pred0 is the main output prediction with highest resolution
            pred0, _, _, _ = model(x, flow)

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

    # 直接加载整合后的训练数据集
    train_dataset = FullDataset(args.train_path, 512, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8)
    print(f"Loaded training dataset with {len(train_dataset)} samples")

    # 加载验证数据集 - 直接加载整合后的文件夹
    val_dataloaders = {}

    # 检查测试路径是否存在
    if os.path.isdir(args.test_path):
        # 遍历测试路径下的每个数据集文件夹 (如CAD, MoCA_Mask等)
        for dataset_name in sorted(os.listdir(args.test_path)):
            dataset_path = os.path.join(args.test_path, dataset_name)
            if os.path.isdir(dataset_path):
                try:
                    val_dataset = FullDataset(dataset_path, 512, mode='val')
                    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=8)
                    val_dataloaders[dataset_name] = val_dataloader
                    print(f"Added validation dataset '{dataset_name}' with {len(val_dataset)} samples")
                except Exception as e:
                    print(f"Error loading validation dataset '{dataset_name}': {str(e)}")

    if not val_dataloaders:
        print("Warning: No validation datasets found!")

    # --- Model, Optimizer, Scheduler Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SAM2(args.hiera_path)
    model.to(device)
    print("Model loaded.")

    optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    print("Optimizer and scheduler setup.")

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Saving checkpoints to: {args.save_path}")

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
            scheduler = CosineAnnealingLR(optim, args.epoch - start_epoch, eta_min=1.0e-7)
            print(f"Resumed from epoch {checkpoint['epoch']} with best MAE {best_val_mae:.4f}")
        else:
            print(f"No checkpoint found at '{args.resume}'. Starting training from scratch.")

    # --- Training and Validation Loop ---
    for epoch in range(start_epoch, args.epoch):
        print(f"\nEpoch {epoch + 1}/{args.epoch}")
        print("-" * 20)

        # --- Training Phase ---
        model.train()
        epoch_loss_meter = AvgMeter()  # 整个epoch的平均损失

        print("Training...")
        for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            x = batch['image'].to(device)
            flow = batch['flow'].to(device)
            target = batch['label'].to(device)

            optim.zero_grad()

            # Forward pass
            pred0, pred1, pred2, pred3 = model(x, flow)

            # Calculate loss
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss3 = structure_loss(pred3, target)
            loss = loss0 + loss1 + loss2 + loss3

            loss.backward()
            optim.step()

            # Update loss meter
            epoch_loss_meter.update(loss.item(), x.size(0))

        scheduler.step()  # 每个epoch结束后更新学习率
        print(f"Epoch {epoch + 1} Training finished. Average Train Loss: {epoch_loss_meter.avg:.4f}")

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
            checkpoint_path_latest = os.path.join(args.save_path, 'SAM2-UNet-VCOD-latest.pth')
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
                checkpoint_path_best = os.path.join(args.save_path, 'SAM2-UNet-VCOD-best_mae.pth')
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
            checkpoint_path_latest = os.path.join(args.save_path, 'SAM2-UNet-VCOD-latest.pth')
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


if __name__ == "__main__":
    main(args)
