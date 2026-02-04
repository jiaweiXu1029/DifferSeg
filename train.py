import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset1 import FullDataset
from SAM2UNet import SAM2
from tqdm import tqdm
from utils.AvgMeter import AvgMeter
import glob

parser = argparse.ArgumentParser("SAM2-UNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
# 训练集路径
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
# 验证集路径
parser.add_argument("--val_datasets_dir", type=str, required=True,
                    help="directory containing multiple validation datasets")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
# 恢复训练参数
parser.add_argument("--resume", type=str, default=None,
                    help="path to the checkpoint to resume training from")
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def validate_dataset(model, val_loader, device, dataset_name=""):
    """
    对单个数据集执行验证并计算平均绝对误差(MAE)
    """
    model.eval()
    mae_meter = AvgMeter()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"验证 {dataset_name}")):
            x = batch['image'].to(device)
            target = batch['label'].to(device)

            # 前向传播(只需要最终输出进行评估)
            pred0, pred1, pred2,pred3 = model(x)

            # 应用sigmoid获取概率(0到1)
            pred = torch.sigmoid(pred0)

            # 计算批次的MAE
            mae = torch.mean(torch.abs(pred - target))

            # 更新平均计量器
            mae_meter.update(mae.item(), x.size(0))

    print(f"{dataset_name} 验证完成。平均MAE: {mae_meter.avg:.4f}")
    return mae_meter.avg


def validate(model, val_datasets_dir, device, batch_size=12, num_workers=8):

    model.eval()
    print("开始验证所有数据集...")

    # 获取所有数据集文件夹
    dataset_folders = [f for f in os.listdir(val_datasets_dir)
                       if os.path.isdir(os.path.join(val_datasets_dir, f))]

    if not dataset_folders:
        print(f"警告: 在 {val_datasets_dir} 中未找到数据集文件夹")
        return float('inf')

    all_maes = []
    dataset_results = {}

    for dataset_folder in dataset_folders:
        dataset_path = os.path.join(val_datasets_dir, dataset_folder)
        image_path = os.path.join(dataset_path, "image")
        mask_path = os.path.join(dataset_path, "mask")

        # 检查路径是否存在
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"警告: {dataset_folder} 中缺少images或masks文件夹，跳过")
            continue

        try:
            # 加载数据集
            val_dataset = FullDataset(image_path, mask_path, 512, mode='val')
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers)

            print(f"验证数据集 '{dataset_folder}' (大小: {len(val_dataset)})")

            # 验证当前数据集
            mae = validate_dataset(model, val_dataloader, device, dataset_folder)
            all_maes.append(mae)
            dataset_results[dataset_folder] = mae

        except Exception as e:
            print(f"验证数据集 '{dataset_folder}' 时出错: {str(e)}")

    # 计算所有数据集的平均MAE
    if all_maes:
        avg_mae = sum(all_maes) / len(all_maes)
        print("\n所有数据集的验证结果:")
        for dataset, mae in dataset_results.items():
            print(f"- {dataset}: MAE = {mae:.4f}")
        print(f"平均MAE (所有数据集): {avg_mae:.4f}")
        return avg_mae
    else:
        print("没有成功验证任何数据集")
        return float('inf')


def main(args):
    # 数据集和数据加载器设置
    print("加载训练数据集...")
    # 训练集
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 512, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print(f"训练集大小: {len(train_dataset)}")

    # 模型、优化器、调度器设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = SAM2(args.hiera_path)
    model.to(device)
    print("模型已加载.")

    optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    print("优化器和调度器设置完成.")

    os.makedirs(args.save_path, exist_ok=True)
    print(f"检查点将保存至: {args.save_path}")

    # 恢复训练逻辑
    start_epoch = 0
    best_val_mae = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_mae = checkpoint.get('best_val_mae', float('inf'))
            # 重新初始化调度器以确保与加载的优化器状态同步
            scheduler = CosineAnnealingLR(optim, args.epoch - start_epoch, eta_min=1.0e-7)
            print(f"从epoch {checkpoint['epoch']} 恢复训练，最佳MAE为 {best_val_mae:.4f}")
        else:
            print(f"在'{args.resume}'未找到检查点。从头开始训练。")

    # 训练和验证循环
    for epoch in range(start_epoch, args.epoch):
        # 训练阶段
        model.train()
        train_loss_meter = AvgMeter()
        print(f"\nEpoch {epoch + 1}/{args.epoch}")
        print("-" * 20)

        for i, batch in enumerate(tqdm(train_dataloader, desc="训练中")):
            x = batch['image'].to(device)
            target = batch['label'].to(device)

            optim.zero_grad()

            # 前向传播
            pred0, pred1, pred2,pred3 = model(x)

            # 计算损失
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss3 = structure_loss(pred3, target)
            loss = loss0 + loss1 + loss2+loss3

            loss.backward()
            optim.step()

            train_loss_meter.update(loss.item(), x.size(0))

        scheduler.step()
        print(f"Epoch {epoch + 1} 训练完成。平均训练损失: {train_loss_meter.avg:.4f}")

        # 验证阶段 - 遍历所有验证数据集
        print("\n开始验证阶段...")
        val_mae = validate(model, args.val_datasets_dir, device, args.batch_size)
        model.train()  # 验证后将模型设回训练模式

        # 检查点保存
        # 始终保存最新的模型
        checkpoint_path_latest = os.path.join(args.save_path, 'SAM2-UNet-latest.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'best_val_mae': best_val_mae,
        }, checkpoint_path_latest)
        print('[保存最新快照:]', checkpoint_path_latest)

        # 保存具有最佳验证MAE的模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint_path_best = os.path.join(args.save_path, 'SAM2-UNet-best_mae.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'best_val_mae': best_val_mae,
            }, checkpoint_path_best)
            print(f'[保存最佳MAE快照:] {checkpoint_path_best} (MAE: {best_val_mae:.4f})')


    print("\n训练完成.")
    print(f"达到的最佳验证MAE: {best_val_mae:.4f}")


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
    # seed_torch(1024)
    main(args)
