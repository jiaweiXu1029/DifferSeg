import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms


class ToTensor(object):
    def __call__(self, data):
        image, flow, label = data['image'], data['flow'], data['label']
        return {'image': F.to_tensor(image), 'flow': F.to_tensor(flow), 'label': F.to_tensor(label)}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, flow, label = data['image'], data['flow'], data['label']

        return {
            'image': F.resize(image, self.size),
            'flow': F.resize(flow, self.size),
            'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)
        }


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, flow, label = data['image'], data['flow'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'flow': F.hflip(flow), 'label': F.hflip(label)}

        return {'image': image, 'flow': flow, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, flow, label = data['image'], data['flow'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'flow': F.vflip(flow), 'label': F.vflip(label)}

        return {'image': image, 'flow': flow, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 flow_mean=[0.5, 0.5, 0.5], flow_std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        self.flow_mean = flow_mean
        self.flow_std = flow_std

    def __call__(self, sample):
        image, flow, label = sample['image'], sample['flow'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        flow = F.normalize(flow, self.flow_mean, self.flow_std)
        return {'image': image, 'flow': flow, 'label': label}


class FullDataset(Dataset):
    def __init__(self, base_root, size, mode):
        # 使用整合后的文件结构
        self.image_root = os.path.join(base_root, 'image/')
        self.flow_root = os.path.join(base_root, 'flow/')
        self.gt_root = os.path.join(base_root, 'mask/')

        # 获取文件列表
        self.images = [f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.flows = [f for f in os.listdir(self.flow_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [f for f in os.listdir(self.gt_root) if f.endswith('.png')]

        # 确保文件名匹配
        self.images = sorted(self.images)
        self.flows = sorted(self.flows)
        self.gts = sorted(self.gts)

        print(f"加载的图像数量: {len(self.images)}")
        print(f"加载的光流图数量: {len(self.flows)}")
        print(f"加载的标签数量: {len(self.gts)}")

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(os.path.join(self.image_root, self.images[idx]))
        flow = self.flow_loader(os.path.join(self.flow_root, self.flows[idx]))
        label = self.binary_loader(os.path.join(self.gt_root, self.gts[idx]))
        data = {'image': image, 'flow': flow, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def flow_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')  # 三通道光流

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, base_root, size):
        # 使用整合后的文件结构
        self.image_root = os.path.join(base_root, 'image/')
        self.flow_root = os.path.join(base_root, 'flow/')
        self.gt_root = os.path.join(base_root, 'mask/')

        # 获取文件列表
        self.images = [f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.flows = [f for f in os.listdir(self.flow_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [f for f in os.listdir(self.gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.flows = sorted(self.flows)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.flow_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 修改为三通道
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(os.path.join(self.image_root, self.images[self.index]))
        image = self.transform(image).unsqueeze(0)

        flow = self.flow_loader(os.path.join(self.flow_root, self.flows[self.index]))
        flow = self.flow_transform(flow).unsqueeze(0)

        gt = self.binary_loader(os.path.join(self.gt_root, self.gts[self.index]))
        gt = np.array(gt)

        name = self.images[self.index]

        self.index += 1
        return image, flow, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def flow_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')  # 三通道光流

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
