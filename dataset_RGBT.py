import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):
    def __call__(self, data):
        image, thermal, label = data['image'], data['thermal'], data['label']
        return {'image': F.to_tensor(image), 'thermal': F.to_tensor(thermal), 'label': F.to_tensor(label)}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, thermal, label = data['image'], data['thermal'], data['label']

        return {
            'image': F.resize(image, self.size),
            'thermal': F.resize(thermal, self.size),
            'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)
        }


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, thermal, label = data['image'], data['thermal'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'thermal': F.hflip(thermal), 'label': F.hflip(label)}

        return {'image': image, 'thermal': thermal, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, thermal, label = data['image'], data['thermal'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'thermal': F.vflip(thermal), 'label': F.vflip(label)}

        return {'image': image, 'thermal': thermal, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], thermal_mean=[0.5], thermal_std=[0.5]):
        self.mean = mean
        self.std = std
        self.thermal_mean = thermal_mean
        self.thermal_std = thermal_std

    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        thermal = F.normalize(thermal, self.thermal_mean, self.thermal_std)
        return {'image': image, 'thermal': thermal, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, thermal_root, gt_root, size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.thermals = [thermal_root + f for f in os.listdir(thermal_root) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.thermals = sorted(self.thermals)
        self.gts = sorted(self.gts)
        print(f"加载的图像数量: {len(self.images)}")
        print(f"加载的热力图数量: {len(self.thermals)}")
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
        image = self.rgb_loader(self.images[idx])
        thermal = self.thermal_loader(self.thermals[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'thermal': thermal, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def thermal_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')  # 热力图通常为单通道

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, image_root, thermal_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.thermals = [thermal_root + f for f in os.listdir(thermal_root) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.thermals = sorted(self.thermals)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.thermal_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        thermal = self.thermal_loader(self.thermals[self.index])
        thermal = self.thermal_transform(thermal).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, thermal, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def thermal_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')  # 热力图通常为单通道

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
