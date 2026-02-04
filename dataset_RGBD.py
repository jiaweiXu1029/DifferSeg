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
        image, depth, label = data['image'], data['depth'], data['label']
        return {'image': F.to_tensor(image), 'depth': F.to_tensor(depth), 'label': F.to_tensor(label)}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, depth, label = data['image'], data['depth'], data['label']

        return {
            'image': F.resize(image, self.size),
            'depth': F.resize(depth, self.size),
            'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)
        }


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, depth, label = data['image'], data['depth'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'depth': F.hflip(depth), 'label': F.hflip(label)}

        return {'image': image, 'depth': depth, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, depth, label = data['image'], data['depth'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'depth': F.vflip(depth), 'label': F.vflip(label)}

        return {'image': image, 'depth': depth, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], depth_mean=[0.5], depth_std=[0.5]):
        self.mean = mean
        self.std = std
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        depth = F.normalize(depth, self.depth_mean, self.depth_std)
        return {'image': image, 'depth': depth, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, depth_root, gt_root, size, mode):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        print(f"加载的图像数量: {len(self.images)}")
        print(f"加载的深度图数量: {len(self.depths)}")
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
        depth = self.depth_loader(self.depths[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'depth': depth, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')  # 深度图通常为单通道

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, image_root, depth_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.depth_transform = transforms.Compose([
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

        depth = self.depth_loader(self.depths[self.index])
        depth = self.depth_transform(depth).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, depth, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')  # 深度图通常为单通道

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
