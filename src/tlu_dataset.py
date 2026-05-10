import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import json
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

class TLUStatesDataset(data.Dataset):
    def __init__(self, mode='train', root='.', transform=None, image_size=84):
        '''
        Args:
            mode: 'train', 'val', or 'test'
            root: Root path. Expects `split.json` and `tlu-states/images` inside or relative to it.
            image_size: Target size for resizing images
        '''
        super(TLUStatesDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.image_size = image_size
        
        # Determine paths
        self.split_file = os.path.join(root, 'split.json')
        self.images_dir = root
        
        # Define normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        box_size = int(image_size * 1.15)

        # Cấu hình transform theo phong cách AGNN
        if mode == 'train':
            # Phần cố định để cache: Resize
            self.transform = transforms.Resize((box_size, box_size))
            # Phần ngẫu nhiên áp dụng mỗi lần lấy ảnh: Augmentation
            self.aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                transforms.ToTensor(),
                normalize
            ])
        else: # val or test
            # Deterministic hoàn toàn cho val/test
            self.transform = transforms.Compose([
                transforms.Resize((box_size, box_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize
            ])
            self.aug_transform = None

        if not os.path.exists(self.split_file):
            raise RuntimeError(f'Split file not found at {self.split_file}')

        # Load split
        with open(self.split_file, 'r') as f:
            splits = json.load(f)
        
        if mode not in splits:
            raise ValueError(f"Mode {mode} not found in split file.")
        
        self.class_names = splits[mode]
        self.all_items = [] # List of (image_path, class_index)
        
        # Index classes (0 to N-1)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Scan folders
        for cls_name in self.class_names:
            cls_folder = os.path.join(self.images_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
                
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.all_items.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))

        # Create label2ind mapping for easier sampling
        self.label2ind = {}
        for idx, (_, target) in enumerate(self.all_items):
            if target not in self.label2ind:
                self.label2ind[target] = []
            self.label2ind[target].append(idx)
        
        self.full_class_list = list(self.label2ind.keys())
        self.data_size = (3, image_size, image_size)
        self.y = [target for _, target in self.all_items]

        # Caching logic (Tối ưu theo AGNN)
        self.cache = []
        print(f"==> Pre-loading {mode} dataset into RAM (AGNN-style)...")
        for img_path, target in tqdm(self.all_items, desc=f"Caching {mode}"):
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                # Train: cache PIL sau resize. Val/Test: cache Tensor hoàn chỉnh.
                self.cache.append(image)
            except Exception as e:
                print(f"Error caching {img_path}: {e}")
                self.cache.append(None)

    def __getitem__(self, idx):
        if self.cache and self.cache[idx] is not None:
            image = self.cache[idx]
            # Nếu là tập train, áp dụng augmentation ngẫu nhiên mỗi lần gọi
            if self.mode == 'train' and self.aug_transform:
                return self.aug_transform(image), self.y[idx]
            return image, self.y[idx]
            
        # Fallback nếu không có cache
        img_path, target = self.all_items[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.mode == 'train':
                # Kết hợp cả 2 transform
                image = self.transform(image)
                return self.aug_transform(image), target
            else:
                return self.transform(image), target
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, self.image_size, self.image_size)), target

    def __len__(self):
        return len(self.all_items)

    def _get_pil(self, idx):
        '''Internal helper for sampling logic that might need raw tensors or PIL-like handling'''
        if self.cache:
            return self.cache[idx]
        img_path, _ = self.all_items[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
