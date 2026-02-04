import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import json
import torch
import torchvision.transforms as transforms

class TLUStatesDataset(data.Dataset):
    def __init__(self, mode='train', root='.', transform=None):
        '''
        Args:
            mode: 'train', 'val', or 'test'
            root: Root path. Expects `split.json` and `tlu-states/images` inside or relative to it.
        '''
        super(TLUStatesDataset, self).__init__()
        self.root = root
        self.mode = mode
        
        # Determine paths
        # Assuming root is like '/content/drive/MyDrive/Do_an_Data'
        self.split_file = os.path.join(root, 'split.json')
        self.images_dir = os.path.join(root, 'tlu-states', 'images')
        
        # If transform is None, use default for ProtoNet
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

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

        # Create self.y for PrototypicalBatchSampler
        self.y = [target for _, target in self.all_items]

    def __getitem__(self, idx):
        img_path, target = self.all_items[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy
            return torch.zeros((3, 84, 84)), target

    def __len__(self):
        return len(self.all_items)
