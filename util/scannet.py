import os
import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_scannet
import glob

class Scannetv2(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1, labeled_point=0.1):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point = labeled_point
        self.num_classes = 20  
        self.ignore_label = 20  

        if split in ["train", "val"]:
            search_path = os.path.join(data_root, "data", split, "*.pth")
        else:
            raise ValueError(f"Invalid split: {split}")
            
        self.data_list = glob.glob(search_path)
        
        if len(self.data_list) == 0:
            raise RuntimeError(f"No data found in {search_path}")
            
        print(f"Found {len(self.data_list)} samples in {split} set at {search_path}")
        print(f"First file: {self.data_list[0]}")
        
    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        
        try:
            data = torch.load(data_path)
            coord, feat = data[0], data[1]
            label = data[2]
            
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            label = label.long()  
            
            label[label < 0] = self.ignore_label
            label[label >= self.num_classes] = self.ignore_label
            
            if self.split=='train' or self.split=='trainval':
                if '%' in self.labeled_point:
                    r = float(self.labeled_point[:-1]) / 100
                    num_pts = len(coord)
                    num_with_anno = max(int(num_pts * r), 1)
                    num_without_anno = num_pts - num_with_anno
                    idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                    label[idx_without_anno] = self.ignore_label
            
            if isinstance(coord, np.ndarray):
                coord = torch.from_numpy(coord).float().contiguous()
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).float().contiguous()
                
            coord, feat, label = data_prepare_scannet(coord, feat, label, 
                                                    self.split,
                                                    self.voxel_size, 
                                                    self.voxel_max,
                                                    self.transform,
                                                    self.shuffle_index)
                                                    
            coord = coord.float().contiguous()
            feat = feat.float().contiguous()
            label = label.long().contiguous()
            
            return coord, feat, label
            
        except Exception as e:
            print(f"Error loading {data_path}: {str(e)}")
            dummy_size = 100 
            return (torch.zeros(dummy_size, 3).float().contiguous(),
                   torch.zeros(dummy_size, 3).float().contiguous(),
                   torch.full((dummy_size,), self.ignore_label, dtype=torch.long).contiguous())

    def __len__(self):
        return len(self.data_list) * self.loop