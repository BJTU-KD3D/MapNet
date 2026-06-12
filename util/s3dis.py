import os
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
import atexit
import torch
from util.data_util import sa_create, clean_SharedArray, data_prepare  # Add data_prepare import


class S3DIS(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1, labeled_point=0.1):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop
        self.labeled_point = labeled_point
        self.num_classes = list(range(13)) + [255]  # Define num_classes
        
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            clean_SharedArray()
        
        self.shared_arrays = []
        
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        
        if split == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]

        for item in self.data_list:
            try:
                data_path = os.path.join(data_root, item + '.npy')
                if not os.path.exists("/dev/shm/{}".format(item)):
                    data = np.load(data_path)
                    sa_create("shm://{}".format(item), data)
                self.shared_arrays.append(item)
            except Exception as e:
                print(f"Error creating shared memory for {item}: {str(e)}")
                continue
                
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))
        
        atexit.register(self.cleanup)
        
    def cleanup(self):
        for item in self.shared_arrays:
            try:
                SA.delete("shm://{}".format(item))
            except:
                pass

    def __del__(self):
        self.cleanup()

    def __getitem__(self, idx):
        try:
            data_idx = self.data_idx[idx % len(self.data_idx)]
            data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
            
            # labeled point
            if self.split=='train' or self.split=='trainval':
                if '%' in self.labeled_point:
                    r = float(self.labeled_point[:-1]) / 100
                    #print('----------------------')
                    num_pts = data.shape[0]
                    num_with_anno = max(int(num_pts * r), 1)
                    num_without_anno = num_pts - num_with_anno
                    idx_without_anno = np.random.choice(num_pts, num_without_anno, replace=False)
                    data[idx_without_anno,6]=255 #Unlabeled
                    #
                else:
                    for i in self.num_classes:
                        ind_per_class = np.where(data[:,6] == i)[0]  # index of points belongs to a specific class
                        num_per_class = len(ind_per_class)
                        if num_per_class > 0:
                            num_with_anno = int(self.labeled_point) #max(int(num_per_class * r), 1) #int(labeled_point)
                            num_without_anno = num_per_class - num_with_anno
                            idx_without_anno = np.random.choice(ind_per_class, num_without_anno, replace=False)
                            data[idx_without_anno,6] = 255

            coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
            
            coord, feat, label = data_prepare(coord, feat, label, 
                                            self.split,
                                            self.voxel_size,
                                            self.voxel_max,
                                            self.transform,
                                            self.shuffle_index)
                                            
            # Ensure the data types and devices are correct
            coord = coord.float()
            feat = feat.float()
            label = label.long()
            
            return coord, feat, label
            
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            # Return a valid default value
            return (torch.zeros(1,3), 
                   torch.zeros(1,3), 
                   torch.zeros(1).long())

    def __len__(self):
        return len(self.data_idx) * self.loop