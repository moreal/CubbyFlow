from os import listdir, path

import numpy as np

import albumentations as AB
from albumentations.pytorch import ToTensor, ToTensorV2
import torchvision
import torch
from torch.utils.data import random_split
import pytorch_lightning as pl

import open3d as o3d

def get_files(mypath): return [f for f in listdir(mypath)]

class PntCloudData(torch.utils.data.Dataset):
    def __init__(self, from_dir, to_dir, out_op):
        self.from_dir = from_dir
        self.to_dir = to_dir
        self.from_files = get_files(from_dir)
        self.to_files = get_files(to_dir)
        self.out_op = out_op
    
    def __len__(self):
        return len(self.from_files)
    
    def __getitem__(self, index):
        from_data = o3d.io.read_point_cloud(path.join(self.from_dir, self.from_files[index]))
        to_data = o3d.io.read_point_cloud(path.join(self.to_dir, self.to_files[index]))
        return np.asarray(from_data.points).astype(np.float), np.asarray(to_data.points).astype(np.float)

class PntCloudDL(pl.LightningDataModule):
    def __init__(self, from_dir, to_dir, 
        out_op = 'origin',
        train_batchsize=2, val_batchsize=2, test_batchsize=2, workers=0):
        super().__init__()
        self.from_dir = from_dir
        self.to_dir = to_dir
        self.train_batchsize = train_batchsize
        self.val_batchsize = val_batchsize
        self.test_batchsize = test_batchsize
        self.workers = workers
        self.out_op = out_op

    def prepare_data(self):
        # download
        None

    def setup(self, stage=None):
        data = PntCloudData(self.from_dir, self.to_dir, self.out_op)
        ss = int(0.8*len(data))
        self.pnt_train, self.pnt_val = random_split(data, [ss, len(data)-ss])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.pnt_train, batch_size=self.train_batchsize)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.pnt_val, batch_size=self.val_batchsize)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.pnt_val, batch_size=self.test_batchsize)