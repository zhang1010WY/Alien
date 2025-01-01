# coding=utf-8
from torch.utils.data import Dataset
import numpy as np


class dataset(object):
    def __init__(self, data_dir):

        self.x = np.loadtxt(data_dir + "x")
        self.labels = np.loadtxt(data_dir + "y")

        print("loading dataset: ", data_dir)
        print("loading x size: ",self.x.shape)
        print("loading labels size: ",self.labels.shape)

    def __getitem__(self, index):

        flow = self.x[index]
        label = self.labels[index]
        
        return flow, label

    def __len__(self):
        
        return len(self.labels)
    


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        flow, label = self.dataset[self.idxs[item]]
        return flow, label
