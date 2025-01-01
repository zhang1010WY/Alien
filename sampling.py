#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from numpy.random import dirichlet
from utils.dataset import dataset


def sample_iid(dataset, num_users):
    """
    Sample I.I.D. client data from flows dataset
    :param dataset:
    :param num_users:
    :return: dict of index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def sample_non_iid(dataset, num_users):
    """
    这里狄利克雷分布的默认参数为10
    """
    ratios = dirichlet([10]*num_users, 2)
    indx_0 = []
    indx_1 = []
    for i in range(len(dataset.labels)):
        if dataset.labels[i] == 0:
            indx_0.append(i)
        elif dataset.labels[i] == 1:
            indx_1.append(i)
        else:
            print("sample non iid error")
    
    length_0 = len(indx_0)
    length_1 = len(indx_1)

    dict_users = {}
    for i in range(num_users):
        select_0 = set(np.random.choice(indx_0, int(ratios[0][i]*length_0),replace=False))
        select_1 = set(np.random.choice(indx_1, int(ratios[1][i]*length_1),replace=False))
        dict_users[i] = select_0.union(select_1)
        indx_0 = list(set(indx_0) - select_0)
        indx_1 = list(set(indx_1) - select_1)
    return dict_users

def sample_all(dataset, num_users):
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = set([i for i in range(len(dataset))])
    return dict_users

    
if __name__ == '__main__':
    root_dir = "/home/liang/DG-FL/dataset/UNSW-NB15-split2/"
    dataset_train = dataset(root_dir + "train/s_train/")
    res = sample_non_iid(dataset_train, 10)
    print(res)


    