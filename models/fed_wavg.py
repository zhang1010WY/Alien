#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedWAvg(args, w, idx_list, train_dict_users):
    nums = []
    nums_sum = 0
    for idx in idx_list:
        nums.append(len(train_dict_users[idx]))
        nums_sum += len(train_dict_users[idx])
    nums = [float(i)/nums_sum for i in nums]
    
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k]).cpu()
        for i in range(0, len(w)):
            w_avg[k] += torch.mul(w[i][k], nums[i])
    return w_avg
