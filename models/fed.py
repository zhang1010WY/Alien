#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np


def FedOur(w_clients, w_server, f1_clients, stepsize, metric, dp, alpha):
    """
    Attentive aggregation
    :param w_clients: list of client model parameters
    :param w_server: server model parameters
    :param stepsize: step size for aggregation
    :param metric: similarity
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """

    f1_clients = np.array(f1_clients)
    att_f1 = F.softmax(torch.from_numpy(f1_clients), dim=0)
    # print("f1 weight: ", att_f1)

    w_next = copy.deepcopy(w_server)

    att, att_mat = {}, {}
    for k in w_server.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        for i in range(0, len(w_clients)):
            # print('w_server:', w_server[k].size())
            # print('w_clients:', w_clients[i][k].size())
            att[k][i] = torch.from_numpy(np.array(np.linalg.norm(w_server[k].numpy()-w_clients[i][k].numpy(), ord=metric)))
    for k in w_next.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        att[k] = F.softmax(att[k], dim=0)
        print("params: ", str(k))
        # print("att weight: ", att[k])
        att[k] = att[k] + torch.mul(att_f1, alpha)
        # print("att weight: ", att[k])
        att[k] = torch.mul(att[k], 1.0/(1+alpha))
        print("att weight: ", att[k])
    for k in w_next.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape), dp)
    
    return w_next
