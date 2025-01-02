#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np
from models.loss_transfer import TransferLoss
from torch import optim

def FedOurDistance(args, w_clients, net, w_server, f1_clients, stepsize, metric, dp, alpha):
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
    weighted_local_parameters = copy.deepcopy(w_clients)
    for k in w_server.keys(): # k is the layer name of the model
        if w_next[k].size() == torch.Size([]):
            continue
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()

    for k in w_next.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        for i in range(0, len(w_clients)):
            # print('w_server of k layer:', w_server[k].size())
            # print('w_clients of k layer:', w_clients[i][k].size())
            # weight of each layer, compute the L2_distance of each local layer and global layer
            att[k][i] = torch.from_numpy(np.array(np.linalg.norm(w_server[k].numpy()-w_clients[i][k].numpy(), ord=metric)))

    # compute the weight of each layer of each clients, softmax att + f_1 score
    for k in w_next.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        att[k] = F.softmax(att[k], dim=0)
        # print("params: ", str(k))
        # print("att weight: ", att[k])
        att[k] = att[k] + torch.mul(att_f1, alpha)
        # print("att weight: ", att[k])
        att[k] = torch.mul(att[k], 1.0/(1+alpha))
        # print("att weight: ", att[k])

    for k in w_next.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        # att_weight is the weight parameters of each layer
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            att_weight += torch.mul(w_server[k]-w_clients[i][k], att[k][i])# weighted local parameters
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape), dp) # noise for privacy

    # compute weighted_local_parameters, the weighted local parameters
    for k in w_next.keys():
        if w_next[k].size() == torch.Size([]):
            continue
        for i in range(0, len(w_clients)):
            weighted_local_parameters[i][k] = torch.mul(w_clients[i][k], att[k][i])

    # compute the loss
    criterion_transder = TransferLoss(
        loss_type='cosine', input_dim=len(w_next.keys()))

    # global_net_list = list(w_server.values())
    # current_local_net_list = list(net.cpu().state_dict().values())
    loss_transfer = torch.zeros((1,)).to(args.device)

    for i in range(len(weighted_local_parameters)):
        for j in range(len(weighted_local_parameters)):
            weighted_local_parameters_value_list_i =  list(weighted_local_parameters[i].values())
            weighted_local_parameters_value_list_j = list(weighted_local_parameters[j].values())
            for z in range(len(weighted_local_parameters_value_list_j)):
                loss_trans = criterion_transder.compute(weighted_local_parameters_value_list_i[z].float(), weighted_local_parameters_value_list_j[z].float())
                loss_transfer = loss_transfer + loss_trans

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    loss = loss_transfer
    loss.requires_grad_(True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    return net.cpu().state_dict()
