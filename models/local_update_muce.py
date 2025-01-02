#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch import optim

from models.supconloss import SupConLoss
from models.test import accuracy
from utils.dataset import DatasetSplit

class LocalUpdate_muce(object):
    def __init__(self, args, node_id, dataset_train=None, idxs_train=None, dataset_valid_s=None, inds_valid_s=None, dataset_valid_t=None, inds_valid_t=None):
        self.args = args
        self.node_id = node_id
        self.criterion = SupConLoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.local_data_train = DataLoader(DatasetSplit(dataset_train, idxs_train), batch_size=int(self.args.local_bs/2), shuffle=True)
        self.local_data_valid_s = DataLoader(DatasetSplit(dataset_valid_s, inds_valid_s), batch_size=self.args.test_bs, shuffle=True)
        self.local_data_valid_t = DataLoader(DatasetSplit(dataset_valid_t, inds_valid_t), batch_size=self.args.test_bs, shuffle=True)

        # print("node: ", node_id)
        # print(idxs_train)

    def train(self, net):

        # train and update
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer =  optim.Adam(net.parameters(), lr=self.args.lr,  weight_decay=self.args.weight_decay)

        epoch_loss_sup = []
        epoch_loss_ce = []
        for iter in range(self.args.local_ep):
            batch_loss_sup = []
            batch_loss_ce = []
            for batch_idx, (flows, labels) in enumerate(self.local_data_train):
                flows, labels = flows.float().to(self.args.device), labels.long().to(self.args.device)
                idx_list = np.arange(flows.size()[0])

                for idx in range(flows.size()[0]):
                    [s_idx] = np.random.choice(idx_list, 1, replace=False)

                    lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
                    while lam < 1e-1:
                        lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
                    
                    if lam > 0.5:
                        flow_tem = flows[s_idx]
                        label_tem = labels[s_idx]
                    else:
                        flow_tem = flows[idx]
                        label_tem = labels[idx]

                    flow_aug = lam * flows[idx] + (1-lam) * flows[s_idx]
                    
                    if idx == 0:
                        flows_ori = flow_tem
                        flows_aug = flow_aug
                        labels_aug = label_tem
                    else:
                        flows_ori = torch.vstack((flows_ori, flow_tem))
                        flows_aug = torch.vstack((flows_aug, flow_aug))
                        labels_aug = torch.vstack((labels_aug, label_tem))
                
                flows_ori = flows_ori.float().to(self.args.device)
                flows_aug = flows_aug.float().to(self.args.device)
                labels_aug = labels_aug.to(self.args.device)
                labels_aug = labels_aug.squeeze()
                
                result_flows = net(flows_aug)[1]
                loss_ce1 = self.entropy_loss(result_flows, labels_aug)

                loss_sup = 0

                result_flows = net(flows)[1]
                loss_ce2 = self.entropy_loss(result_flows, labels)

                loss = (loss_ce1 + loss_ce2) / 2.

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss_sup.append(loss_sup)
                batch_loss_ce.append(loss)
            
            epoch_loss_sup.append(sum(batch_loss_sup) / len(batch_loss_sup))
            epoch_loss_ce.append(sum(batch_loss_ce) / len(batch_loss_ce))
        

        print("--------------------FOR node: %d----------------------" % self.node_id)

        (precision, recall, f1_train, acc) = accuracy(self.args, net, self.local_data_train)
        print("train result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_train, acc))

        (precision, recall, f1_s, acc) = accuracy(self.args,net, self.local_data_valid_s)
        print("s valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_s, acc))

        (precision, recall, f1_t, acc) = accuracy(self.args, net, self.local_data_valid_t)
        print("t valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_t, acc))

        
        return net.cpu().state_dict(), sum(epoch_loss_sup) / len(epoch_loss_sup), sum(epoch_loss_ce) / len(epoch_loss_ce), f1_s, f1_t

