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
from models.core import Smooth
from models.loss_transfer import TransferLoss

class LocalUpdateLocalWithoutRadiusDistance(object):
    def __init__(self, args, node_id, dataset_train=None, idxs_train=None, dataset_valid_s=None, inds_valid_s=None, dataset_valid_t=None, inds_valid_t=None, normal_radius=0, abnormal_radius=0, net_last_global=None):
        self.args = args
        self.node_id = node_id
        self.criterion = SupConLoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()
        self.local_data_train = DataLoader(DatasetSplit(dataset_train, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        self.local_data_valid_s = DataLoader(DatasetSplit(dataset_valid_s, inds_valid_s), batch_size=self.args.test_bs, shuffle=True)
        self.local_data_valid_t = DataLoader(DatasetSplit(dataset_valid_t, inds_valid_t), batch_size=self.args.test_bs, shuffle=True)

        self.noise_sd = args.noise_sd
        self.normal_radius = normal_radius
        self.abnormal_radius = abnormal_radius
        self.net_last_global = net_last_global.cpu().state_dict()


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

                if self.args.epochs == 0:
                    radius = 0
                else:
                    base_classifier = net
                    # base_classifier = base_classifier.to(self.args.device)
                    # # create the smooothed classifier g
                    # smoothed_classifier = Smooth(self.args, base_classifier, 2, self.args.sigma)
                    # dataset_train_for_certify = flows


                # flows : torch.Size([128, 43])  labels : torch.Size([128])

                flows, labels = flows.float().to(self.args.device), labels.long().to(self.args.device)

                flows = flows + torch.randn_like(flows) * self.noise_sd

                all_idx_list = np.arange(flows.size()[0])

                anomaly_idx_list = []
                normal_idx_list = []
                for idx in range(flows.size()[0]):
                    if labels[idx] == 1:
                        anomaly_idx_list.append(idx)
                    else:
                        normal_idx_list.append(idx)
                anomaly_idx_list = np.array(anomaly_idx_list)
                normal_idx_list = np.array(normal_idx_list)

                first = True
                left_flows_batch_list = []
                right_flows_batch_list = []
                labels_batch_list = []
                counterpart_count = 0
                for idx in range(flows.size()[0]):

                    if labels[idx] == 0:
                        aug_num = 1
                        self_idx_list = normal_idx_list
                        counterpart_idx_list = anomaly_idx_list
                    else:
                        aug_num = self.args.aug_abnormal
                        self_idx_list = anomaly_idx_list
                        counterpart_idx_list = normal_idx_list

                    # augment self class and add to the current batch list
                    [self_random_idx] = np.random.choice(self_idx_list, 1, replace=False)
                    lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
                    while lam < 1e-1:
                        lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)
                    # print('<><><><><><><><> lam <><><><><><><><> : ', lam)

                    flow_self_aug = lam * flows[idx] + (1 - lam) * flows[self_random_idx]
                    label_self_aug = labels[idx]

                    left_flows_batch_list.append(flows[idx])
                    right_flows_batch_list.append(flow_self_aug)
                    labels_batch_list.append(label_self_aug)



                    if counterpart_idx_list.tolist():
                        counterpart_count += 1
                        [counterpart_random_idx_0] = np.random.choice(counterpart_idx_list, 1, replace=False)
                        [counterpart_random_idx_1] = np.random.choice(counterpart_idx_list, 1, replace=False)

                        flow_counterpart_aug_0 = lam * flows[idx] + (1 - lam) * flows[counterpart_random_idx_0]
                        flow_counterpart_aug_1 = flow_counterpart_aug_0 + torch.randn_like(flow_counterpart_aug_0) * self.noise_sd

                        if lam > 0.5:
                            label_counterpart_aug = labels[idx]
                        if lam < 0.5:
                            label_counterpart_aug = labels[counterpart_random_idx_0]


                        left_flows_batch_list.append(flow_counterpart_aug_0)
                        right_flows_batch_list.append(flow_counterpart_aug_1)
                        labels_batch_list.append(label_counterpart_aug)



                left_flows_batch_list_tensor = torch.tensor([item.cpu().detach().numpy() for item in left_flows_batch_list])
                right_flows_batch_list_tensor = torch.tensor([item.cpu().detach().numpy() for item in right_flows_batch_list])
                labels_batch_list_tensor = torch.tensor(labels_batch_list)

                left_flows_batch_list_tensor = left_flows_batch_list_tensor.float().to(self.args.device)
                right_flows_batch_list_tensor = right_flows_batch_list_tensor.float().to(self.args.device)
                labels_batch_list_tensor = labels_batch_list_tensor.to(self.args.device)
                labels_batch_list_tensor = labels_batch_list_tensor.squeeze()



                left_network_return = net(left_flows_batch_list_tensor)
                right_network_return = net(right_flows_batch_list_tensor)

                left_encode_feature = left_network_return[0]
                right_encode_feature = right_network_return[0]

                left_classify_result = left_network_return[1]
                right_classify_result = right_network_return[1]


                feature_unsuqeeze = torch.cat([left_encode_feature.unsqueeze(1), right_encode_feature.unsqueeze(1)], dim=1)
                # print('>>>>>>>>>>>feature_unsuqeeze.shape>>>>>>>>>>>> : ', feature_unsuqeeze.shape)
                # >> >> >> >> >> > feature_unsuqeeze.shape >> >> >> >> >> >>: torch.Size([256, 2, 32])

                loss_sup = self.criterion(feature_unsuqeeze, labels_batch_list_tensor, device=self.args.device)

                # result_flows = net(flows)[1]
                # all_classify_result = torch.vstack((left_classify_result, right_classify_result))
                loss_ce = self.entropy_loss(left_classify_result, labels_batch_list_tensor)

                criterion_transder = TransferLoss(
                    loss_type='cosine', input_dim=len(self.net_last_global.keys()))

                global_net_list = list(self.net_last_global.values())
                current_local_net_list = list(net.cpu().state_dict().values())

                loss_transfer = torch.zeros((1,)).to(self.args.device)
                for j in range(len(global_net_list)):
                    loss_trans = criterion_transder.compute(global_net_list[j].float(), current_local_net_list[j].float())
                    loss_transfer = loss_transfer + loss_trans

                loss = loss_sup + self.args.beta * loss_transfer
                # loss = loss_sup
                net.to(self.args.device)
                # print('optimizer.zero_grad() : ', optimizer)
                optimizer.zero_grad()
                loss.backward()
                # print('loss : ', loss)
                # print('optimizer .step(): ', optimizer)
                optimizer.step()

                batch_loss_sup.append(loss_sup)
                batch_loss_ce.append(loss_ce)
            
            epoch_loss_sup.append(sum(batch_loss_sup) / len(batch_loss_sup))
            epoch_loss_ce.append(sum(batch_loss_ce) / len(batch_loss_ce))


        print("--------------------FOR node: %d----------------------" % self.node_id)

        (precision, recall, f1_train, acc) = accuracy(self.args, net, self.local_data_train)
        print("train result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_train, acc))

        (precision, recall, f1_s, acc) = accuracy(self.args, net, self.local_data_valid_s)
        print("s valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_s, acc))

        (precision, recall, f1_t, acc) = accuracy(self.args, net, self.local_data_valid_t)
        print("t valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_t, acc))

        
        return net.cpu().state_dict(), sum(epoch_loss_sup) / len(epoch_loss_sup), sum(epoch_loss_ce) / len(epoch_loss_ce), f1_s, f1_t

