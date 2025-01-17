#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
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
import z_backup as bk
import os

class LocalUpdateLocalWithoutRadius(object):
    def __init__(self, args, node_id, dataset_train=None, idxs_train=None, dataset_valid_s=None, inds_valid_s=None, dataset_valid_t=None, inds_valid_t=None, normal_radius=0, abnormal_radius=0, epoch=None):
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
        self.epoch = epoch
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


                if self.args.epochs == 0:
                    local_abnormal_radius = 0
                    local_normal_radius = 0
                    radius = 0
                else:
                    base_classifier = net
                    base_classifier = base_classifier.to(self.args.device)
                    # create the smooothed classifier g
                    smoothed_classifier = Smooth(self.args, base_classifier, 2, self.args.sigma)
                    dataset_train_for_certify = flows


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

                    # compute radius
                    # prediction, radius = smoothed_classifier.certify(flows[idx], self.args.N0, self.args.N, self.args.alpha,
                    #                                                  self.args.certify_batch)
                    # correct = int(prediction == labels[idx].item())

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
                        # flow_counterpart_aug_1 = lam * flows[idx] + (1 - lam) * flows[counterpart_random_idx_1]

                        # L2_distance = np.linalg.norm(flow_counterpart_aug_0.cpu() - flows[idx].cpu())   #item.cpu().detach().numpy()
                        # print('===============================================================')
                        # print('label, prediction, correct,radius, L2_distance: ', labels[idx], prediction, correct, radius, L2_distance)

                        # radius_for_current = radius
                        # if L2_distance <= radius_for_current:
                        #     label_counterpart_aug = labels[idx]
                        # else:
                        #     label_counterpart_aug = labels[counterpart_random_idx_0]

                        left_flows_batch_list.append(flow_counterpart_aug_0)
                        right_flows_batch_list.append(flow_counterpart_aug_1)
                        labels_batch_list.append(label_counterpart_aug)



                    # print('counterpart_count : ', counterpart_count)
                    # else:
                    #     print('counterpart_idx_list is none, current opposite labels is', labels[idx])


                left_flows_batch_list_tensor = torch.tensor([item.cpu().detach().numpy() for item in left_flows_batch_list])
                right_flows_batch_list_tensor = torch.tensor([item.cpu().detach().numpy() for item in right_flows_batch_list])
                labels_batch_list_tensor = torch.tensor(labels_batch_list)

                left_flows_batch_list_tensor = left_flows_batch_list_tensor.float().to(self.args.device)
                right_flows_batch_list_tensor = right_flows_batch_list_tensor.float().to(self.args.device)
                labels_batch_list_tensor = labels_batch_list_tensor.to(self.args.device)
                labels_batch_list_tensor = labels_batch_list_tensor.squeeze()



                # >> >> >> >> >> > len(flows_ori_encode_by_net) >> >> >> >> >> >>: 2
                # >> >> >> >> >> > len(flows_aug_encode_by_net) >> >> >> >> >> >>: 2
                # >> >> >> >> >> > flows_ori_encode_by_net[1].shape >> >> >> >> >> >>: torch.Size([212, 2])
                # >> >> >> >> >> > flows_aug_encode_by_net[1].shape >> >> >> >> >> >>: torch.Size([212, 2])
                # >> >> >> >> >> > encoder_flows_ori.shape >> >> >> >> >> >>: torch.Size([212, 32])
                # >> >> >> >> >> > encoder_flows_aug.shape >> >> >> >> >> >>: torch.Size([212, 32])
                # >> >> >> >> >> > feature_unsuqeeze.shape >> >> >> >> >> >>: torch.Size([212, 2, 32])


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

                loss_ce = self.entropy_loss(left_classify_result, labels_batch_list_tensor)

                loss = loss_sup
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record the gradient
                grad_list = []
                # for name, parms in net.named_parameters():
                #     # print('-->name:', name, '-->grad_requirs:', parms.requires_grad,' -->grad_value:', parms.grad)
                #
                #     if parms.grad != None:
                #         # print('shape:', name, parms.grad.shape)
                #         grad_list_for_layers = parms.grad.tolist()
                #         grad_list_for_layers.append(name)
                #         # print(name, grad_list_for_layers)
                #         grad_list.append(grad_list_for_layers)
                # save_dir = './gradient_output_with_mixup/'
                # if not os.path.isdir(save_dir):
                #     os.makedirs(save_dir)
                # bk.bk_line_list(grad_list, save_dir + 'epoch_' + str(self.epoch) + '_grad_node_id_' + str(
                #     self.node_id) + '_iter_' + str(iter) + '_batch_' + str(batch_idx) + '.txt')

                batch_loss_sup.append(loss_sup)
                batch_loss_ce.append(loss_ce)
                # print('batch net.cpu().state_dict() : ', self.node_id, net.cpu().state_dict())
                # net.to(self.args.device)
            
            epoch_loss_sup.append(sum(batch_loss_sup) / len(batch_loss_sup))
            epoch_loss_ce.append(sum(batch_loss_ce) / len(batch_loss_ce))
            print(self.node_id, 'iter', iter, 'end')
            # print('epoch net.cpu().state_dict() : ', net.cpu().state_dict())
            # net.to(self.args.device)



        print("--------------------FOR node: %d----------------------" % self.node_id)

        (precision, recall, f1_train, acc) = accuracy(self.args, net, self.local_data_train)
        print("train result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_train, acc))

        (precision, recall, f1_s, acc) = accuracy(self.args, net, self.local_data_valid_s)
        print("s valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_s, acc))

        (precision, recall, f1_t, acc) = accuracy(self.args, net, self.local_data_valid_t)
        print("t valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_t, acc))

        
        return net.cpu().state_dict(), sum(epoch_loss_sup) / len(epoch_loss_sup), sum(epoch_loss_ce) / len(epoch_loss_ce), f1_s, f1_t

