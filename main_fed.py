#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch

from models.local_update import LocalUpdate
from models.local_update_ce import LocalUpdate_ce
from models.local_update_muce import LocalUpdate_muce
from models.fed_avg import FedAvg
from models.fed import FedOur
from models.fed_att import FedAtt
from models.fed_wavg import FedWAvg
from models.test import accuracy

from sklearn.preprocessing import StandardScaler
from utils.options import args_parser, print_args
from utils.dataset import dataset
from utils.sampling import sample_iid, sample_all, sample_non_iid
from models import networks_UNSW
from models import networks_CICIDS
from utils.dataset import DatasetSplit
from torch.utils.data import DataLoader, Dataset


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(print_args(args, []))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load dataset and split users
    if args.dataset == 'UNSW':
        root_dir = "./dataset/UNSW-NB15-split2/"
        dataset_train = dataset(root_dir + "train/s_train/")
        dataset_valid_s = dataset(root_dir + "train/s_valid/")
        dataset_valid_t = dataset(root_dir + "train/t/")

        dataset_test_s = dataset(root_dir + "test/s/")
        dataset_test_t = dataset(root_dir + "test/t/")

        if args.iid:
            train_dict_users = sample_iid(dataset_train, args.num_dataset_segmentation)
        else:
            train_dict_users = sample_non_iid(dataset_train, args.num_dataset_segmentation)
        valid_s_dict_users = sample_iid(dataset_valid_s, args.num_dataset_segmentation)
        valid_t_dict_users = sample_iid(dataset_valid_t, args.num_dataset_segmentation)
        # valid_s_dict_users = sample_all(dataset_valid_s, args.num_users)
        # valid_t_dict_users = sample_all(dataset_valid_t, args.num_users)

    elif args.dataset == 'CICIDS':
        root_dir = "./dataset/CICIDS2017-split3/"
        dataset_train = dataset(root_dir + "train/s_train/")
        dataset_valid_s = dataset(root_dir + "train/s_valid/")
        ## 对于CICIDS数据集，无valid_t，为了方便，重复计算。
        dataset_valid_t = dataset(root_dir + "test/t/")

        dataset_test_s = dataset(root_dir + "test/s/")
        dataset_test_t = dataset(root_dir + "test/t/")

        if args.iid:
            train_dict_users = sample_iid(dataset_train, args.num_dataset_segmentation)
        else:
            train_dict_users = sample_non_iid(dataset_train, args.num_dataset_segmentation)
        valid_s_dict_users = sample_iid(dataset_valid_s, args.num_dataset_segmentation)
        valid_t_dict_users = sample_iid(dataset_valid_t, args.num_dataset_segmentation)
        # valid_s_dict_users = sample_all(dataset_valid_s, args.num_users)
        # valid_t_dict_users = sample_all(dataset_valid_t, args.num_users)

    else:
        exit('Error: unrecognized dataset')
    
    scaler = StandardScaler()
    scaler.fit(dataset_train.x)
    dataset_train.x = scaler.transform(dataset_train.x)
    dataset_valid_s.x = scaler.transform(dataset_valid_s.x)
    dataset_valid_t.x = scaler.transform(dataset_valid_t.x)
    dataset_test_s.x = scaler.transform(dataset_test_s.x)
    dataset_test_t.x = scaler.transform(dataset_test_t.x)

    # Gaussian noise


    # build model
    if args.dataset == 'UNSW':
        net = networks_UNSW.Net().to(args.device)
    elif args.dataset == 'CICIDS':
        net = networks_CICIDS.Net().to(args.device)
    else:
        exit('Error: unrecognized dataset')
    
    net.train()

    for epoch in range(args.epochs):
        print("################################EPOCH: %d################################ \n" % epoch)
        loss_supcon_locals = []
        loss_ce_locals = []

        w_locals = []

        f1_v_s_locals = []
        f1_v_t_locals = []

        idx_list = []
        if epoch <= 29:
            for idx in range(args.num_users):
                local = LocalUpdate(args, idx, dataset_train, train_dict_users[idx], dataset_valid_s, valid_s_dict_users[idx], dataset_valid_t, valid_t_dict_users[idx])
                local_weight, local_loss_supcon, local_ce_loss, local_f1_s, local_f1_t = local.train(copy.deepcopy(net).to(args.device))

                is_agg = True
                for k in local_weight.keys():
                    if torch.isnan(local_weight[k]).int().sum() > 0 :
                        is_agg = False
                if is_agg:

                    w_locals.append(local_weight)

                    loss_supcon_locals.append(local_loss_supcon)
                    loss_ce_locals.append(local_ce_loss)

                    f1_v_s_locals.append(local_f1_s)
                    f1_v_t_locals.append(local_f1_t)

                    idx_list.append(idx)
        else:
            for idx in range(args.num_dataset_segmentation):
                local = LocalUpdate(args, idx, dataset_train, train_dict_users[idx], dataset_valid_s,
                                    valid_s_dict_users[idx], dataset_valid_t, valid_t_dict_users[idx])
                local_weight, local_loss_supcon, local_ce_loss, local_f1_s, local_f1_t = local.train(
                    copy.deepcopy(net).to(args.device))

                is_agg = True
                for k in local_weight.keys():
                    if torch.isnan(local_weight[k]).int().sum() > 0:
                        is_agg = False
                if is_agg:
                    w_locals.append(local_weight)

                    loss_supcon_locals.append(local_loss_supcon)
                    loss_ce_locals.append(local_ce_loss)

                    f1_v_s_locals.append(local_f1_s)
                    f1_v_t_locals.append(local_f1_t)

                    idx_list.append(idx)
        # e_w = FedAvg(w_locals_encoder)
        # c_w = FedAvg(w_locals_classifier)
        w = FedOur(w_locals, net.cpu().state_dict(),f1_v_s_locals, args.epsilon, args.ord, dp=args.dp, alpha=args.alpha)
        # w = FedAvg(w_locals)
        # w = FedAtt(w_locals, net.cpu().state_dict(),f1_v_s_locals, args.epsilon, args.ord, dp=args.dp, alpha=args.alpha)
        # w = FedWAvg(args, w_locals, idx_list, train_dict_users)

        net.load_state_dict(w)

        net = net.to(args.device)

        # # begin mongo
        # torch.save({
        #     'epoch': epoch + 1,
        #     'state_dict': net.cpu().state_dict(),
        #     # 'optimizer': optimizer.state_dict(),
        # }, os.path.join(args.model_out_dit, 'checkpoint.pth.tar'))
        # # end mongo

        ##TEST
        print("===========================GLOBAL EPOCH: %d==========================="% epoch)

        print("loss sup for all locals:")
        print(loss_supcon_locals)

        print("loss ce for all locals:")
        print(loss_ce_locals)

        print("loss sup: %.4f, loss ce: %.4f" % (sum(loss_supcon_locals)/len(loss_supcon_locals), sum(loss_ce_locals)/len(loss_ce_locals)))

        (precision, recall, f1_s, acc) = accuracy(args, net, DataLoader(dataset_valid_s, batch_size=args.test_bs, shuffle=True))
        print("s valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_s, acc))

        (precision, recall, f1_t, acc) = accuracy(args, net, DataLoader(dataset_valid_t, batch_size=args.test_bs, shuffle=True))
        print("t valid result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_t, acc))

        (precision, recall, f1_s, acc) = accuracy(args, net, DataLoader(dataset_test_s, batch_size=args.test_bs, shuffle=True))
        print("s test result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_s, acc))

        (precision, recall, f1_t, acc) = accuracy(args, net, DataLoader(dataset_test_t, batch_size=args.test_bs, shuffle=True))
        print("t test result: prec:%.4f, rec:%.4f, f1:%.4f,  acc:%.4f, \n" %(precision, recall, f1_t, acc))

        




