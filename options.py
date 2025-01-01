#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='IOT', help="name of dataset") #UNSW CICIDS
    parser.add_argument('--mixupalpha', type=float, default=0.5, help='mixup hyper-param')
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--sigma', default=0.75, type=float, help="noise hyperparameter ")

    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_dataset_segmentation', type=int, default=8, help="number of data segmentation: K")
    parser.add_argument('--num_users', type=int, default=7, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments



    parser.add_argument('--epsilon', type=float, default=1.2, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    parser.add_argument('--dp', type=float, default=0.001, help='differential privacy')
    parser.add_argument('--alpha', type=float, default=0.7, help='weight of accuracy and explore')
    parser.add_argument('--beta', type=float, default=0.3, help='weight of sup and ce')
    parser.add_argument('--aug_abnormal', type=int, default=7, help='aug num for anomaly class 9 for UNSW, 7 for CICIDS')

    # other arguments
    #nohup python -u main_fed_dynamic_client.py >results_0311/UNSW_noniid__beta_03_alpha112_3_31.out &
    #nohup python -u main_fed_dynamic_client.py >results_0311/CICIDS_noniid__beta_03_alpha112_3_31.out &

    parser.add_argument('--iid', default=False, action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")

    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    parser.add_argument('--noise_sd', default=0.25, type=float,
                        help="standard deviation of Gaussian noise for data augmentation")
    parser.add_argument('--model_out_dit', default='model_output_dir', type=str,
                        help="model_output_dir checkpoint")
    # certify
    # parser.add_argument('--sigma', default=0.75, type=float,help="noise hyperparameter ")
    parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
    parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
    parser.add_argument("--certify_split", choices=["train", "test"], default="test", help="train or test set")
    parser.add_argument("--N0", type=int, default=1000) #100 100000
    parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
    parser.add_argument("--certify_alpha", type=float, default=0.1, help="failure probability") #0.001
    parser.add_argument("--certify_batch", type=int, default=1000, help="certify_batch size")

    args = parser.parse_args()
    return args


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s
