# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

def accuracy(args, net, loader):
    net.eval()
    first_data = True
    num_correct = 0
    
    for idx, (data, target) in enumerate(loader):

        data = data.float().to(args.device)
        target = target.long().to(args.device)

        out = net(data)
        prediction = torch.max(F.softmax(out[1].cpu()),1)[1]
        # print('test prediction : ', type(prediction), prediction.shape)
        p = prediction.data.squeeze()
        # print('test prediction.data.squeeze() : ', p, type(p), p.shape)

        if first_data:
            y_all = torch.squeeze(target)
            p_all = p
            first_data = False
        else:
            y_all = torch.hstack((y_all, torch.squeeze(target)))
            p_all = torch.hstack((p_all, p))
            
        num_correct += torch.eq(p.cpu(), torch.squeeze(target).cpu()).sum().float().item()
        
    y_all = y_all.cpu()
    p_all = p_all.cpu()

    precision = precision_score(y_all, p_all)
    recall = recall_score(y_all, p_all)
    f1 = f1_score(y_all, p_all)
    acc = num_correct / len(loader.dataset)

    print("real anomaly num: ", torch.sum(y_all, dim=0))
    print("real normal num: ", y_all.size()[0] - torch.sum(y_all, dim=0))

    net.train()
    return (precision, recall, f1, acc)

