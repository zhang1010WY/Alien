from utils.dataset import dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.preprocessing import StandardScaler
from models.supconloss import SupConLoss

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(43, 64)
        self.b1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.b2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.b3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 32)
        self.b4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 12)
        self.b5 = nn.BatchNorm1d(12)
        self.fc6 = nn.Linear(12, 2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        # x = self.b1(x)
        x = F.elu(self.fc2(x))
        # x = self.b2(x)
        x = F.elu(self.fc3(x))
        # x = self.b3(x)
        x2 = F.elu(self.fc4(x))
        # x = self.b4(x)
        x2 = F.elu(self.fc5(x2))

        x2 = F.elu(self.fc6(x2))
        # return self.softmax(x)
        return x, x2
    

root_dir = "/home/gpu89/liang/DG-FL/dataset/UNSW-NB15-split2/"
dataset_train = dataset(root_dir + "train/s_train/")
dataset_valid_s = dataset(root_dir + "train/t/")
# dataset_valid_t = dataset(root_dir + "train/t/")
# dataset_test_s = dataset(root_dir + "test/s/")
# dataset_test_t = dataset(root_dir + "test/t/")
scaler = StandardScaler()
scaler.fit(dataset_train.x)
dataset_train.x = scaler.transform(dataset_train.x)
dataset_valid_s.x = scaler.transform(dataset_valid_s.x)

device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
net = Net()
net = net.to(device)
# optimizer = torch.optim.SGD(net.parameters(),lr=0.01, momentum = 0.5)
optimizer =  torch.optim.Adam(net.parameters(), lr=0.001,  weight_decay=5e-4)
loss_func = torch.nn.CrossEntropyLoss()
criterion = SupConLoss()

for it in range(100):
    loss_batches = []
    for batch_idx, (flows, labels) in enumerate(DataLoader(dataset_train, batch_size=64, shuffle=True)):
        flows, labels = flows.float().to(device), labels.long().to(device)
        net.train()

        idx_list = np.arange(flows.size()[0])

        for idx in range(flows.size()[0]):
            [s_idx] = np.random.choice(idx_list, 1, replace=False)

            lam = np.random.beta(0.2, 0.2)
            while lam < 1e-1:
                lam = np.random.beta(0.2, 0.2)
                    
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
                
        flows_ori = flows_ori.float().to(device)
        flows_aug = flows_aug.float().to(device)
        labels_aug = labels_aug.to(device)
        labels_aug = labels_aug.squeeze()

        # idx_list = []
        # for idx in range(flows.size()[0]):
        #     if labels[idx] == 1:
        #         idx_list.append(idx)
        
        # idx_list = np.array(idx_list)

        # for idx in range(flows.size()[0]):
        #     if labels[idx] == 0:
        #         flow_tem = flows[idx]
        #         label_tem = labels[idx]
        #         if idx == 0:
        #             flows_aug = flow_tem
        #             labels_aug = label_tem
        #         else:
        #             flows_aug = torch.vstack((flows_aug, flow_tem))
        #             labels_aug = torch.vstack((labels_aug, label_tem))
        #     else:
        #         for aug in range(10):
        #             [s_idx] = np.random.choice(idx_list, 1, replace=False)
        #             lam = np.random.beta(1, 1)
        #             while lam < 0.5:
        #                 lam = np.random.beta(10, 10)
        #             flow_tem = lam * flows[idx] + (1-lam) * flows[s_idx]
        #             label_tem = labels[idx]
        #             if idx == 0:
        #                 flows_aug = flow_tem
        #                 labels_aug = label_tem
        #             else:
        #                 flows_aug = torch.vstack((flows_aug, flow_tem))
        #                 labels_aug = torch.vstack((labels_aug, label_tem))

                            
        # flows_aug = flows_aug.float().to(device)
        # labels_aug = labels_aug.long().to(device)

        encoder_flows_ori = net(flows_ori)
        encoder_flows_aug = net(flows_aug)
        feature_unsuqeeze = torch.cat([encoder_flows_ori[0].unsqueeze(1), encoder_flows_aug[0].unsqueeze(1)], dim=1)
        loss_sup = criterion(feature_unsuqeeze, labels_aug)

        encoder_flows = net(flows)
        loss_ce = loss_func(encoder_flows[1], labels)


        
        loss = loss_sup + loss_ce
        loss_batches.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("iter: %d, loss: %.4f" %(it, sum(loss_batches)/len(loss_batches)))

    if it % 1 == 0:
        first_data = True
        for batch_idx, (flows, labels) in enumerate(DataLoader(dataset_train, batch_size=256, shuffle=True)):
            flows, labels = flows.float().to(device), labels.long().to(device)
            net.eval()
            out = net(flows)
            # print("train out")
            # print(out[:20])
            prediction = torch.max(F.softmax(out[1].cpu()),1)[1]
            pred_y = prediction.data.squeeze()

            if first_data:
                y_all = labels
                p_all = pred_y
                first_data = False
            else:
                y_all = torch.hstack((y_all, labels))
                p_all = torch.hstack((p_all, pred_y))
    

        y_all = y_all.cpu()
        p_all = p_all.cpu()

        precision = precision_score(y_all, p_all)
        recall = recall_score(y_all, p_all)
        f1 = f1_score(y_all, p_all)

        print("###########train, prec:%.4f, rec:%.4f, f1:%.4f \n" %(precision, recall, f1))
    
    if it % 1 == 0:
        first_data = True
        for batch_idx, (flows, labels) in enumerate(DataLoader(dataset_valid_s, batch_size=256, shuffle=True)):
            flows, labels = flows.float().to(device), labels.long().to(device)
            out = net(flows)
            # print("valid out")
            # print(out[:20])
            prediction = torch.max(F.softmax(out[1].cpu()),1)[1]
            pred_y = prediction.data.squeeze()

            if first_data:
                y_all = labels
                p_all = pred_y
                first_data = False
            else:
                y_all = torch.hstack((y_all, labels))
                p_all = torch.hstack((p_all, pred_y))
    

        y_all = y_all.cpu()
        p_all = p_all.cpu()

        precision = precision_score(y_all, p_all)
        recall = recall_score(y_all, p_all)
        f1 = f1_score(y_all, p_all)

        print("###########valid, prec:%.4f, rec:%.4f, f1:%.4f \n" %(precision, recall, f1))

    





