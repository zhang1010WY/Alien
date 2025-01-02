from torch import nn
import torch.nn.functional as F
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(77, 128)
        self.b1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.b2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.b3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
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