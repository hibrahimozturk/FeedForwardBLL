import torch
import torch.nn as nn
import torch.optim as optim


class BLLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu    = nn.ReLU(inplace=True)

        self.linear_1_1 = nn.Linear(300, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.linear_1_2 = nn.Linear(256,128)
        
        self.linear_2_1 = nn.Linear(300,256)
        self.bn2_1 = nn.BatchNorm1d(256)
        self.linear_2_2 = nn.Linear(256,128)
        
        self.bn2 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256,256)

        self.bn3 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256,32)
        self.fc3 = nn.Linear(32,1)
        
        self.sigmoid = nn.Sigmoid()
        
  
    def forward(self, source_vector, target_vector):
        
        
        branch_1 = self.bn1_1(self.relu(self.linear_1_1(source_vector)))
        branch_1 = self.relu( self.linear_1_2(branch_1))
        
        branch_2 = self.bn2_1(self.relu(self.linear_2_1(target_vector)))
        branch_2 = self.relu(self.linear_2_2(branch_2))     
        
        x =  self.bn2(torch.cat((branch_1, branch_2), dim=1))
        
        x =  self.bn3(self.relu(self.fc1(x) ))
        x =  self.relu( self.fc2(x) )
        x =  self.sigmoid( self.fc3(x) )
        
        return x
        
