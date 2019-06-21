import torch
import torch.nn as nn
import torch.optim as optim


class BLLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu    = nn.LeakyReLU()

        self.ln1 = nn.Linear(600, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.ln2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.ln3 = nn.Linear(256,256)
        self.bn3 = nn.BatchNorm1d(256)
        self.ln4 = nn.Linear(256,128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(128,64)

        self.bn5 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64,32)
        
        self.bn6 = nn.BatchNorm1d(32)

        self.fc3 = nn.Linear(32,1)
        
        self.sigmoid = nn.Sigmoid()
        
  
    def forward(self, source_vector, target_vector):
        
        x =  torch.cat((source_vector, target_vector), dim=1)

        x = self.bn1(self.relu(self.ln1(x)))
        x = self.bn2(self.relu(self.ln2(x)))
        x = self.bn3(self.relu(self.ln3(x)))
        x = self.bn4(self.relu(self.ln4(x)))

        x =  self.bn5( self.relu(self.fc1(x) ))
        x =  self.bn6( self.relu( self.fc2(x) ))
        x =  self.sigmoid( self.fc3(x) )
        
        return x
        
