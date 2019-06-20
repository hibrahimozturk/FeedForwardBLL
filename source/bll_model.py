import torch
import torch.nn as nn
import torch.optim as optim


class BLLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu    = nn.ReLU(inplace=True)

        self.linear_1_1 = nn.Linear(300, 256)
        self.linear_1_2 = nn.Linear(256,128)
        
        self.linear_2_1 = nn.Linear(300,256)
        self.linear_2_2 = nn.Linear(256,128)
        
        
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,1)
        
        self.sigmoid = nn.Sigmoid()
        
  
    def forward(self, source_vector, target_vector):
        
        
        branch_1 = self.relu( nn.functional.dropout(self.linear_1_1(source_vector), p=0.5 ) )
        branch_1 = self.relu( nn.functional.dropout( self.linear_1_2(branch_1), p=0.5 ) )
        
        branch_2 = self.relu(nn.functional.dropout(self.linear_2_1(target_vector), p=0.5))
        branch_2 = self.relu(nn.functional.dropout(self.linear_2_2(branch_2), p=0.5))     
        
        x = torch.cat((branch_1, branch_2), dim=1)
        
        x =  self.relu( nn.functional.dropout(self.fc1(x), p=0.5 ) )
        x =  self.relu( nn.functional.dropout(self.fc2(x), p=0.5 ) )
        x =  self.sigmoid( self.fc3(x) )
        
        return x
        
