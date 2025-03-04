import torch
import torch.nn as nn
import torch.nn.functional as F

# different weight init function but I have not used them as the model was performing good
def xavier_init(param):
    nn.init.xavier_normal_(param,gain=1.0)


def zero_init(param):
    nn.init.zeros_(param)

def kaiming_normal(param):
    nn.init.kaiming_normal(param)

# simple architecture 
class SimpleInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Each branch gets 1/4 of output channels
        branch_channels = out_channels // 4
        
        # Simple inception branches
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Conve Layers 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.inception = SimpleInception(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        #FC layers 
          # self.fc1 = nn.Linear(64, 64)
          # self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64,32)
        self.dropout2 = nn.Dropout(0.02)

        self.fc_out = nn.Linear(32,10)       


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.inception(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # x = F.relu(self.fc1(x))
        # x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)            

        return self.fc_out(x)
      



#Deep architecture 
class DeeperInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Each branch gets 1/4 of output channels
        branch_channels = out_channels // 4
        
        # 2 conv per branch
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.Conv2d(branch_channels, branch_channels, 1)
        )
        
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1)
        )
        
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 5, padding=2),
            nn.Conv2d(branch_channels, branch_channels, 5, padding=2)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.Conv2d(branch_channels, branch_channels, 1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch_1x1(x),
            self.branch_3x3(x),
            self.branch_5x5(x),
            self.branch_pool(x)
        ], dim=1)


class DEEPMNISTNet(nn.Module):
      def __init__(self):
        super().__init__()
        
        #Conve Layers 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.inception = DeeperInception(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        #FC layers 
        self.fc1 = nn.Linear(64, 64)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64,32)
        self.dropout2 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(32,10)         


      def forward(self, x):
          x = F.relu(self.conv1(x))
          x = F.relu(self.inception(x))
          x = self.pool(x)
          x = x.view(x.size(0), -1)

          x = F.relu(self.fc1(x))
          x = self.dropout1(x)

          x = F.relu(self.fc2(x))
          x = self.dropout2(x)
             

          return self.fc_out(x)