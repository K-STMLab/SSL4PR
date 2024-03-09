import torch
from torch import nn 
import torch.nn.functional as F
from torch import nn 
import time

class CNN1D(nn.Module):
    def __init__(self,
            num_classes = 1,
            input_channels = 1
        ):
        super(CNN1D,self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # first block of convolutions
        self.conv1 = nn.Conv1d(
            in_channels = self.input_channels,
            out_channels = 16,
            kernel_size = 64
        )
        self.maxpool1 = nn.MaxPool1d(2, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        # second block of convolutions
        self.conv2 = nn.Conv1d(
            in_channels = 16, 
            out_channels = 32, 
            kernel_size = 32
        )
        self.maxpool2 = nn.MaxPool1d(2, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        # third block of convolutions
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 64, 
            kernel_size = 16
        )
        self.maxpool3 = nn.MaxPool1d(2, stride=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # global average pooling and flattening
        self.glob_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # classifier
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(p=0.25)
        
        self.init_weights()

    def init_weights(self):
        # initialize weights of classifier
        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
        
    def forward(self, input_data):
        # first block of convolutions
        x = F.relu(
            self.conv1(input_data)
        )
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.bn1(x)
        
        # second block of convolutions
        x = F.relu(
            self.conv2(x)
        )
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.bn2(x)
        
        # third block of convolutions
        x = F.relu(
            self.conv3(x)
        )
        x = self.maxpool3(x)
        x = self.dropout(x)
        x = self.bn3(x)
        # global average pooling and flattening
        x = self.glob_avg_pool(x)
        x = self.flatten(x)
        
        # classifier
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        predictions = F.sigmoid(x) 
        return predictions
    
    
if __name__ == '__main__':
    cnn = CNN1D()
    in_tensor = torch.randn((4, 1, 16000))
    output = cnn(in_tensor)
    print(output.shape)