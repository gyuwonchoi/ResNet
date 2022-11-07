import torch 
import torch.nn as nn
import torch.nn.functional as F

# check paddings , pooling, activiation, batch norm 
class ResNet(nn.Module):
    def __init__(self, layers):
        super(ResNet, self).__init__() 
        self.layers = layers - 2
        self.blocks = int(self.layers / 3)
        self.n = int(self.blocks / 2)
            
        # === optimize here === #
        # input channel, output channel, kernel, stride, padding 
        self.conv1   = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(16, 16, 3, 1, 1)
        # self.conv2_2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(16, 32, 3, 2, 1)                   # subsampling
        
        self.conv3_1 = nn.Conv2d(32, 32, 3, 1, 1)
        # self.conv3_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(32, 64, 3, 2, 1)                   # subsampling
        
        self.conv4_1 = nn.Conv2d(64, 64, 3, 1, 1)
        
        self.relu = nn.ReLU()
        
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.short2 = nn.Conv2d(16, 32, 1, 2)
        self.short3 = nn.Conv2d(32, 64, 1, 2)
        
        self.maxpool=nn.MaxPool2d(3, 2) 
        self.avgpool= nn.AvgPool2d(8, 1)
        self.fc = nn.Linear(64, 10)            # CIFAR-10
 

    def forward(self, x): # miss the x
        
        # conv1 
        x= self.conv1(x)
        x= self.bn2(x)                          # batch norm before activation 
        
        # conv2~4 block
        for block in range(3):     
            # optimize here 
            if block == 0:                      # block 0
                f = self.conv2_1
                f2 = self.conv2_2
                bn = self.bn2                   # 16
                bn2 = self.bn3                  # 32
                short = self.short2
           
            elif(block == 1):                   # block 1
                f = self.conv3_1
                f2 = self.conv3_2
                bn = self.bn3                   # 32
                bn2 = self.bn4                  # 64
                short = self.short3
            
            else:                               # block 2
                f = self.conv4_1
                bn = self.bn4
             
            for i in range(self.n):
                shortcut = x
                if(i != (self.n -1)):
                    for j in range(2):
                        x = f(x)
                        
                        if j==0:
                            x = self.relu(bn(x))
                        else:  
                            x = bn(x)
                    x = shortcut + x
                    x = self.relu(bn(x))
               
                else:
                    shortcut = x
                    
                    x = f(x)
                    x = self.relu(bn(x))     
                    
                    if(block!=2): 
                        shortcut = short(shortcut)
                        
                        x = bn2(f2(x))               # subsampling 
                        # print(x.shape)
                        
                        x = self.relu(shortcut + x)
                        # print((shortcut + x).shape)
        
        # conv5 block 
        x = self.avgpool(x)

        # fc layer 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        
        return x
