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

        self.conv1  = nn.Conv2d(3, 16, 3, 1, 1)
        self.relu   = nn.ReLU()
        self.bn2    = nn.BatchNorm2d(16)
        self.avgpool= nn.AvgPool2d(8, 1)
        self.fc = nn.Linear(64, 10)            
        
        # ========================================================= #
        self.conv_ = [nn.Conv2d(16, 16, 3, 1, 1),
                     nn.Conv2d(32, 32, 3, 1, 1),
                     nn.Conv2d(64, 64, 3, 1, 1)]

        self.bn_ = [nn.BatchNorm2d(16),
                   nn.BatchNorm2d(32),
                   nn.BatchNorm2d(64)]
        
        self.sub_sample_ = [
                           nn.Conv2d(16, 16, 3, 1, 1),
                           nn.Conv2d(16, 32, 3, 2, 1),
                           nn.Conv2d(32, 64, 3, 2, 1)]
        
        self.short_ = [nn.Conv2d(16, 16, 1, 1),
                      nn.Conv2d(16, 32, 1, 2),
                      nn.Conv2d(32, 64, 1, 2)
                     ]
                
        self.conv =  nn.ModuleList(self.conv_)
        self.bn =  nn.ModuleList(self.bn_)
        self.sub_sample=  nn.ModuleList(self.sub_sample_)
        self.short =  nn.ModuleList(self.short_)

    def forward(self, x): # miss the x
        
        # conv1 
        x= self.conv1(x)
        x= self.bn2(x)                          
        
        # conv2~4 block
        for block in range(3):     

            # layers 
            for i in range(self.n):             # reiteration of pair 
                shortcut = x
            
                if(i == 0):                             # 1st pair 
                    x = self.sub_sample[block](x)       # 1st layer of pair 
                    x = self.relu(self.bn[block](x))
                    
                    x = self.conv[block](x)             # 2nd layer of pair
                    x = self.bn[block](x)
                    
                    shortcut = self.short[block](shortcut)
                    x = self.relu(self.bn[block](shortcut + x))
                
                else:
                    x = self.conv[block](x)            # 1st layer of pair 
                    x = self.relu(self.bn[block](x))
                    
                    x = self.conv[block](x)            # 2nd layer of pair
                    x = self.bn[block](x)
                    
                    x = self.relu(self.bn[block](shortcut + x))
                     
        # conv5 block 
        x = self.avgpool(x)

        # fc layer 
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        
        return x
