# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.BasicModule import BasicModule
torch.manual_seed(1)

# Knowledge Distillation Loss class. This is used to calculate the loss for knowledge distillation.
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        #kl divergence loss
        self.kl_div = nn.KLDivLoss(reduction = "sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
    
# Self-attention module helps the model to focus on different parts of the input.
class self_attention(nn.Module):
    def __init__(self, in_channles):
        super(self_attention, self).__init__()
        self.in_channels = in_channles

        self.f = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_channles, out_channels=in_channles, kernel_size=1)
        self.softmax_ = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

        #self.init_weight(self.f)
        #self.init_weight(self.g)
        #self.init_weight(self.h)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        #assert channels == self.in_channels
        f = self.f(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x).view(batch_size, -1, height * width)  # B * C//8 * (H * W)

        attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = self.softmax_(attention)

        h = self.h(x).view(batch_size, channels, -1)  # B * C * (H * W)

        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W

        return self.gamma * self_attention_map + x
    

class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('conv1',nn.Conv1d(in_channels = in_channels, out_channels = growth_rate , kernel_size= 7, padding=3))
        #self.add_module('norm1',nn.BatchNorm2d(growth_rate))
        self.add_module('relu1',nn.ReLU(inplace=True))
        
        self.add_module('dropout1',nn.Dropout(p=0.2))
        

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))

class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))



# Encoder class. It processes the input data and encodes it into a feature representation.   
class Encoder(BasicModule):
     def __init__(self,h_features=128,input_features=128):
         super(Encoder,self).__init__()
         
         self.conv1 = nn.Conv1d(in_channels = 21, out_channels = 128, kernel_size= 3, padding=1) #卷积核尺寸，通道数参数可调
        
         self.lstm = nn.LSTM(input_size=128,  
                hidden_size=128, #参数可调
                batch_first= True, bidirectional=False)
         self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
     def forward(self,inputs):
         
         out = self.conv1(inputs)
         
         out = out.transpose(2,1)
         out=self.lstm(out)
         
         out = out[0]
         out = out.contiguous().view(out.size()[0], -1)
         
         #out = self.pool(out)
         return out
     
# Classifier0 class. It takes the encoded features and classifies them into two classes.    
class Classifier0(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier0,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)
    
# Multiple classifier classes with the same structure. They are used for different species.
class Classifier1(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier1,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier2(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier2,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier3(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier3,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier4(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier4,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier5(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier5,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier6(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier6,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier7(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier7,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier8(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier8,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

class Classifier9(BasicModule):
    def __init__(self,input_features=3968):
        super(Classifier9,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_features,64)#参数可调
        self.re = nn.ReLU(inplace=True)
        self.dorp = nn.Dropout(p=0.2)#参数可调
        self.fc2 = nn.Linear(64,2)
        self.re = nn.ReLU(inplace=True)
       

    def forward(self,input):
        #out = self.relu(input)
        out = self.fc1(input)
        out = self.dorp(out)
        out = self.re(out)
        return F.softmax(self.fc2(out),dim=1)

