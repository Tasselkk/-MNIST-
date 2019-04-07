import torch as t
from torch import nn
from torch.autograd import Variable

class simplenet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simplenet,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)
    
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

    class activation_net(nn.Module):
        def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
            super(activation_net,self).__init__()
            self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
            self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
            self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))
        def forward(self,x):
            x=self.layer1(x)
            x=self.layer2(x)
            x=self.layer3(x)
            return x
    
    class batch_net(nn.Module):
        def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
            super(batch_net,self).__init__()
            self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
            self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
            self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))

        def forward(self,x):
            x=self.layer1(x)
            x=self.layer2(x)
            x=self.layer3(x)
            return x

        
