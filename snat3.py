import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, SNATConv
import torch.nn.functional as F
from linear_layer import Linear

class SNAT3(nn.Module):
    def __init__(self,
                 g,
                 threhold,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,
                 heads,
                 num_hiddenT, ##MLP的隐藏层大小
                 activation,
                 activationP,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(SNAT3, self).__init__()
        self.g = g
        self.heads=heads
        self.num_hidden=num_hidden
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.snat_layers = nn.ModuleList()
        self.activation = activation
        self.activationP = activationP
        # input projection (no residual)
        self.embedding = Linear(in_features= in_dim, out_features = num_hidden, bias=False)
        
        self.snat_layers.append(SNATConv( True,                      
            num_hidden, num_hidden, heads[0], num_hiddenT[0],
            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False, activation=self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.snat_layers.append(SNATConv(False,
                        num_hidden * heads[l-1], num_hidden, heads[l], num_hiddenT[l],
                        feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False, activation=self.activation ))
        # output projection
  
        self.snat_layers.append(SNATConv(False,
                         num_hidden * heads[-2], num_hidden, heads[-1], num_hiddenT[-1],
                         feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False, activation=self.activation)) 
       
       #output layer
        self.output0 = Linear( in_features= (self.num_layers+1)*num_hidden, out_features= out_dim)
        self.output1 = Linear( in_features= out_dim, out_features= out_dim)
        self.output2 = Linear( in_features= out_dim, out_features= 1)
        #self.output_rear = Linear(in_features=out_dim, out_features=1)
       
    def forward(self, inputs):
        h = inputs
        h = self.embedding(h)
        h = torch.tanh(h)
        h, index1, index0 = self.snat_layers[0](self.g,h,index_1=[],index_0=[])

        h=torch.flatten(h,start_dim=1,end_dim=2)
        H=h
        for l in range(self.num_layers-1):
            h,index1, index0 = self.snat_layers[l+1](self.g, h, index_1=index1, index_0=index0)  
            h=torch.flatten(h,start_dim=1,end_dim=2)
            H=torch.cat((H,h),1)
        # output projection
        features, index1, index0 = self.snat_layers[-1](self.g, h, index_1=index1, index_0=index0)
        features=features.mean(1)
        features = torch.cat((H,features),1)
        out = self.output0(features) 
        if self.activationP is not None:
            out = self.activationP(out)  
        out = self.output1(out)  
        if self.activationP is not None:
            out = self.activationP(out)
        out = self.output2(out)  
        out = F.relu(out)#最后一层用值域在0到+无穷的激活函数
        return out