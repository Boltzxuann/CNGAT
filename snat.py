import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, SNATConv
import torch.nn.functional as F
from linear_layer import Linear

class SNAT(nn.Module):
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
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(SNAT, self).__init__()
        self.g = g
        self.heads=heads
        self.num_hidden=num_hidden
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.snat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.snat_layers.append(SNATConv( True,                      
            in_dim, num_hidden, heads[0], num_hiddenT[0],
            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False, activation=self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.snat_layers.append(SNATConv(False,
                        num_hidden * heads[l-1], num_hidden, heads[l], num_hiddenT[l],
                        feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False, activation=self.activation ))
        # output projection
  
        self.snat_layers.append(SNATConv(False,
                         num_hidden * heads[-2], out_dim, heads[-1], num_hiddenT[-1],
                         feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False, activation=self.activation)) 
       
       #output layer
        self.output = Linear(in_features=out_dim, out_features=1)
        #self.output_rear = Linear(in_features=out_dim, out_features=1)
       
    def forward(self, inputs):
        h = inputs
        h, index1, index0 = self.snat_layers[0](self.g,h,index_1=[],index_0=[])

        h=torch.flatten(h,start_dim=1,end_dim=2)
        for l in range(self.num_layers-1):
            h,index1, index0 = self.snat_layers[l+1](self.g, h, index_1=index1, index_0=index0)  
            h=torch.flatten(h,start_dim=1,end_dim=2)
        # output projection
        features, index1, index0 = self.snat_layers[-1](self.g, h, index_1=index1, index_0=index0)
        features=features.mean(1)
        out = self.output(features)
        #out = torch.zeros(index0.size(0),1)
        #out = index0[:,0].view(-1,1).float()
        #index00 = index0[:,0].view(-1,1)
        #index11 = index1[:,0].view(-1,1)
        #out[index0[:,0]] = self.output( features[ index00.expand( index0[:,0].size(0), self.out_dim ) ].view(-1,self.out_dim ) )
        #out[index1[:,0]] = self.output_rear(features[index11.expand(index1[:,0].size(0), self.out_dim) ].view(-1,self.out_dim ))
        out = F.relu(out)  #最后一层用值域在0到+无穷的激活函数
        
        return out