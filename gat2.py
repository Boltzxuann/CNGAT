import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn.functional as F
from linear_layer import Linear

class GAT2(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 out_dim,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT2, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, True,self.activation)) #最后一层用值域在0到+无穷的激活函数
        
        
        self.output0 = Linear( in_features= (self.num_layers+1)*num_hidden, out_features= out_dim)
        self.output1 = Linear( in_features= out_dim, out_features= 1)
        

    def forward(self, inputs):
        h = inputs
        h = self.gat_layers[0](self.g,h)
        h=torch.flatten(h,start_dim=1,end_dim=2)
        H=h
        for l in range(self.num_layers-1):
            h= self.gat_layers[l+1](self.g, h)  
            h=torch.flatten(h,start_dim=1,end_dim=2)
            H=torch.cat((H,h),1)
        # output projection
        features = self.gat_layers[-1](self.g, h)
        features=features.mean(1)
        features = torch.cat((H,features),1)
        out = self.output0(features)    
        out = F.relu(out)  #最后一层用值域在0到+无穷的激活函数
        out = self.output1(out)  
        out = F.relu(out)
        return out
