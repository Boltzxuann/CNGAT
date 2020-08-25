"""Torch modules for category node attention networks(CNAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from .... import function as fn
from ..softmax import edge_softmax
from ..utils import Identity
from ....utils import expand_as_pair

# pylint: enable=W0235
class SNATConv(nn.Module):

    def __init__(self,
                 first_layer,
                 
                 in_feats,
                 out_feats,
                 num_heads,
                 num_hiddenT, ##MLP的隐藏层大小
                 threhold=0,
                 index_rear=[],
                 index_other=[],
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(SNATConv, self).__init__()
        self.first_layer = first_layer
        self.threhold = threhold
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._num_hiddenT = num_hiddenT
        if isinstance(in_feats, tuple):
            self.h1_src = nn.Linear(self._in_src_feats, num_hiddenT, bias=False) ##第一层
            self.h1_src_rear = nn.Linear(self._in_src_feats, num_hiddenT, bias=False) ##第一层 (显著点)
            #hl_src = th.sigmoid(h1_src)
            self.fc_src = nn.Linear(
                num_hiddenT, out_feats * num_heads, bias=False)  ##第二层
            self.fc_src_rear = nn.Linear(
                num_hiddenT, out_feats * num_heads, bias=False)  ##第二层            
            self.h1_dst = nn.Linear(self._in_dst_feats, num_hiddenT, bias=False) ##第一层
            self.h1_dst_rear = nn.Linear(self._in_dst_feats, num_hiddenT, bias=False) ##第一层
            self.fc_dst = nn.Linear(
                num_hiddenT, out_feats * num_heads, bias=False)
            self.fc_dst_rear = nn.Linear(
                num_hiddenT, out_feats * num_heads, bias=False)
        else:
            self.h1 = nn.Linear(self._in_src_feats, num_hiddenT, bias=False) ##第一层
            self.h1_rear = nn.Linear(self._in_src_feats, num_hiddenT, bias=False) ##第一层
            self.fc_rear = nn.Linear(num_hiddenT, out_feats * num_heads, bias=False)
            self.fc = nn.Linear(num_hiddenT, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'h1'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_rear.weight, gain=gain)
            nn.init.xavier_normal_(self.h1.weight, gain=gain)
            nn.init.xavier_normal_(self.h1_rear.weight, gain=gain)
        else: # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.h1_src.weight, gain=gain)
            nn.init.xavier_normal_(self.h1_src_rear.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_src_rear.weight, gain=gain)
            nn.init.xavier_normal_(self.h1_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.h1_dst_rear, gain=gain)
            nn.init.xavier_normal_(self.fc_dst_rear, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, index_1, index_0):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        if self.first_layer:
            if isinstance(feat, tuple):
                index_rear_src = feat[0] > self.threhold  ##找到大于阈值的节点索引
                index_other_src = feat[0] <=self.threhold 
                index_rear_dst = feat[1] > self.threhold  ##找到大于阈值的节点索引
                index_other_dst = feat[1] <=self.threhold  
            else:
                index_rear = feat > self.threhold  ##找到大于阈值的节点索引
                index_other = feat<=self.threhold
            #index_0 = index_other
            #index_1 = index_rear
        else:
            index_00 = index_0[:,0].view(-1,1)
            index_11 = index_1[:,0].view(-1,1)
            if isinstance(feat, tuple):
                
                index_rear_src = index_11.expand(index_11.size(0),self._in_src_feats)  ##索引要和特征向量维度匹配
                index_other_src = index_00.expand(index_00.size(0),self._in_src_feats)
                index_rear_dst = index_11.expand(index_11.size(0),self._in_dst_feats)  ##索引要和特征向量维度匹配
                index_other_dst = index_00.expand(index_00.size(0),self._in_dst_feats)      
            else:
                index_rear = index_11.expand(index_11.size(0),self._in_src_feats)
                index_other = index_00.expand(index_00.size(0),self._in_src_feats)
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0]) ##该节点作为源节点的表征
            h_dst = self.feat_drop(feat[1]) ##该节点作为目的节点的表征

            h1_s = self.h1_src(h_src[index_other_src].view(-1,self._in_src_feats))
            h1_s = self.activation(h1_s)
            #h1_s = th.tanh(h1_s)
            
            h1_d = self.h1_dst(h_dst[index_other_dst].view(-1,self._in_src_feats))
            #h1_d = F.relu(h1_d)
            h1_d = self.activation(h1_d)
            feat_src = self.fc_src(h1_s).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h1_d).view(-1, self._num_heads, self._out_feats)
            feat_src = self.activation(feat_src)
            feat_dst = self.activation(feat_dst)
            
            
            h1_s_rear = self.h1_src_rear(h_src[index_rear_src].view(-1,self._in_src_feats))
            h1_s_rear = self.activation(h1_s_rear)
            h1_d_rear = self.h1_dst(h_dst[index_rear_dst].view(-1,self._in_src_feats))
            h1_d_rear = self.activation(h1_d_rear)
            feat_src_rear = self.fc_src(h1_s_rear)
            feat_dst_rear = self.fc_dst(h1_d_rear)
            feat_src_rear = self.activation(feat_src_rear).view(-1, self._num_heads, self._out_feats)
            feat_dst_rear = self.activation(feat_dst_rear).view(-1, self._num_heads, self._out_feats)

        else:
            h_src = h_dst = self.feat_drop(feat)
            
            
            if self.first_layer:
                H1 = self.h1( h_src[index_other].view(-1,feat.shape[1]) )
            else:

                hs_0=h_src[index_other]###???
                hs_0=hs_0.view(-1,self._in_src_feats)
                H1 = self.h1(hs_0)
            H1 = self.activation(H1)
            feat_src = feat_dst = self.fc(H1)##6.18
            
            feat_src = feat_dst = self.activation(feat_src).view(
                -1, self._num_heads, self._out_feats)
            
            
            if self.first_layer:
                H1_rear = self.h1(h_src[index_rear].view(-1,feat.shape[1]))
            else:
                H1_rear = self.h1_rear(h_src[index_rear].view(-1,self._in_src_feats))##？？？

            H1_rear = self.activation(H1_rear)
            feat_src_rear = feat_dst_rear = self.fc_rear(H1_rear)
            feat_src_rear = feat_dst_rear = self.activation(feat_src_rear).view(
                -1, self._num_heads, self._out_feats)           
            #feat_src = feat_dst = th.sigmoid(feat_src)

        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        el_rear = (feat_src_rear * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er_rear = (feat_dst_rear * self.attn_r).sum(dim=-1).unsqueeze(-1)        
        #graph.nodes[index_other.flatten(0)].srcdata.update({'ft': feat_src, 'el': el})
        if self.first_layer:
            #id_other = np.where(index_other.flatten(0).cpu().numpy() == True)
            id_other = th.where(index_other.flatten(0) == True)
            id_other = id_other[0].flatten()         
        else:
            #id_other = np.where(index_other[:,0].flatten(0).cpu().numpy() == True)
            id_other = th.where(index_other[:,0].flatten(0))
            id_other = id_other[0].flatten()
        graph.nodes[id_other].data['ft']=feat_src
        graph.nodes[id_other].data['el']=el
        #graph.nodes[index_other.flatten(0)].dstdata.update({'er': er})
        graph.nodes[id_other].data['er']=er
        #graph.nodes[index_rear.flatten(0)].srcdata.update({'ft': feat_src_rear, 'el': el_rear})
        if self.first_layer:
            #id_rear = np.where(index_rear.flatten(0).cpu().numpy() == True)
            id_rear = th.where(index_rear.flatten(0) == True)
            id_rear = id_rear[0].flatten()   
        else:
            #id_rear = np.where(index_rear[:,0].flatten(0).cpu().numpy() == True)
            id_rear = th.where(index_rear[:,0].flatten(0) == True)
            id_rear = id_rear[0].flatten() 
        graph.nodes[id_rear].data['ft']=feat_src_rear
        graph.nodes[id_rear].data['el']=el_rear
        #graph.nodes[index_rear.flatten(0)].dstdata.update({'er': er_rear})
        graph.nodes[id_rear].data['er']=er_rear
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))  ###边的注意力分数
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst, index_rear, index_other 