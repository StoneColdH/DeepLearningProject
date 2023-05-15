import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
"""
	TGTN-GNN Layers

"""

class PosEncoding(nn.Module):
    
    def __init__(self, dim, device, base=10000, bias=0):
        
        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension 
		:param device: where to train model
		:param base: the encoding base
		:param bias: the encoding bias
        """
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)

class TransEmbedding(nn.Module):
    
    def __init__(self, df=None, device='cpu', dropout=0.2, in_feats=82,cat_features=None):
        """
        Initialize the attribute embedding and feature learning compoent
        
        :param df: the feature
		:param device: where to train model
		:param dropout: the dropout rate
		:param in_feat: the shape of input feature in dimension 1
		:param cat_feature: category features
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats, device=device, base=100)
        #time_emb = time_pe(torch.sin(torch.tensor(df['time_span'].values)/86400*torch.pi))
        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique())+1, in_feats).to(device) for col in cat_features if col not in {"Labels", "Time"}})
        self.label_table = nn.Embedding(3, in_feats, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features=cat_features
        self.forward_mlp = nn.ModuleList([nn.Linear(in_feats, in_feats) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)
    
    def forward_emb(self, df):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        #print(self.emb_dict)
        #print(df['trans_md'])
        support = {col: self.emb_dict[col](df[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        #self.time_emb = self.time_pe(torch.sin(torch.tensor(df['time_span'])/86400*torch.pi))
        #support['time_span'] = self.time_emb
        #support['labels'] = self.label_table(df['labels'])
        return support
    
    def forward(self, df):
        support = self.forward_emb(df)
        output = 0
        for i, k in enumerate(support.keys()):
            #if k =='time_span':
            #    print(df[k].shape)
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            output = output + support[k]
        return output


class TransformerConv(nn.Module):
   
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 #feat_drop=0.6,
                 #attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
    
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
	    :param in_feat: the shape of input feature
	    :param out_feats: the shape of output feature
	    :param num_heads: the number of multi-head attention 
	    :param bias: whether to use bias
	    :param allow_zero_in_degree: whether to allow zero in degree
	    :param skip_feat: whether to skip some feature 
	    :param gated: whether to use gate
	    :param layer_norm: whether to use layer regularization
	    :param activation: the type of activation function   
        """

        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_key = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_value = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)

        #self.feat_dropout = nn.Dropout(p=feat_drop)
        #self.attn_dropout = nn.Dropout(p=attn_drop)
        if skip_feat:
            self.skip_feat = nn.Linear(self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(3*self._out_feats*self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
	    :param feat: input feat
	    :param get_attention: whether to get attention
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # Step 0. q, k, v
        q_src = self.lin_query(h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(h_src).view(-1, self._num_heads, self._out_feats)
        # Assign features to nodes
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'] / self._out_feats**0.5)

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'), fn.sum('attn', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u'].reshape(-1, self._out_feats*self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst
        


