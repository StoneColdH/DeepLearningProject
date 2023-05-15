
import torch.nn as nn
from layers import *
"""
	TGTN-GNN Layers

"""


class GraphAttnModel(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 device='cpu'):
        
        """
        Initialize the GTAN-GNN model
        :param in_feats: the shape of input feature
		:param hidden_dim: model hidden layer dimension
		:param n_layers: the number of GTAN layers
		:param n_classes: the number of classification
		:param heads: the number of multi-head attention 
		:param activation: the type of activation function
		:param skip_feat: whether to skip some feature
		:param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
		:param post_proc: whether to use post processing
		:param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
		:param ref_df: whether to refer other node features
		:param cat_features: category features
        :param device: where to train model
        """

        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.activation = activation
        #self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        # self.pn = PairNorm(mode=pairnorm)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(ref_df, device=device,in_feats=in_feats,cat_features=cat_features)
        else:
            self.n2v_mlp = lambda x: x
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(n_classes+1, in_feats, padding_idx=n_classes))
        self.layers.append(nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Linear(self.in_feats, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim*self.heads[0], in_feats)
        ))

        # build multiple layers
        self.layers.append(TransformerConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        if post_proc:
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                                             nn.BatchNorm1d(self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes))

    def forward(self, blocks, features, labels, n2v_feat=None):
        """
        :param blocks: train blocks
		:param features: train features
		:param labels: train labels
        :param n2v_feat: whether to use n2v features
        """

        if n2v_feat is None:
            h = features
        else:
            h = self.n2v_mlp(n2v_feat)
            h = features + h
        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](h) + self.layers[2](label_embed)
        #label_embed = self.layers[1](h)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed
        # print(h)

        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l+4](blocks[l], h))

        logits = self.layers[-1](h)

        return logits
