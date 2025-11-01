import math
import pandas as pd
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import numpy as np
from utils import get_activation, to_etype_name

th.set_printoptions(profile="full")


class GDC(nn.Module):
    def __init__(self, alpha=0.2):
        super(GDC, self).__init__()
        self.alpha = alpha

    def ppr(self, adj):

        if adj.is_sparse:
            adj = adj.to_dense()

        n = adj.size(0)
        d = adj.sum(dim=1)
        d_inv_sqrt = d.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        norm_adj = adj * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)


        identity = th.eye(n, device=adj.device)
        ppr_matrix = self.alpha * th.linalg.inv(identity - (1 - self.alpha) * norm_adj + 1e-5 * identity)

        return ppr_matrix


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, top_k, use_bn=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.top_k = top_k
        self.dropout = dropout
        self.use_bn = use_bn
        self.weight_strong = Parameter(th.FloatTensor(in_features, out_features))
        self.weight_weak = Parameter(th.FloatTensor(in_features, out_features))
        self.beta = nn.Parameter(th.FloatTensor(1))
        self.gamma = nn.Parameter(th.FloatTensor(1))
        self.beta.data.fill_(0.5)
        self.gamma.data.fill_(0.5)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_strong.size(1))
        self.weight_strong.data.uniform_(-stdv, stdv)
        self.weight_weak.data.uniform_(-stdv, stdv)
    def split_adj(self, adj):
        dense_adj = adj.to_dense()
        N = dense_adj.size(0)
        strong_adj = th.zeros_like(dense_adj)
        weak_adj = th.zeros_like(dense_adj)

        for i in range(N):
            vals, indices = th.topk(dense_adj[i], self.top_k)
            strong_adj[i, indices] = dense_adj[i, indices]
            weak_adj[i] = dense_adj[i] - strong_adj[i]

        return strong_adj.to_sparse(), weak_adj.to_sparse()
    def forward(self, x, adj):
        strong_adj, weak_adj = self.split_adj(adj)

        output_strong = th.matmul(strong_adj, x)
        output_weak = th.matmul(weak_adj, x)
        output = self.beta * output_strong + self.gamma * output_weak
        #x = th.matmul(adj, output)
        x = self.fc(output)
        if self.use_bn:
            x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class FGCN(nn.Module):
    def __init__(self, fdim_drug, fdim_disease, nhid1, nhid2, dropout, top_k, alpha=0.2):
        super(FGCN, self).__init__()
        self.gdc = GDC(alpha=alpha)

        self.gcn1_drug = GCNLayer(fdim_drug, nhid1, dropout, top_k)
        self.gcn2_drug = GCNLayer(nhid1, nhid2, dropout, top_k)

        self.gcn1_disease = GCNLayer(fdim_disease, nhid1, dropout, top_k)
        self.gcn2_disease = GCNLayer(nhid1, nhid2, dropout, top_k)

        self.dropout = dropout

    def forward(self, drug_graph, drug_sim_feat, dis_graph, disease_sim_feat):
        drug_ppr = self.gdc.ppr(drug_graph)
        drug_feat = th.matmul(drug_ppr, drug_sim_feat)

        drug_feat = self.gcn1_drug(drug_feat, drug_graph)
        drug_feat = self.gcn2_drug(drug_feat, drug_graph)

        dis_ppr = self.gdc.ppr(dis_graph)
        dis_feat = th.matmul(dis_ppr, disease_sim_feat)

        dis_feat = self.gcn1_disease(dis_feat, dis_graph)
        dis_feat = self.gcn2_disease(dis_feat, dis_graph)

        return drug_feat, dis_feat


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = th.mm(input, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),  # in_size=75
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = th.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class GCMCGraphConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats  # 909
        self._out_feats = out_feats  # 600
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        # init.xavier_uniform_(self.att)

    def forward(self, graph, feat, weight=None, Two_Stage=False):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat  # dst feature not used [drug or disease num , 3]
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise dgl.DGLError('External weight is provided while at the same time the'
                                       ' module has defined its own weight parameter. Please'
                                       ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCMCLayer(nn.Module):

    def __init__(self, rating_vals,  # [0, 1]
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,  # 75
                 dropout_rate=0.0,  # 0.3
                 agg='stack',  # 'sum'
                 agg_act=None,  # Tanh()
                 share_user_item_param=False,  # True
                 basis_units=4, device=None):  # True 4
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals  # [0, 1]
        self.agg = agg  # sum
        self.share_user_item_param = share_user_item_param  # True
        self.ufc = nn.Linear(msg_units, out_units)  # Linear(in_features=1800, out_features=75, bias=True)
        self.user_in_units = user_in_units  # 909
        self.msg_units = msg_units  # 1800
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            # divide the original msg unit size by number of rel_values to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)

        msg_units = msg_units // 3  # 600
        self.msg_units = msg_units  # 600
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = {}
        subConv = {}
        self.basis_units = basis_units  # 4
        self.att = nn.Parameter(th.randn(len(self.rating_vals), basis_units))  # [2, 4]
        self.basis = nn.Parameter(th.randn(basis_units, user_in_units, msg_units))  # [4, 909, 600]
        for i, rating in enumerate(rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                subConv[rating] = GCMCGraphConv(user_in_units, msg_units, weight=False, device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(user_in_units, msg_units, weight=False, device=device,
                                                    dropout_rate=dropout_rate)
            else:
                subConv[rating] = GCMCGraphConv(user_in_units, msg_units, weight=True, device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units, msg_units, weight=True, device=device,
                                                    dropout_rate=dropout_rate)

        subConv['similar'] = GCMCGraphConv(user_in_units, msg_units, weight=True, device=device,
                                           dropout_rate=dropout_rate)

        subConv['disease-similar'] = GCMCGraphConv(movie_in_units, msg_units, weight=True, device=device,
                                                   dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, drug_feat=None, dis_feat=None, Two_Stage=False):
        in_feats = {'drug': drug_feat, 'disease': dis_feat}
        mod_args = {}
        self.W = th.matmul(self.att, self.basis.view(self.basis_units, -1))
        self.W = self.W.view(-1, self.user_in_units, self.msg_units)
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating

            mod_args[rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)
            mod_args[rev_rating] = (self.W[i, :, :] if self.W_r is not None else None, Two_Stage)
            # Similarity edge arguments
        mod_args['similar'] = (None, Two_Stage)
        mod_args['disease-similar'] = (None, Two_Stage)
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        drug_feat = out_feats['drug']
        dis_feat = out_feats['disease']

        if in_feats['disease'].shape == dis_feat.shape:
            ufeat = dis_feat.view(dis_feat.shape[0], -1)
            ifeat = drug_feat.view(drug_feat.shape[0], -1)

        drug_feat = self.agg_act(drug_feat)
        drug_feat = self.dropout(drug_feat)

        dis_feat = self.agg_act(dis_feat)
        dis_feat = self.dropout(dis_feat)

        drug_feat = self.ifc(drug_feat)
        dis_feat = self.ufc(dis_feat)

        return drug_feat, dis_feat


class MLPDecoder(nn.Module):
    def __init__(self,
                 in_units,
                 dropout_rate=0.2):
        super(MLPDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.lin1 = nn.Linear(2 * in_units, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)
        self.lin4 = nn.Linear(64, 2)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        #self.lin4.reset_parameters()
    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            graph.apply_edges(udf_u_mul_e)
            out = graph.edata['m']

            out = F.relu(self.lin1(out))

            out = self.dropout(out)

            out = F.relu(self.lin2(out))

            out = self.dropout(out)
            out = self.lin3(out)


        return out


def udf_u_mul_e_norm(edges):
    return {'reg': edges.src['reg'] * edges.dst['ci']}
    # out_feats = edges.src['reg'].shape[1] // 3 return {'reg' : th.cat([edges.src['reg'][:, :out_feats] * edges.dst[
    # 'ci'], edges.src['reg'][:, out_feats:out_feats*2], edges.src['reg'][:, out_feats*2:]], 1)}


def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], 1)}
    # return {'m': (edges.src['h']) * (edges.dst['h'])}


def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix. A feat, B weight
    # feat size [313, 3] weight size [909, 600]
    if A is None:
        return B
    elif A.shape[1] == 3:
        if device is None:
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1)
        else:
            # return th.cat([B[A[:, 0].long()], B[A[:, 2].long()]], 1).to(device)  # only train one-hop
            # return th.cat([B[A[:, 0].long()], B[A[:, 1].long()]], 1).to(device)  # only train two-hop
            # return B[A[:, 0].long()].to(device)
            return th.cat([B[A[:, 0].long()], B[A[:, 1].long()], B[A[:, 2].long()]], 1).to(device)
    else:
        return A


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layers = args.layers
        self._act = get_activation(args.model_activation)
        self.TGCN = nn.ModuleList()
        self.TGCN.append(GCMCLayer(args.rating_vals,
                                   args.src_in_units,
                                   args.dst_in_units,
                                   args.gcn_agg_units,
                                   args.gcn_out_units,
                                   args.dropout,
                                   args.gcn_agg_accum,
                                   agg_act=self._act,
                                   share_user_item_param=args.share_param,
                                   device=args.device))

        self.gcn_agg_accum = args.gcn_agg_accum
        self.rating_vals = args.rating_vals
        self.device = args.device
        self.gcn_agg_units = args.gcn_agg_units
        self.src_in_units = args.src_in_units

        for i in range(1, args.layers):
            if args.gcn_agg_accum == 'stack':
                gcn_out_units = args.gcn_out_units * len(args.rating_vals)
            else:
                gcn_out_units = args.gcn_out_units
            self.TGCN.append(GCMCLayer(args.rating_vals,
                                       args.gcn_out_units,
                                       args.gcn_out_units,
                                       gcn_out_units,
                                       args.gcn_out_units,
                                       args.dropout,
                                       args.gcn_agg_accum,
                                       agg_act=self._act,
                                       share_user_item_param=args.share_param,
                                       device=args.device))

        self.FGCN = FGCN(args.fdim_drug,
                         args.fdim_disease,
                         args.nhid1,
                         args.nhid2,
                         args.dropout,
                         args.top_k)

        self.attention = Attention(args.gcn_out_units)
        self.decoder = MLPDecoder(in_units=args.gcn_out_units)
        self.rating_vals = args.rating_vals
        self.abc = nn.Linear(75, 3)
        self.bbc = nn.Linear(75, 3)


    def forward(self, enc_graph, dec_graph,
                drug_graph, drug_sim_feat, drug_feat,
                dis_graph, disease_sim_feat, dis_feat,
                subgraphs,
                Two_Stage=False):

        all_drug_out_subgraphs = []
        all_dis_out_subgraphs = []

        for subgraph in subgraphs:
            drug_feat_sub, dis_feat_sub = drug_feat, dis_feat
            drug_out_subgraph, dis_out_subgraph = None, None
            for i in range(0, self.layers):
                drug_o_sub, dis_o_sub = self.TGCN[i](subgraph, drug_feat_sub, dis_feat_sub, Two_Stage)
                if i == 0:
                    drug_out_subgraph = drug_o_sub
                    dis_out_subgraph = dis_o_sub
                else:
                    drug_out_subgraph += drug_o_sub / float(i + 1)
                    dis_out_subgraph += dis_o_sub / float(i + 1)

                drug_feat_sub = drug_o_sub
                dis_feat_sub = dis_o_sub

            all_drug_out_subgraphs.append(drug_out_subgraph)
            all_dis_out_subgraphs.append(dis_out_subgraph)

        drug_out_subgraph_combined = sum(all_drug_out_subgraphs) / len(all_drug_out_subgraphs)
        dis_out_subgraph_combined = sum(all_dis_out_subgraphs) / len(all_dis_out_subgraphs)

        drug_out_combined = drug_out_subgraph_combined
        dis_out_combined = dis_out_subgraph_combined

        drug_feat = drug_out_combined
        dis_feat = dis_out_combined
        pred_ratings = self.decoder(dec_graph, drug_feat, dis_feat)

        return pred_ratings, drug_out_combined, dis_out_combined
