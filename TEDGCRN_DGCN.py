import torch
import torch.nn as nn
from collections import OrderedDict


class FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FC, self).__init__()

        self.hyperGNN_dim = 16
        self.middle_dim = 2

        self.mlp = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                ('sigmoid1', nn.Sigmoid()),
                ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                ('sigmoid2', nn.Sigmoid()),
                ('fc3', nn.Linear(self.middle_dim, dim_out))
            ])
        )

    def forward(self, x):
        ho = self.mlp(x)
        return ho


class TEDGCRNGraphConv(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super().__init__()

        self.cheb_k = cheb_k

        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k * 2 + 1, dim_in, dim_out)
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(cheb_k * 2 + 1, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, dim_out)
        )
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))

        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.time_dim = time_dim

        self.gcn = gcn(cheb_k)
        self.fc1 = FC(dim_in, time_dim)
        self.fc2 = FC(dim_in, time_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias_pool)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj, node_embedding):
        # 计算两个方向邻接矩阵对应的多阶图扩散特征
        x_g = self.gcn(x, adj)

        # 根据节点嵌入生成每个节点对应的卷积参数
        weights = torch.einsum(
            'nd,dkio->nkio',
            node_embedding,
            self.weights_pool
        )
        bias = torch.matmul(node_embedding, self.bias_pool)

        x_g = x_g.permute(0, 2, 1, 3)

        # 对不同阶数的图特征进行节点自适应映射
        x_gconv = torch.einsum(
            'bnki,nkio->bno',
            x_g,
            weights
        ) + bias

        return x_gconv


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # 按照邻接矩阵聚合相邻节点的特征
        x = torch.einsum("bnm,bmc->bnc", A, x)
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, k=2):
        super(gcn, self).__init__()

        self.nconv = nconv()
        self.k = k

    def forward(self, x, support):
        # 保留原始特征，并计算各方向邻接矩阵的多阶扩散结果
        out = [x]

        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)

            for k in range(2, self.k + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.stack(out, dim=1)
        return h