from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TEDGCRN_DGCN import TEDGCRNGraphConv


class FC(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(dim_in, self.hyperGNN_dim)),
                    ("sigmoid1", nn.Sigmoid()),
                    ("fc2", nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                    ("sigmoid2", nn.Sigmoid()),
                    ("fc3", nn.Linear(self.middle_dim, dim_out)),
                ]
            )
        )

    def forward(self, x):
        return self.mlp(x)


class TEDGCRNCell(nn.Module):
    def __init__(
        self,
        node_num,
        dim_in,
        dim_out,
        cheb_k,
        embed_dim,
        time_dim,
    ):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out

        self.gate = TEDGCRNGraphConv(
            dim_in + self.hidden_dim,
            2 * dim_out,
            cheb_k,
            embed_dim,
            time_dim,
        )
        self.update = TEDGCRNGraphConv(
            dim_in + self.hidden_dim,
            dim_out,
            cheb_k,
            embed_dim,
            time_dim,
        )

        self.fc1 = FC(dim_in + self.hidden_dim, time_dim)
        self.fc2 = FC(dim_in + self.hidden_dim, time_dim)

    def forward(self, x, state, node_embeddings):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        filter1 = self.fc1(input_and_state)
        filter2 = self.fc2(input_and_state)

        # 根据当前输入和隐藏状态生成动态图节点向量
        nodevec1 = torch.tanh(
            torch.einsum(
                "bd,bnd->bnd",
                node_embeddings[0],
                filter1,
            )
        )
        nodevec2 = torch.tanh(
            torch.einsum(
                "bd,bnd->bnd",
                node_embeddings[1],
                filter2,
            )
        )

        # 构造反对称的节点关联矩阵
        affinity = torch.matmul(
            nodevec1,
            nodevec2.transpose(-2, -1),
        ) - torch.matmul(
            nodevec2,
            nodevec1.transpose(-2, -1),
        )

        # 分别保留两个方向的正关联，得到互补的有向图
        adj_forward = self.preprocessing(F.relu(affinity))
        adj_backward = self.preprocessing(
            F.relu(affinity.transpose(-2, -1))
        )
        adj = [adj_forward, adj_backward]

        # 利用动态图卷积计算门控状态
        z_r = torch.sigmoid(
            self.gate(
                input_and_state,
                adj,
                node_embeddings[2],
            )
        )
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)

        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(
            self.update(
                candidate,
                adj,
                node_embeddings[2],
            )
        )

        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size):
        return torch.zeros(
            batch_size,
            self.node_num,
            self.hidden_dim,
        )

    @staticmethod
    def preprocessing(adj):
        # 添加自环并进行行归一化
        num_nodes = adj.shape[-1]
        identity = torch.eye(
            num_nodes,
            dtype=adj.dtype,
            device=adj.device,
        )
        adj = adj + identity
        degree = adj.sum(dim=-1, keepdim=True)
        return adj / degree