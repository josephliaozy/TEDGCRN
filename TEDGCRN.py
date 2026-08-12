import torch
import torch.nn as nn
from model.TEDGCRNCell import TEDGCRNCell
import numpy as np


class SpectralFilter(nn.Module):
    def __init__(self, dim, fft_len):
        super().__init__()
        self.fft_len = fft_len
        self.complex_weight = nn.Parameter(torch.randn(dim, fft_len, 2, dtype=torch.float32) * 0.02)
        self.amp_gate = nn.Sequential(
            nn.Linear(fft_len, fft_len),
            nn.ReLU(),
            nn.Linear(fft_len, fft_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N, T, D]
        B, N, T, D = x.shape

        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')  # [B, N, F, D]

        weight = torch.view_as_complex(self.complex_weight)  # [D, F]
        weight = weight.permute(1, 0).unsqueeze(0).unsqueeze(0)  # [1, 1, F, D]
        x_fft = x_fft * weight

        amplitude = x_fft.abs()
        amp_energy = amplitude.mean(dim=-1)  # [B, N, F]

        mask = self.amp_gate(amp_energy).unsqueeze(-1)  # [B, N, F, 1]
        x_fft = x_fft * mask.to(x_fft.dtype)

        return torch.fft.irfft(x_fft, n=T, dim=2, norm='ortho')


class MultiScaleConv1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_in = nn.Conv1d(dim, dim, kernel_size=1)

        self.conv_s = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv_m = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_l = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.proj_out = nn.Conv1d(dim * 3, dim, kernel_size=1)
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, N, T, D]
        B, N, T, D = x.shape
        x_in = x.reshape(B * N, T, D).permute(0, 2, 1)  # [B*N, D, T]

        x_in = self.proj_in(x_in)

        s = self.conv_s(x_in)
        m = self.conv_m(x_in)
        l = self.conv_l(x_in)

        out = torch.cat([s, m, l], dim=1)
        out = self.proj_out(out)
        out = self.norm(self.act(out + x_in))

        return out.permute(0, 2, 1).reshape(B, N, T, D)


class AdvancedTemporalBlock(nn.Module):
    def __init__(self, in_dim, out_dim, fft_len, drop=0.1):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Temporal hidden dimension must be at least 2.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.freq_module = SpectralFilter(out_dim, fft_len=fft_len)
        self.time_module = MultiScaleConv1D(out_dim)
        self.fusion_gate = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # x: [B, T, N, D_in]
        x = self.input_proj(x)
        x = x.permute(0, 2, 1, 3)  # [B, N, T, D_hidden]

        residual = x

        feat_freq = self.freq_module(x)
        feat_time = self.time_module(x)

        combined = torch.cat([feat_time, feat_freq], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        fused = feat_time * gate + feat_freq * (1 - gate)

        x = residual + self.dropout(fused)
        x = self.norm(x)
        return x.permute(0, 2, 1, 3)  # [B, T, N, D_hidden]


class TEDGCRNEncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super().__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        self.cells.append(TEDGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.cells.append(TEDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]     #x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]   #state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):   #如果有两层GRU，则第二层的GGRU的输入是前一层的隐藏状态
                state = self.cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :], node_embeddings[1][:, t, :], node_embeddings[2]])#state=[batch,steps,nodes,input_dim]
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state,[node_embeddings[0], node_embeddings[1]])
                inner_states.append(state)   #一个list，里面是每一步的GRU的hidden状态
            output_hidden.append(state)  #每层最后一个GRU单元的hidden状态
            current_inputs = torch.stack(inner_states, dim=1)
            #拼接成完整的上一层GRU的hidden状态，作为下一层GRRU的输入[batch,steps,nodes,hiddensize]
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class TEDGCRNDecoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super().__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        self.cells.append(TEDGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.cells.append(TEDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_inputs, init_state[i], [node_embeddings[0], node_embeddings[1], node_embeddings[2]])
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class TEDGCRN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.cl_decay_steps = args.lr_decay_step
        self.node_embeddings1 = nn.Parameter(torch.empty(self.num_node, args.embed_dim))
        self.T_i_D_emb1 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(7, args.time_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(7, args.time_dim))

        self.encoder = TEDGCRNEncoder(args.num_nodes, args.time_dim, args.rnn_units, args.cheb_k,
                                       args.embed_dim, args.time_dim, args.num_layers)
        self.decoder = TEDGCRNDecoder(args.num_nodes, args.output_dim, args.rnn_units, args.cheb_k,
                                       args.embed_dim, args.time_dim, args.num_layers)
        #predictor
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        fft_len = args.lag // 2 + 1
        self.temporal_enhance = AdvancedTemporalBlock(
            in_dim=self.input_dim,
            out_dim=args.time_dim,
            fft_len=fft_len,
            drop=0.2
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize standard layers without overwriting spectral filters."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.node_embeddings1)
        nn.init.xavier_uniform_(self.T_i_D_emb1)
        nn.init.xavier_uniform_(self.D_i_W_emb1)
        nn.init.xavier_uniform_(self.T_i_D_emb2)
        nn.init.xavier_uniform_(self.D_i_W_emb2)

    def forward(self, source, traget=None, batches_seen=None):
        #source: B, T_1, N, D
        #target: B, T_2, N, D


        source_raw = source
        target_raw = traget
        t_i_d_data1 = source_raw[..., 0, -2]
        t_i_d_data2 = target_raw[..., 0, -2]
        # T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        T_i_D_emb1_en = self.T_i_D_emb1[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb2_en = self.T_i_D_emb2[(t_i_d_data1 * 288).type(torch.LongTensor)]

        T_i_D_emb1_de = self.T_i_D_emb1[(t_i_d_data2 * 288).type(torch.LongTensor)]
        T_i_D_emb2_de = self.T_i_D_emb2[(t_i_d_data2 * 288).type(torch.LongTensor)]
        if self.use_W:
            d_i_w_data1 = source_raw[..., 0, -1]
            d_i_w_data2 = target_raw[..., 0, -1]
            # D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            D_i_W_emb1_en = self.D_i_W_emb1[(d_i_w_data1).type(torch.LongTensor)]
            D_i_W_emb2_en = self.D_i_W_emb2[(d_i_w_data1).type(torch.LongTensor)]

            D_i_W_emb1_de = self.D_i_W_emb1[(d_i_w_data2).type(torch.LongTensor)]
            D_i_W_emb2_de = self.D_i_W_emb2[(d_i_w_data2).type(torch.LongTensor)]

            node_embedding_en1 = torch.mul(T_i_D_emb1_en, D_i_W_emb1_en)
            node_embedding_en2 = torch.mul(T_i_D_emb2_en, D_i_W_emb2_en)

            node_embedding_de1 = torch.mul(T_i_D_emb1_de, D_i_W_emb1_de)
            node_embedding_de2 = torch.mul(T_i_D_emb2_de, D_i_W_emb2_de)
        else:
            node_embedding_en1 = T_i_D_emb1_en
            node_embedding_en2 = T_i_D_emb2_en

            node_embedding_de1 = T_i_D_emb1_de
            node_embedding_de2 = T_i_D_emb2_de


        en_node_embeddings=[node_embedding_en1, node_embedding_en2, self.node_embeddings1]

        source_value = source_raw[..., :self.input_dim]
        # Temporal enhancement is an intrinsic part of TEDGCRN. Its output
        # enters each encoder cell and therefore participates directly in
        # both sample-specific graph generation and recurrent prediction.
        source = self.temporal_enhance(source_value)

        init_state = self.encoder.init_hidden(source.shape[0]).to(source.device)  # [2,64,307,64] 前面是2是因为有两层GRU
        state, _ = self.encoder(source, init_state, en_node_embeddings)  # B, T, N, hidden
        state = state[:, -1:, :, :].squeeze(1)

        ht_list = [state] * self.num_layers

        go = torch.zeros((source.shape[0], self.num_node, self.output_dim), device=source.device)
        out = []
        for t in range(self.horizon):
            state, ht_list = self.decoder(go, ht_list, [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1])
            go = self.proj(state)
            out.append(go)
            if self.training:     #这里的课程学习用了给予一定概率用真实值代替预测值来学习的教师-学生学习法（名字忘了，大概跟着有关）
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):  #如果满足条件，则用真实值代替预测值训练
                    go = traget[:, t, :, 0].unsqueeze(-1)
        output = torch.stack(out, dim=1)


        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x


