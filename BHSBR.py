from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class HGNN(nn.Module):
    def __init__(self, n_hid, dropout=0.2):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.out_fc = nn.Linear(n_hid, n_hid)

    def forward(self, x, G):
        x1 = self.hgc1(x, G)
        # x1 = F.relu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.hgc2(x1, G)
        # x2 = F.relu(x2)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        return ((x1+x2)/2)
        return self.out_fc((x1+x2)/2)


class HGNN_conv(nn.Module):
    def __init__(self, n_hid, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(n_hid, n_hid))
        nn.init.normal_(self.weight, std=0.02)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_hid))
            nn.init.normal_(self.bias, std=0.02)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


class ItemConv(nn.Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.w_item = {}
        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0.unsqueeze(0)]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % (i)])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final_embeddings = item_embeddings.unsqueeze(0)
            final.append(F.normalize(final_embeddings, dim=-1, p=2))
        final = torch.cat(final, dim=0)
        # item_embeddings = np.sum(final, 0)/(self.layers+1)
        item_embeddings = torch.mean(final, dim=0)
        item_embeddings = torch.cuda.FloatTensor(item_embeddings)
        return item_embeddings

class HierarchicalGNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(HierarchicalGNN, self).__init__()
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True)
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = nn.Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=True)

    def GNNCell(self, A, hidden, macro_items, micro_actions, action_len):

        batch_size, n_edges, n_actions = micro_actions.size(0), micro_actions.size(1), micro_actions.size(2)
        action_len_ = action_len.view(batch_size * n_edges)
        packed = pack_padded_sequence(micro_actions.view(batch_size * n_edges, n_actions, self.embedding_size),
                                      action_len_.tolist(), batch_first=True, enforce_sorted=False)
        _, micro_actions_h = self.gru(packed)  # 1, B*n_edges, dim
        micro_actions_h = micro_actions_h.view(batch_size, n_edges, self.embedding_size)  # B, n_edges, dim
        # macro_hidden = macro_items + micro_actions_h
        macro_hidden = torch.cat([macro_items, micro_actions_h], 2)
        _in = self.linear_edge_in(macro_hidden)
        _out = self.linear_edge_out(macro_hidden)
        input_in = torch.matmul(A[:, :, :n_edges], _in) + self.b_iah
        input_out = torch.matmul(A[:, :, n_edges:], _out) + self.b_ioh

        inputs = torch.cat([input_in, input_out], 2)  #

        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)

        i_r, i_i, i_n = gi.chunk(3, 2)  # 在二维上拆分成3份
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = (1 - inputgate) * hidden + inputgate * newgate
        return hy

    def forward(self, A, hidden, macro_items, micro_actions, micro_len):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden, macro_items, micro_actions, micro_len)
        return hidden


class BHSBR(nn.Module):
    def __init__(self, n_items, adj, n_actions, n_pos, n_action_pairs, x_dim, a_dim, pos_dim, hidden_dim,
                 dropout_rate=0, alpha=12, step=1):
        super(BHSBR, self).__init__()
        self.item_embeddings = nn.Embedding(n_items, x_dim, padding_idx=0)
        self.action_embeddings = nn.Embedding(n_actions, a_dim, padding_idx=0)
        self.pos_embeddings = nn.Embedding(n_pos, pos_dim, padding_idx=0)
        self.action_pairs_embeddings = nn.Embedding(n_action_pairs, a_dim, padding_idx=0)
        self.x_dim = x_dim
        self.hidden_size = hidden_dim
        self.embedding_size = hidden_dim

        self.n_actions = n_actions

        self.adj = adj
        self.layers = 2
        self.ItemGraph = ItemConv(self.layers)

        self.hgnn_layer = HGNN(self.x_dim)
        self.glu1 = nn.Linear(hidden_dim, hidden_dim)
        self.glu2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_s = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))

        self.dropout = nn.Dropout(dropout_rate)
        # self.input_size = x_dim + pos_dim + a_dim

        self.gnn = HierarchicalGNN(hidden_dim)


        # self.gru_relation = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True)

        self.W_q_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_k_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_q_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_k_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.W_g = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=True)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        #self.linear_three = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        #self.linear_zero = nn.Linear(self.embedding_size, 1, bias=False)
        #self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)
        # parameters initialization
        self.step = step  #
        self._reset_parameters()
        self.alpha = alpha

    def _reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(0, 0.1)

    def generate_session(self, sessions, hidden):
        sess_len = torch.count_nonzero(sessions, dim=1).tolist()
        session_emb = torch.div(torch.sum(hidden, 1), sess_len)
        len = session_emb[1]
        hs = session_emb.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(hidden) + self.glu2(hs))
        # beta = torch.matmul(nh, self.w_s)
        sess_emb = torch.sum(nh, 1)
        return sess_emb


    def forward(self, items, A, alias_inputs, item_seq_len, macro_items,
                micro_actions, micro_len, actions, action_pairs, pos):
        item_embeddings = self.ItemGraph(self.adj, self.item_embeddings.weight)
        seq_len = alias_inputs.size(1) + 1
        actions = actions[:, :seq_len]
        pos = pos[:, :seq_len]
        action_pairs = action_pairs[:, :seq_len, :seq_len]
        # hidden, action_pairs_embedding = item_embeddings[items], self.action_pairs_embeddings(action_pairs)
        hidden, action_pairs_embedding = self.item_embeddings(items), self.action_pairs_embeddings(action_pairs)
        actions_embedding = self.action_embeddings(actions)
        pos_embedding = self.pos_embeddings(pos)
        # macro_items_embedding = item_embeddings[macro_items]  # B, n_edges, dim
        macro_items_embedding = self.item_embeddings(macro_items)  # B, n_edges, dim
        micro_actions_embedding = self.action_embeddings(micro_actions)  # B, n_edges, n_micro_a, dim
        mask = alias_inputs.gt(-1)
        h_0 = hidden
        hidden_mask = items.gt(0)
        length = torch.sum(hidden_mask, 1).unsqueeze(1)

        star_node = torch.sum(hidden, 1) / length  # B,d
        # star_0 = star_node
        for i in range(self.step):
            hidden, star_node = self.update_item_layer(hidden, star_node.squeeze(), A, macro_items_embedding,
                                                       micro_actions_embedding, micro_len)  # batch, hidden_size
        h_f = self.highway_network(hidden, h_0)

        # action_output, action_hidden = self.gru_relation(actions_embedding, None)
        alias_inputs = alias_inputs.masked_fill(mask == False, 0)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.hidden_size)
        new_mask = actions.gt(0)
        seq_hidden = torch.gather(h_f, dim=1, index=alias_inputs)
        seq_hidden = torch.cat([star_node, seq_hidden], dim=1)
        # self_input = seq_hidden + pos_embedding + actions_embedding
        self_input = seq_hidden + actions_embedding
        # item_seq_len 是没有special 的序列长度
        ht = self.gather_indexes(self_input, item_seq_len)

        Gs = self.build_Gs_unique(macro_items, micro_actions)
        batch_size = macro_items.shape[0]
        n_objs = torch.count_nonzero(macro_items, dim=1).tolist()
        indexed_emb = list()
        for batch_idx in range(batch_size):
            n_obj = n_objs[batch_idx]
            indexed_emb.append(item_embeddings[macro_items[batch_idx][:n_obj]])
        indexed_emb = torch.cat(indexed_emb, dim=0)
        hgnn_emb =self.hgnn_layer(indexed_emb, Gs)
        emb_list = []
        start = 0
        for batch_idx in range(batch_size):
            n_obj = n_objs[batch_idx]
            embs = hgnn_emb[start:start+n_obj]
            start += n_obj

            embs = torch.mean(embs, dim=0)
            emb_list.append(embs)
        hgnn_emb = torch.stack(emb_list, dim=0)

        seq_out = torch.stack((ht, hgnn_emb), dim=0)
        weights = (torch.matmul(seq_out, self.attn_weights.unsqueeze(0).unsqueeze(0)) * self.attn).sum(-1)
        # 2, b, l, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_out = (seq_out * score).sum(0)

        # seq_out = ht

        c = seq_out.squeeze()
        c = torch.mean(c, dim=0)
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))  # \hat{m}
        l_emb = self.item_embeddings.weight[1:] / torch.norm(self.item_embeddings.weight[1:], dim=-1).unsqueeze(1)
        # l_emb = self.item_embeddings.weight[1:] / torch.norm(self.item_embeddings.weight[1:], dim=-1).unsqueeze(1) # v_i
        y = self.alpha * torch.matmul(l_c, l_emb.t())
        return y

    def update_item_layer(self, h_last, star_node_last, A, macro_items_e, micro_actions_e, micro_len):
        hidden = self.gnn(A, h_last, macro_items_e, micro_actions_e, micro_len)
        q_one = self.W_q_one(hidden)  # B, L, d
        k_one = self.W_k_one(star_node_last)  # B, d
        alpha_i = torch.bmm(q_one, k_one.unsqueeze(2)) / math.sqrt(self.embedding_size)  # B, L, 1
        new_h = (1 - alpha_i) * hidden + alpha_i * star_node_last.unsqueeze(1)  # B, L, d

        q_two = self.W_q_two(star_node_last)
        k_two = self.W_k_two(new_h)
        beta = torch.softmax(torch.bmm(k_two, q_two.unsqueeze(2)) / math.sqrt(self.embedding_size), 1)  # B, L, 1
        new_star_node = torch.bmm(beta.transpose(1, 2), new_h)  # B, 1, d

        return new_h, new_star_node
        #return hidden, star_node_last

    def highway_network(self, hidden, h_0):
        g = torch.sigmoid(self.W_g(torch.cat((h_0, hidden), 2)))  # B,L,d
        h_f = g * h_0 + (1 - g) * hidden
        return h_f

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def build_Gs_unique(self, seqs, actions):
        Gs = []
        n_objs = torch.count_nonzero(seqs, dim=1).tolist()
        for batch_idx in range(seqs.shape[0]):
            seq = seqs[batch_idx]
            n_obj = n_objs[batch_idx]
            seq = seq[:n_obj].cpu()
            seq_list = seq.tolist()

            seq_actions = actions[batch_idx]

            n_edge = self.n_actions
            # hyper graph: n_obj, n_edge
            H = torch.full((n_obj, n_edge), 0.0001, device=seq.device)

            for i, item in enumerate(seq_list):
                item_action = seq_actions[i]
                j = 0
                while j < len(item_action) and item_action[j] != 0:
                    k = item_action[j]-1
                    H[i, k] += 1.0
                    j += 1
                # multi-behavior hyperedge
            # print(H.detach().cpu().tolist())
            # W = torch.ones(n_edge, device=H.device)
            # W = torch.diag(W)
            DV = torch.sum(H, dim=1)
            DE = torch.sum(H, dim=0)
            invDE = torch.diag(torch.pow(DE, -1))
            invDV = torch.diag(torch.pow(DV, -1))
            # DV2 = torch.diag(torch.pow(DV, -0.5))
            HT = H.t()
            G = invDV.mm(H).mm(invDE).mm(HT)
            # G = DV2.mm(H).mm(invDE).mm(HT).mm(DV2)
            assert not torch.isnan(G).any()
            Gs.append(G.to(seqs.device))
        Gs_block_diag = torch.block_diag(*Gs)
        return Gs_block_diag