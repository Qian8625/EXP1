import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


# ===== 多头注意力（融合共线频率） =====
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 可学习系数，调节结构信息权重
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [B, H, N, d_k]

    def forward(self, input_graph, adj):
        """
        input_graph: [B, N, D]
        adj: [N, N] (单图) or [B, N, N] (批量图)，已归一化连接强度
        """
        nodes_q = self.query(input_graph)
        nodes_k = self.key(input_graph)
        nodes_v = self.value(input_graph)

        nodes_q_t = self.transpose_for_scores(nodes_q)
        nodes_k_t = self.transpose_for_scores(nodes_k)
        nodes_v_t = self.transpose_for_scores(nodes_v)

        # QK^T / sqrt(d_k)
        attention_scores = torch.matmul(nodes_q_t, nodes_k_t.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 扩展邻接矩阵
        if adj.dim() == 2:
            adj_expand = adj.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        elif adj.dim() == 3:
            adj_expand = adj.unsqueeze(1)  # [B, 1, N, N]
        else:
            raise ValueError("adj must be (N, N) or (B, N, N)")

        # 融合结构信息
        attention_scores = attention_scores + self.beta * adj_expand

        # mask 非邻居
        mask = (adj_expand > 0).float()
        attention_scores = attention_scores.masked_fill(mask == 0, -9e15)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        nodes_new = torch.matmul(attention_probs, nodes_v_t)
        nodes_new = nodes_new.permute(0, 2, 1, 3).contiguous()
        new_nodes_shape = nodes_new.size()[:-2] + (self.all_head_size,)
        nodes_new = nodes_new.view(*new_nodes_shape)
        return nodes_new  # [B, N, D]


# ===== 融合结构信息的 GAT 层 =====
class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.mha = MultiHeadAttention(config)

        self.fc_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_in = nn.BatchNorm1d(config.hidden_size)
        self.dropout_in = nn.Dropout(config.hidden_dropout_prob)

        self.fc_int = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn_out = nn.BatchNorm1d(config.hidden_size)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_graph, adj):
        attention_output = self.mha(input_graph, adj)
        attention_output = self.fc_in(attention_output)
        attention_output = self.dropout_in(attention_output)
        attention_output = self.bn_in((attention_output + input_graph).permute(0, 2, 1)).permute(0, 2, 1)

        intermediate_output = self.fc_int(attention_output)
        intermediate_output = F.relu(intermediate_output)
        intermediate_output = self.fc_out(intermediate_output)
        intermediate_output = self.dropout_out(intermediate_output)

        graph_output = self.bn_out((intermediate_output + attention_output).permute(0, 2, 1)).permute(0, 2, 1)
        return graph_output


# ===== GAT 配置类 =====
class GATopt(object):
    def __init__(self, hidden_size, num_layers, num_attention_heads=8, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


# ===== 最终的 MMGCN_Enc（保留原始特征融合） =====
class MMGCN_Enc(nn.Module):
    def __init__(self, graph_embed_size, num_layers=2, dropout=0.2):
        super(MMGCN_Enc, self).__init__()
        config_all = GATopt(graph_embed_size, num_layers=num_layers)
        self.gat_layers = nn.ModuleList([GATLayer(config_all) for _ in range(num_layers)])

    def forward(self, text_embeddings, visual_embeddings, adj_all):
        """
        text_embeddings: [N_text, D]
        visual_embeddings: [N_visual, D]
        adj_all: [N_text + N_visual, N_text + N_visual]  (共线频率已归一化)
        """
        # 融合文本与视觉节点特征
        original_features = torch.cat([text_embeddings, visual_embeddings], dim=0)
        node_features = original_features.unsqueeze(0)  # [1, N, D]

        # 迭代 GAT 更新
        for gat in self.gat_layers:
            node_features = gat(node_features, adj_all)

        # 残差加入原始特征
        node_features = node_features.squeeze(0) + original_features

        # L2 归一化
        final_codes = F.normalize(node_features, dim=-1)
        return final_codes