"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


from lib.modules.blocks.decoder_layer import DecoderLayer
from lib.modules.embedding.transformer_embedding import TransformerEmbedding


class ConceptDecoder(nn.Module):
    def __init__(self, d_output, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, d_output)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        for layer in self.layers:
            # trg = layer(trg, enc_src, trg_mask, src_mask)
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output, attention




# origin decoder
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation 
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen, 2020
# ------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_graph):
        nodes_q = self.query(input_graph)
        nodes_k = self.key(input_graph)
        nodes_v = self.value(input_graph)

        nodes_q_t = self.transpose_for_scores(nodes_q)
        nodes_k_t = self.transpose_for_scores(nodes_k)
        nodes_v_t = self.transpose_for_scores(nodes_v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(nodes_q_t, nodes_k_t.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in GATModel forward() function)
        attention_scores = attention_scores 

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        nodes_new = torch.matmul(attention_probs, nodes_v_t)
        nodes_new = nodes_new.permute(0, 2, 1, 3).contiguous()
        new_nodes_shape = nodes_new.size()[:-2] + (self.all_head_size,)
        nodes_new = nodes_new.view(*new_nodes_shape)
        return nodes_new


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

    def forward(self, input_graph):
        attention_output = self.mha(input_graph) # multi-head attention
        attention_output = self.fc_in(attention_output)
        attention_output = self.dropout_in(attention_output)
        attention_output = self.bn_in((attention_output + input_graph).permute(0, 2, 1)).permute(0, 2, 1)
        intermediate_output = self.fc_int(attention_output)
        intermediate_output = F.relu(intermediate_output)
        intermediate_output = self.fc_out(intermediate_output)
        intermediate_output = self.dropout_out(intermediate_output)
        graph_output = self.bn_out((intermediate_output + attention_output).permute(0, 2, 1)).permute(0, 2, 1)
        return graph_output

