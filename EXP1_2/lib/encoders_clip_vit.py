"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from transformers import BertModel
from lib.MMGCN import MMGCN_Enc
from lib.modules.Vit_transformer_timm import VitTransformerFeatureExtractor

from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP
import torch.nn.functional as F
from torch.nn import Parameter
import logging
import pickle
import os
import math
import random

logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)
    
def get_text_encoder(embed_size, no_txtnorm=False):
    return EncoderText(embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(data_name, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = VitTransformerFeatureExtractor(backbone_source, backbone_path, fixed_blocks=0)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm)

    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(images) + features

        features, pool_weights = self.gpool(features, image_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn

        self.linear = nn.Linear(768, 1024)
        self.init_weights()

    def init_weights(self):
        """
        Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.linear.in_features +
                                  self.linear.out_features)
        self.linear.weight.data.uniform_(-r, r)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        base_features, output = self.backbone(images)
        base_features = self.linear(base_features)
        base_features = l2norm(base_features, dim=-1)
        return base_features, output


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert_base_uncased')
        self.linear = nn.Linear(768, embed_size)
        self.gpool = GPO(32, 32)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features, cap_emb, cap_len


class Encoder_CLIPgpo(nn.Module):
    def __init__(self, input_dim, embed_size, no_norm=False):
        super(Encoder_CLIPgpo, self).__init__()
        self.embed_size = embed_size
        self.no_norm = no_norm
        self.linear = nn.Linear(input_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.linear.in_features +
                                  self.linear.out_features)
        self.linear.weight.data.uniform_(-r, r)
        self.linear.bias.data.fill_(0)

    def forward(self, x):

        pooled_features = self.linear(x)

        return pooled_features



def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n



                  

class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
    
    def forward(self, tokens, attention_x, attention_y=None):
        """
        tokens: [B, L, C]
        attention_x: sim_self [B, L]
        attention_y: sim_cross [B, L]
        """
        B_v, L_v, C = tokens.size()

        # 如果 attention_y 没有提供，就直接用 attention_x
        if attention_y is None:
            score = attention_x
        else:
            # —— 引入你的动态权重公式 ——
            epsilon = 1e-6
            coverage_score = attention_y.mean(dim=1) / (attention_x.mean(dim=1) + epsilon)  # [B]
            alpha = torch.sigmoid(1 - coverage_score)  # 文本覆盖低 -> alpha高
            beta = 1 - alpha

            # 按 batch 融合两个相似性
            score = alpha.unsqueeze(1) * attention_x + beta.unsqueeze(1) * attention_y  # [B, L]

        # 保留多少 token
        num_keep_token = math.ceil(L_v * self.sparse_ratio)
    
        # top-k 选取
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        keep_policy = score_index[:, :num_keep_token]

        # 得到掩码
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        
        # 选取的 tokens
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # 未选取的 tokens
        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        
        # 对未保留 token 权重做 softmax
        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)

        # 融合保留外的 token 成一个额外 token
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True) 

        return select_tokens, extra_token, score_mask

class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        
        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, keeped_patches)
                        )
        
        self.scale = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, x, keep_policy=None):

        # (B, N, C) -> (B, N, N_s)
        weight = self.weight(x)

        #  (B, N, N_s) -> (B, N_s, N)
        weight = weight.transpose(2, 1) * self.scale       

        if keep_policy is not None:
            # keep_policy (B, N) -> (B, 1, N)
            keep_policy = keep_policy.unsqueeze(1)
            # increase a large number for mask patches
            weight = weight - (1 - keep_policy) * 1e10

        # learning a set of weight matrices
        weight = F.softmax(weight, dim=2)
        
        # (B, N_s, C)
        # multiply with patch features
        x = torch.bmm(weight, x)
        
        return x



    


class Denoise_Decoder(nn.Module):
    def __init__(self, seq_len_q, seq_len_k, hidden_dim):
        super(Denoise_Decoder, self).__init__()
        # MLP for updating Sab (输入为行向量，维度为seq_len_k)
        self.mlp_sab = nn.Sequential(
            nn.Linear(seq_len_k, hidden_dim),  # 输入：一行Sab分数（长度seq_len_k）
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len_k)   # 输出：修正后的行向量
        )
        # MLP for updating Saa (输入为行向量，维度为seq_len_q)
        self.mlp_saa = nn.Sequential(
            nn.Linear(seq_len_q, hidden_dim),  # 输入：一行Saa分数（长度seq_len_q）
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len_q)   # 输出：修正后的行向量
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, concept):
        concept = concept.expand(query.size(0), -1, -1)  # Shape: (batch_size, seq_len_k, d_model)
        
        # 计算点积相似性 Sab 和 Saa
        # 咩有
        d_tensor = query.size(-1)
        Sab = torch.matmul(query, concept.transpose(1, 2)) / math.sqrt(d_tensor)  # Shape: (batch_size, seq_len_q, seq_len_k)
        Saa = torch.matmul(query, query.transpose(1, 2)) / math.sqrt(d_tensor)  # Shape: (batch_size, seq_len_q, seq_len_q)
        
        # 更新Sab：逐行处理
        batch_size, seq_len_q, seq_len_k = Sab.shape
        Sab_updated = []
        for i in range(batch_size):
            Sab_updated.append(self.mlp_sab(Sab[i]))  # 输入一行Sab，输出修正后的行
        Sab_updated = torch.stack(Sab_updated, dim=0)  # Shape: (batch_size, seq_len_q, seq_len_k)
        
        # 更新Saa：逐行处理
        Saa_updated = []
        for i in range(batch_size):
            Saa_updated.append(self.mlp_saa(Saa[i]))  # 输入一行Saa，输出修正后的行
        Saa_updated = torch.stack(Saa_updated, dim=0)  # Shape: (batch_size, seq_len_q, seq_len_q)
        
        # 使用Saa的对角线作为权重
        Saa_diag = torch.diagonal(Saa_updated, dim1=1, dim2=2).unsqueeze(-1)  # Shape: (batch_size, seq_len_q, 1)
        weights = torch.sigmoid(Sab_updated * Saa_diag)  # Shape: (batch_size, seq_len_q, seq_len_k)
        # weights = torch.sigmoid(torch.bmm(Saa_updated, Sab_updated))
        
        
        # 最终相似性
        S_final = weights * Sab  # Shape: (batch_size, seq_len_q, seq_len_k)
        score_final = self.softmax(S_final)
        query_final = torch.matmul(score_final, concept)  # Shape: (batch_size, seq_len_q, d_model)
        
        return query_final



class Denoise_Decoder2(nn.Module):
    def __init__(self, seq_len_k, hidden_dim):
        super(Denoise_Decoder2, self).__init__()
        # MLP for updating Sab (输入为行向量，维度为seq_len_k)
        self.mlp_sab = nn.Sequential(
            nn.Linear(seq_len_k, hidden_dim),  # 输入：一行Sab分数（长度seq_len_k）
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len_k)   # 输出：修正后的行向量
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, concept):
        # A: (N, 1024), B: (N, 900, 1024)
        # 计算点积相似性 Sab
        d_model = query.size(-1)
        Sab = torch.matmul(query.unsqueeze(1), concept.transpose(1, 2)).squeeze(1) / math.sqrt(d_model)  # Shape: (N, 900)
        
        # 更新Sab：将每一行输入MLP
        Sab_updated = self.mlp_sab(Sab)  # Shape: (N, 900)
        
        # 归一化注意力分数
        score = self.softmax(Sab_updated)  # Shape: (N, 900)
        
        # 计算加权和（如果需要）
        query_final = torch.matmul(score.unsqueeze(1), concept).squeeze(1)  # Shape: (N, 1024)
        
        return query_final  # 返回分数和加权结果（可选）


class Denoise_Decoder3(nn.Module):
    def __init__(self, seq_len_k, hidden_dim):
        super(Denoise_Decoder3, self).__init__()
        # MLP for updating Sab (输入为行向量，维度为seq_len_k)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, concept):
        # A: (N, 1024), B: (N, 900, 1024)
        # 计算点积相似性 Sab
        d_model = query.size(-1)
        Sab = torch.matmul(query.unsqueeze(1), concept.transpose(1, 2)).squeeze(1) / math.sqrt(
            d_model)  # Shape: (N, 900)

        # 归一化注意力分数
        score = self.softmax(Sab)  # Shape: (N, 900)

        # 计算加权和（如果需要）
        query_final = torch.matmul(score.unsqueeze(1), concept).squeeze(1)  # Shape: (N, 1024)

        return query_final  # 返回分数和加权结果（可选）


class OSIT(nn.Module):
    def __init__(self, embed_size):
        super(OSIT, self).__init__()
        self.embed_size = embed_size
        # 参数修改
        self.cat_weight = 0.90 
        self.aggr_ratio = 0.5
        self.sparse_ratio = 0.8

        text_concept_nodes = np.load('../cvpr26/data/f30k_concepts_word2vec_300_bert.pkl', allow_pickle=True)
        visual_concept_nodes = np.load('../cvpr26/data/f30k_concepts_visual2vec_300.pkl', allow_pickle=True)
        self.text_concept_nodes = Parameter(torch.Tensor(text_concept_nodes), requires_grad = False)
        self.visual_concept_nodes =Parameter(torch.Tensor(visual_concept_nodes), requires_grad = False)
        self.num_text_nodes = self.text_concept_nodes.size(0)
        self.num_visual_nodes = self.visual_concept_nodes.size(0)

        with open(os.path.join('../cvpr26/data/adj_matrix_300.pkl'), 'rb') as f:
            adj_matrix = torch.Tensor(pickle.load(f))

        diag = torch.diag_embed(torch.diag(adj_matrix))
        adj_matrix = adj_matrix - diag
        self.adj_all = Parameter(adj_matrix / adj_matrix.sum(dim=1,keepdim=True), requires_grad = False)


        self.concept_enc = MMGCN_Enc(embed_size)

        self.text_linear = nn.Linear(self.text_concept_nodes.size(-1), embed_size)
        self.visual_linear = nn.Linear(self.visual_concept_nodes.size(-1), embed_size)

        # self.keeped_patches = int(196 * self.aggr_ratio * self.sparse_ratio)
        #
        # # sparse network
        # self.sparse_net = TokenSparse(embed_dim=self.embed_size,
        #                               sparse_ratio=self.sparse_ratio,
        #                               )
        # # aggregation network
        # self.aggr_net= TokenAggregation(dim=self.embed_size,
        #                                 keeped_patches=self.keeped_patches,
        #                                 )
        #
        # 图像分支（有 Saa）
        self.de_decoder_vis = Denoise_Decoder3(seq_len_k=900, hidden_dim=900)

        # 文本分支（无 Saa）
        self.de_decoder_txt = Denoise_Decoder3(seq_len_k=900, hidden_dim=900)

        # self.gpool_patch = GPO(32, 32)
        # self.gpool_concept = GPO(32, 32)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """

        r3 = np.sqrt(6.) / np.sqrt(self.text_linear.in_features +
                                  self.text_linear.out_features)
        self.text_linear.weight.data.uniform_(-r3, r3)
        self.text_linear.bias.data.fill_(0)

        r4 = np.sqrt(6.) / np.sqrt(self.visual_linear.in_features +
                                  self.visual_linear.out_features)
        self.visual_linear.weight.data.uniform_(-r4, r4)
        self.visual_linear.bias.data.fill_(0)

    def forward(self, base_features, t_features, g_t_features, g_cap_emb, global_feature=None):
        cls_feature = base_features[:, 0:1, :]
        base_features = base_features[:, 1:, :]

        visual_concept_basis = self.visual_linear(self.visual_concept_nodes)
        text_concept_basis = self.text_linear(self.text_concept_nodes)
        concept_basis = self.concept_enc(text_concept_basis, visual_concept_basis,self.adj_all)
        concept_basis = concept_basis.unsqueeze(dim=0)

        # with torch.no_grad():
        #     # (B_v, L_v, C) ->  (B_v, 1, C)
        #     img_spatial_glo_norm = torch.mean(base_features, dim=1)
        #     img_spatial_glo_norm = l2norm(img_spatial_glo_norm, dim=-1).unsqueeze(dim=1)
        #     # (B_v, L_v, C) -> (B_v, L_v)
        #     img_spatial_self_attention = (img_spatial_glo_norm * base_features).sum(dim=-1)

        # g_t_features = g_t_features.unsqueeze(dim=1)
        # img_spatial_cro_attention = (g_t_features * base_features).sum(dim=-1)
        #
        # if self.training:
        #     drop_prob = 0.3
        #     if random.random() < drop_prob:
        #         img_spatial_cro_attention = None
        # # patch 选择
        # select_tokens, extra_token, score_mask = self.sparse_net(tokens=base_features,
        #                                                         attention_x=img_spatial_self_attention,
        #                                                         attention_y=img_spatial_cro_attention,
        #                                                      )

        # patch 聚合
        # aggr_tokens = self.aggr_net(select_tokens)

        # 冗余聚合
        # keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)

        # keep_spatial_tokens = torch.cat((cls_feature, keep_spatial_tokens), dim=1)

        # keep_spatial_tokens = l2norm(keep_spatial_tokens, dim=-1)

        # 图像去噪模块
        # concept_v = self.de_decoder_vis(keep_spatial_tokens, concept_basis)


        # feat_lengths = torch.zeros(keep_spatial_tokens.size(0)).to(keep_spatial_tokens.device)
        # feat_lengths[:] = keep_spatial_tokens.size(1)

        # concept_v, _ = self.gpool_concept(concept_v, feat_lengths)
        # img_patch, _ = self.gpool_concept(keep_spatial_tokens, feat_lengths)

        # concept_v = torch.mean(concept_v, dim=1)
        # img_patch = torch.mean(keep_spatial_tokens, dim=1)

        # concept_v = l2norm(concept_v, dim=-1)
        # img_patch = l2norm(img_patch, dim=-1)

        # img_patch = torch.mean(base_features, dim=1)
        # img_patch = l2norm(img_patch, dim=-1)

        concept_v = self.de_decoder_vis(cls_feature, concept_basis)
        concept_v = l2norm(concept_v, dim=-1)

        emb_v = torch.sqrt(torch.tensor(self.cat_weight)) * cls_feature + torch.sqrt(
            torch.tensor(1 - self.cat_weight)) * concept_v
        emb_v = l2norm(emb_v, dim=-1)

        concept_t = self.de_decoder_txt(t_features, concept_basis)
        concept_t = l2norm(concept_t, dim=-1)

        emb_t = torch.sqrt(torch.tensor(self.cat_weight)) * t_features + torch.sqrt(
            torch.tensor(1 - self.cat_weight)) * concept_t
        emb_t = l2norm(emb_t, dim=-1)

        return emb_v, emb_t