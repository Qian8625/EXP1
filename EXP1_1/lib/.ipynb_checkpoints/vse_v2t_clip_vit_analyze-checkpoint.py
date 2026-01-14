"""VSE model with comprehensive training diagnostics"""
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from transformers import AutoProcessor

from lib.encoders_clip_vit import get_image_encoder, get_text_encoder, OSIT ,Encoder_CLIPgpo
from lib.loss import ContrastiveLoss

import logging
import matplotlib.pyplot as plt
from collections import defaultdict

logger = logging.getLogger(__name__)

class TrainingDiagnostics:
    """训练诊断工具"""
    
    def __init__(self):
        self.metrics = {
            'loss_history': [],
            'grad_norms': [],
            'feature_norms': {'img': [], 'txt': []},
            'similarity_stats': {'pos': [], 'neg': []}
        }
    
    def record_loss(self, loss_value):
        self.metrics['loss_history'].append(loss_value)
    
    def record_grad_norms(self, model):
        total_norm = 0
        for p in model.params:  # 使用model.params而不是model.parameters()
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.metrics['grad_norms'].append(total_norm)
    
    def record_feature_norms(self, img_emb, txt_emb):
        self.metrics['feature_norms']['img'].append(img_emb.norm(dim=-1).mean().item())
        self.metrics['feature_norms']['txt'].append(txt_emb.norm(dim=-1).mean().item())
    
    def record_similarity_stats(self, img_emb, txt_emb):
        # 计算正样本相似度（对角线）和负样本相似度
        similarities = torch.matmul(img_emb, txt_emb.t())
        pos_sim = torch.diag(similarities).mean().item()
        
        # 计算负样本相似度（非对角线）
        mask = torch.eye(similarities.size(0), device=similarities.device).bool()
        neg_sim = similarities.masked_select(~mask).mean().item()
        
        self.metrics['similarity_stats']['pos'].append(pos_sim)
        self.metrics['similarity_stats']['neg'].append(neg_sim)
    
    def plot_diagnostics(self):
        """绘制诊断图表"""
        if len(self.metrics['loss_history']) == 0:
            logger.warning("没有收集到训练数据")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 损失曲线
        axes[0, 0].plot(self.metrics['loss_history'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 梯度范数曲线
        axes[0, 1].plot(self.metrics['grad_norms'])
        axes[0, 1].set_title('Gradient Norms')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].grid(True)
        
        # 特征范数曲线
        axes[1, 0].plot(self.metrics['feature_norms']['img'], label='Image Features')
        axes[1, 0].plot(self.metrics['feature_norms']['txt'], label='Text Features')
        axes[1, 0].set_title('Feature Norms')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Feature Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 相似度统计
        axes[1, 1].plot(self.metrics['similarity_stats']['pos'], label='Positive Similarity')
        axes[1, 1].plot(self.metrics['similarity_stats']['neg'], label='Negative Similarity')
        axes[1, 1].set_title('Similarity Statistics')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Similarity')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_analysis(self):
        """打印诊断分析"""
        if not self.metrics['loss_history']:
            logger.warning("没有收集到训练数据")
            return
        
        current_loss = self.metrics['loss_history'][-1]
        initial_loss = self.metrics['loss_history'][0] if len(self.metrics['loss_history']) > 1 else current_loss
        loss_change = current_loss - initial_loss
        
        logger.info("=== 训练诊断分析 ===")
        logger.info(f"初始损失: {initial_loss:.6f}")
        logger.info(f"当前损失: {current_loss:.6f}")
        logger.info(f"损失变化: {loss_change:.6f}")
        
        if loss_change > 0:
            logger.warning("⚠️ 损失在增加，可能存在以下问题:")
            logger.warning("  - 学习率过高")
            logger.warning("  - 数据标签错误")
            logger.warning("  - 模型架构问题")
        elif abs(loss_change) < 1e-4 and len(self.metrics['loss_history']) > 10:
            logger.warning("⚠️ 损失几乎没有变化，可能存在以下问题:")
            logger.warning("  - 模型陷入局部最优")
            logger.warning("  - 学习率过低")
            logger.warning("  - 数据质量问题")
        
        # 检查特征范数
        if self.metrics['feature_norms']['img']:
            avg_img_norm = np.mean(self.metrics['feature_norms']['img'][-10:])
            avg_txt_norm = np.mean(self.metrics['feature_norms']['txt'][-10:])
            logger.info(f"平均图像特征范数: {avg_img_norm:.6f}")
            logger.info(f"平均文本特征范数: {avg_txt_norm:.6f}")
            
            if avg_img_norm > 10 or avg_txt_norm > 10:
                logger.warning("⚠️ 特征范数过大，可能导致数值不稳定")
        
        # 检查相似度
        if self.metrics['similarity_stats']['pos']:
            avg_pos_sim = np.mean(self.metrics['similarity_stats']['pos'][-10:])
            avg_neg_sim = np.mean(self.metrics['similarity_stats']['neg'][-10:])
            logger.info(f"平均正样本相似度: {avg_pos_sim:.6f}")
            logger.info(f"平均负样本相似度: {avg_neg_sim:.6f}")
            logger.info(f"相似度差距: {avg_pos_sim - avg_neg_sim:.6f}")
            
            if avg_pos_sim <= avg_neg_sim:
                logger.warning("⚠️ 正样本相似度小于等于负样本相似度，模型学习效果差")

class VSEModel(object):
    """
        The standard VSE model with comprehensive diagnostics
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)
        self.gen_enc = Encoder_CLIPgpo(
            input_dim=512,
            embed_size=opt.embed_size,
            no_norm=opt.no_txtnorm
        )

        self.osit = OSIT(opt.embed_size)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.gen_enc.cuda()
            self.osit.cuda()
            cudnn.benchmark = True

        # 初始化诊断工具
        self.diagnostics = TrainingDiagnostics()
        
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation,
                                         temperature=getattr(opt, 'temperature', 0.07))

        params = list(self.img_enc.parameters())
        params += list(self.txt_enc.parameters())
        params += list(self.gen_enc.parameters())
        params += list(self.osit.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        else:
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                
                # 使用更保守的学习率
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * 0.01},
                    {'params': self.img_enc.linear.parameters(), 'lr': opt.learning_rate},
                    {'params': self.osit.parameters(), 'lr': opt.learning_rate},
                    {'params': self.gen_enc.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, weight_decay=1e-4)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD([
                    {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.backbone_lr_factor,
                     'weight_decay': decay_factor},
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, momentum=0.9, nesterov=True)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.gen_enc.state_dict(), self.osit.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.gen_enc.load_state_dict(state_dict[2])
        self.osit.load_state_dict(state_dict[3])


    def train_start(self):
        """
        switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.gen_enc.train()
        self.osit.train()

    def val_start(self):
        """
        switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.gen_enc.eval()
        self.osit.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.gen_enc = nn.DataParallel(self.gen_enc)
        self.osit = nn.DataParallel(self.osit)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, generated_captions, generated_lengths, image_lengths=None):
        """
        Compute the image and caption embeddings
        """
        
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            generated_captions = generated_captions.cuda()

        if self.opt.precomp_enc_type == 'basic':
            if torch.cuda.is_available(): image_lengths = image_lengths.cuda()
            img_emb_features = self.img_enc(images, image_lengths)
        else:
            img_emb_features, _ = self.img_enc(images)


        orig_lengths_tensor = torch.Tensor(lengths).cuda()
        o_t_features, cap_emb, cap_lens = self.txt_enc(captions, orig_lengths_tensor)

        g_t_features = self.gen_enc(generated_captions)
        g_cap_emb = g_t_features.unsqueeze(1)

        emb_v, emb_t = self.osit(img_emb_features, o_t_features, g_t_features, g_cap_emb)

        return emb_v, emb_t
    

    def forward_loss(self, img_emb, cap_emb):
        """
        Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, generated_captions, generated_lengths, image_lengths=None, warmup_alpha=None):
        """
        One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths, generated_captions, generated_lengths, image_lengths=image_lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # 记录诊断信息
        self.diagnostics.record_loss(loss.item())
        self.diagnostics.record_feature_norms(img_emb, cap_emb)
        self.diagnostics.record_similarity_stats(img_emb, cap_emb)

        # compute gradient and update
        loss.backward()
        
        # 记录梯度范数
        self.diagnostics.record_grad_norms(self)
        
        # 检查梯度问题
        if self.Eiters % 100 == 0:
            # 打印诊断分析
            self.diagnostics.print_analysis()
            
            # 每500次迭代绘制诊断图
            if self.Eiters % 500 == 0:
                self.diagnostics.plot_diagnostics()

        # 严格控制梯度裁剪
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

    def analyze_training(self):
        """分析训练过程"""
        self.diagnostics.print_analysis()
        self.diagnostics.plot_diagnostics()
        return self.diagnostics.metrics