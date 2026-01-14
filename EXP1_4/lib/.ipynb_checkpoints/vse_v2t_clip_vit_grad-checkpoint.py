"""VSE model with gradient monitoring and fixes for gradient explosion"""
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

class GradientMonitor:
    """梯度监控类，用于检查梯度传播情况"""
    
    def __init__(self, model_components):
        self.model_components = model_components  # 传入模型组件字典
        self.gradient_stats = defaultdict(list)
        self.hooks = []
        self.register_hooks()
        
    def register_hooks(self):
        """为模型的所有参数注册梯度钩子"""
        for comp_name, component in self.model_components.items():
            for name, param in component.named_parameters():
                if param.requires_grad:
                    full_name = f"{comp_name}.{name}"
                    hook = param.register_hook(
                        lambda grad, name=full_name: self._save_grad(grad, name)
                    )
                    self.hooks.append((full_name, hook))
    
    def _save_grad(self, grad, name):
        """保存梯度统计信息"""
        if grad is not None:
            grad_norm = torch.norm(grad).item()
            grad_mean = torch.mean(torch.abs(grad)).item()
            grad_std = torch.std(grad).item()
            
            self.gradient_stats[name].append({
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'zero_ratio': (grad == 0).sum().item() / grad.numel()
            })
    
    def get_gradient_stats(self):
        """获取梯度统计信息"""
        stats = {}
        for name, grad_records in self.gradient_stats.items():
            if grad_records:
                norms = [r['norm'] for r in grad_records]
                means = [r['mean'] for r in grad_records]
                stds = [r['std'] for r in grad_records]
                zero_ratios = [r['zero_ratio'] for r in grad_records]
                
                stats[name] = {
                    'last_norm': norms[-1],
                    'last_mean': means[-1],
                    'last_std': stds[-1],
                    'last_zero_ratio': zero_ratios[-1],
                    'avg_norm': np.mean(norms),
                    'avg_mean': np.mean(means),
                    'avg_std': np.mean(stds),
                    'avg_zero_ratio': np.mean(zero_ratios)
                }
        return stats
    
    def check_gradient_flow(self):
        """检查梯度传播情况"""
        stats = self.get_gradient_stats()
        problems = []
        
        for name, stat in stats.items():
            # 检查梯度是否为0（梯度消失）
            if stat['last_norm'] < 1e-6:  # 降低阈值以更好地检测问题
                problems.append(f"梯度消失: {name}, norm={stat['last_norm']:.6f}")
            
            # 检查梯度是否过大（梯度爆炸）
            if stat['last_norm'] > 100:  # 降低梯度爆炸阈值
                problems.append(f"梯度爆炸: {name}, norm={stat['last_norm']:.2f}")
            
            # 检查零梯度比例
            if stat['last_zero_ratio'] > 0.9:
                problems.append(f"零梯度过多: {name}, zero_ratio={stat['last_zero_ratio']:.2f}")
        
        return problems
    
    def print_gradient_analysis(self):
        """打印梯度分析报告"""
        stats = self.get_gradient_stats()
        if not stats:
            logger.warning("没有收集到梯度数据")
            return
        
        logger.info("=== 梯度分析报告 ===")
        logger.info(f"{'参数名':<50} {'梯度范数':<12} {'零梯度比例':<12} {'平均梯度':<12}")
        logger.info("-" * 90)
        
        for name, stat in stats.items():
            logger.info(f"{name:<50} {stat['last_norm']:<12.6f} {stat['last_zero_ratio']:<12.3f} {stat['last_mean']:<12.6f}")
    
    def clear_stats(self):
        """清空统计信息"""
        self.gradient_stats.clear()

class VSEModel(object):
    """
        The standard VSE model with gradient monitoring and fixes for gradient explosion
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

        # 初始化梯度监控器 - 传入模型组件字典
        self.gradient_monitor = GradientMonitor({
            'img_enc': self.img_enc,
            'txt_enc': self.txt_enc,
            'gen_enc': self.gen_enc,
            'osit': self.osit
        })
        
        # Loss and Optimizer
        # 保持原始的margin，但使用更稳定的学习率
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

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
                
                # 降低学习率，特别对BERT使用更小的学习率
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate * 0.1},  # 降低文本编码器学习率
                    {'params': bert_params, 'lr': opt.learning_rate * 0.01},  # 极大降低BERT学习率
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.backbone_lr_factor * 0.5},  # 降低骨干网络学习率
                    {'params': self.img_enc.linear.parameters(), 'lr': opt.learning_rate * 0.2},  # 降低线性层学习率
                    {'params': self.osit.parameters(), 'lr': opt.learning_rate * 0.5},  # 降低OSIT学习率
                    {'params': self.gen_enc.parameters(), 'lr': opt.learning_rate * 0.2},  # 降低生成编码器学习率
                ], lr=opt.learning_rate, weight_decay=decay_factor)
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

        # compute gradient and update
        loss.backward()
        
        # 检查梯度问题
        if self.Eiters % 100 == 0:  # 每100个iteration检查一次
            problems = self.gradient_monitor.check_gradient_flow()
            if problems:
                logger.warning(f"Iteration {self.Eiters}: 发现梯度问题:")
                for problem in problems:
                    logger.warning(f"  - {problem}")
            
            # 打印梯度分析
            if self.Eiters % 500 == 0:  # 每500个iteration打印详细分析
                self.gradient_monitor.print_gradient_analysis()
                self.gradient_monitor.clear_stats()

        # 严格控制梯度裁剪阈值，防止梯度爆炸
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, min(1.0, self.grad_clip))  # 严格限制梯度裁剪阈值
        self.optimizer.step()

    def analyze_gradients(self):
        """分析当前模型的梯度情况"""
        problems = self.gradient_monitor.check_gradient_flow()
        self.gradient_monitor.print_gradient_analysis()
        return problems

    def get_gradient_norms(self):
        """获取各模块的梯度范数"""
        norms = {}
        
        # 图像编码器梯度
        img_grad_norm = 0
        for param in self.img_enc.parameters():
            if param.grad is not None:
                img_grad_norm += param.grad.norm().item()
        norms['img_enc'] = img_grad_norm
        
        # 文本编码器梯度
        txt_grad_norm = 0
        for param in self.txt_enc.parameters():
            if param.grad is not None:
                txt_grad_norm += param.grad.norm().item()
        norms['txt_enc'] = txt_grad_norm
        
        # 生成编码器梯度
        gen_grad_norm = 0
        for param in self.gen_enc.parameters():
            if param.grad is not None:
                gen_grad_norm += param.grad.norm().item()
        norms['gen_enc'] = gen_grad_norm
        
        # OSIT模块梯度
        osit_grad_norm = 0
        for param in self.osit.parameters():
            if param.grad is not None:
                osit_grad_norm += param.grad.norm().item()
        norms['osit'] = osit_grad_norm
        
        return norms

    def check_gradient_flow(self):
        """检查梯度流动情况"""
        return self.gradient_monitor.check_gradient_flow()