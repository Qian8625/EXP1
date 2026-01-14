"""VSE model"""
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

logger = logging.getLogger(__name__)

class VSEModel(object):
    """
        The standard VSE model
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
            cudnn.benchmark = False

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
                
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * 0.01},
                    {'params': self.img_enc.linear.parameters(), 'lr': opt.learning_rate},
                    {'params': self.osit.parameters(), 'lr': opt.learning_rate},
                    {'params': self.gen_enc.parameters(), 'lr': opt.learning_rate},
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
        
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
