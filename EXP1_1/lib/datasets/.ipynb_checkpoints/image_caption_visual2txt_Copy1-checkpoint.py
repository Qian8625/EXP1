"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2

import logging

logger = logging.getLogger(__name__)
np.random.seed(0)


def load_generated_captions(filepath: str) -> list:
    """
    Load generated captions from a file.
    """
    captions_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            captions_list.append(line.strip())
            
    print(f"Load {len(captions_list)} generated captions.")
    return captions_list

class RawImageDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenizer

        if hasattr(opt, 'generated_caps_path') and opt.generated_caps_path:
            self.generated_caps_list = load_generated_captions(opt.generated_caps_path)
        else:
            self.generated_caps_list = None
            if train:
                logger.warning("`generated_caps_path` is not provided in options. Generated captions will not be used in training.")

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')
        loc_mapping = osp.join(data_path, 'f30k_id_mapping.json')

        if 'coco' in data_name:
            self.image_base = osp.join(data_path, 'images')
        else:
            self.image_base = osp.join(data_path, 'images')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)

        # Read Captions (Original captions)
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())

        # Get the image ids
        with open(osp.join(loc_image, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        # Set related parameters according to the pre-trained backbone **
        assert 'backbone' in opt.precomp_enc_type

        self.backbone_source = opt.backbone_source
        self.base_target_size = 224
        self.crop_ratio = 0.875
        self.train_scale_rate = 1
        if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
            self.base_target_size = int(self.base_target_size * opt.input_scale_factor)
            logger.info('Input images are scaled by factor {}'.format(opt.input_scale_factor))
        if 'detector' in self.backbone_source:
            self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        img_index = index // self.im_div
        original_caption = self.captions[index] # Renamed for clarity
        original_caption_tokens = self.tokenizer.basic_tokenizer.tokenize(original_caption)

        original_target = process_caption(self.tokenizer, original_caption_tokens, self.train)

        image_id = self.images[img_index]
        generated_caption = self.generated_caps_list[image_id]
        generated_caption_tokens = self.tokenizer.basic_tokenizer.tokenize(generated_caption)
        generated_target = process_caption(self.tokenizer, generated_caption_tokens, self.train)

        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        try:
            im_in = np.array(imread(image_path))
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}. Using dummy image.")
            im_in = np.zeros((self.base_target_size, self.base_target_size, 3), dtype=np.uint8) # Fallback to dummy

        processed_image = self._process_image(im_in)
        image = torch.Tensor(processed_image)
        image = image.permute(2, 0, 1)

        return image, original_target, generated_target, index, img_index


    def __len__(self):
        return self.length

    def _process_image(self, im_in):
        """
            Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation.
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        if 'detector' in self.backbone_source:
            im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)

        if self.train:
            target_size = self.base_target_size * self.train_scale_rate
        else:
            target_size = self.base_target_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        # Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
        else:
            processed_im = self._imagenet_norm(processed_im)

        return processed_im

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    def _detector_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im


class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """
    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))

        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div
        original_caption = self.captions[index] # Renamed for clarity
        original_caption_tokens = self.tokenizer.basic_tokenizer.tokenize(original_caption)

        # Convert caption (string) to word ids (with Size Augmentation at training time)
        original_target = process_caption(self.tokenizer, original_caption_tokens, self.train)
        image = self.images[img_index]
        image = torch.Tensor(image)

        generated_target = torch.tensor([self.tokenizer.vocab['[CLS]'], self.tokenizer.vocab['[SEP]']], dtype=torch.long)

        return image, original_target, generated_target, index, img_index


    def __len__(self):
        return self.length


def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()
        # print(prob)
        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.tensor(target, dtype=torch.long) # ensure target is long tensor
    return target


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, original_caption, generated_caption) tuples.
    Args:
        data: list of (image, original_caption, generated_caption, id, img_id) tuple.
            - image: torch tensor of shape (3, 256, 256) or (N, D).
            - original_caption: torch tensor of shape (?); variable length.
            - generated_caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, ...).
        original_targets: torch tensor of shape (batch_size, padded_length).
        original_lengths: list; valid length for each padded original caption.
        generated_targets: torch tensor of shape (batch_size, padded_length).
        generated_lengths: list; valid length for each padded generated caption.
        ids: list; original data point IDs.
    """

    images, original_captions_list, generated_captions_list, ids, img_ids = zip(*data)

    # Process original captions (Common to both datasets)
    original_lengths = [len(cap) for cap in original_captions_list]
    original_targets = torch.zeros(len(original_captions_list), max(original_lengths)).long()
    for i, cap in enumerate(original_captions_list):
        end = original_lengths[i]
        original_targets[i, :end] = cap[:end]

    generated_lengths = [len(cap) for cap in generated_captions_list]
    max_gen_len = 0
    if len(generated_lengths) > 0:
        max_gen_len = max(generated_lengths)
    generated_targets = torch.zeros(len(generated_captions_list), max_gen_len).long()
    for i, cap in enumerate(generated_captions_list):
        end = generated_lengths[i]
        generated_targets[i, :end] = cap[:end]

    if len(images[0].shape) == 2: 
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.tensor(img_lengths, dtype=torch.long) # ensure img_lengths is long tensor

        return all_images, img_lengths, original_targets, original_lengths, generated_targets, generated_lengths, ids
    else: 
        images = torch.stack(images, 0)
        return images, original_targets, original_lengths, generated_targets, generated_lengths, ids


def get_loader(data_path, data_name, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.precomp_enc_type == 'basic':
        dset = PrecompRegionDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    else:
        dset = RawImageDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader