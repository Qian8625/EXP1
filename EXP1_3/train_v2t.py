import os
import time
import numpy as np
import torch
from transformers import BertTokenizer

from lib.datasets import image_caption_v2t_clip

from lib.vse_v2t_clip_vit import VSEModel
from lib.evaluation_clip import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sim

import logging
import tensorboard_logger as tb_logger
import random
import arguments
import shutil

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()

def main():
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    train_loader, val_loader = image_caption_v2t_clip.get_loaders( 
        opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)

    model = VSEModel(opt)

    lr_schedules = [opt.lr_update, ]

    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            if not model.is_data_parallel:
                model.make_data_parallel()
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    if not model.is_data_parallel:
        model.make_data_parallel()

    best_rsum = 0
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        if epoch >= opt.vse_mean_warmup_epochs:
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        train(opt, train_loader, model, epoch, val_loader)

        rsum = validate(opt, val_loader, model)

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint.pth', prefix=opt.model_name + '/', epoch=epoch, total_epochs=25)


def train(opt, train_loader, model, epoch, val_loader):
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    logger.info('image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
    logger.info('txt encoder trainable parameters: {}'.format(count_params(model.txt_enc)))

    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train_start()

        data_time.update(time.time() - end)

        model.logger = train_logger

        if opt.precomp_enc_type == 'basic':
            images, img_lengths, orig_caps, orig_lengths, gen_caps, gen_lengths, _ = train_data
            model.train_emb(images, orig_caps, orig_lengths, 
                            generated_captions=gen_caps, generated_lengths=gen_lengths,
                            image_lengths=img_lengths)
        else:
            images, orig_caps, orig_lengths, gen_caps, gen_lengths, _ = train_data
            if epoch == opt.embedding_warmup_epochs:
                warmup_alpha = float(i) / num_loader_iter
                model.train_emb(images,
                                orig_caps,
                                orig_lengths,
                                generated_captions=gen_caps,
                                generated_lengths=gen_lengths,
                                warmup_alpha=warmup_alpha)
            else:
                model.train_emb(images,
                                orig_caps,
                                orig_lengths,
                                generated_captions=gen_caps,
                                generated_lengths=gen_lengths)

        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:
            if opt.precomp_enc_type == 'backbone' and epoch == opt.embedding_warmup_epochs:
                logging.info('Current epoch-{}, the first epoch for training backbone, warmup alpha {}'.format(epoch,
                                                                                                               warmup_alpha))
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)



def validate(opt, val_loader, model):
    logger = logging.getLogger(__name__)
    model.val_start()
    
    result_file = os.path.join(opt.logger_name, 'validation_results.txt')
    
    with torch.no_grad():
        img_embs, cap_embs = encode_data(
            model, val_loader, opt.log_step, logging.info, backbone=opt.precomp_enc_type == 'backbone')

    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = compute_sim(img_embs, cap_embs)
    end = time.time()
    logger.info("calculate similarity time: {}".format(end - start))

    npts = img_embs.shape[0]
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    i2t_result = "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr)
    logger.info(i2t_result)
    
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    t2i_result = "Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr)
    logger.info(t2i_result)
    
    with open(result_file, 'a') as f:
        f.write(f"Epoch {model.Eiters} Validation Results:\n")
        f.write(i2t_result + "\n")
        f.write(t2i_result + "\n")
        f.write("-" * 50 + "\n")
    
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    del img_embs, cap_embs, sims
        
    logger.info('Current rsum is {}'.format(currscore))

    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix='', epoch=None, total_epochs=25):
    logger = logging.getLogger(__name__)
    tries = 15
    
    early_phase = epoch < 10 if epoch is not None else False
    
    while tries:
        try:
            if early_phase:
                if is_best:
                    torch.save(state, prefix + 'model_best.pth')
                    message = "--------save best model at epoch %d---------\n" % (state["epoch"])
                    print(message, flush=True)
                    log_file = os.path.join(prefix, "performance.log")
                    logging_func(log_file, message)
            else:
                epoch_filename = f'checkpoint_epoch{epoch}.pth'
                torch.save(state, os.path.join(prefix, epoch_filename))
                
                if is_best:
                    shutil.copyfile(os.path.join(prefix, epoch_filename), 
                                  os.path.join(prefix, 'model_best.pth'))
                    message = "--------save best model at epoch %d---------\n" % (state["epoch"])
                    print(message, flush=True)
                    log_file = os.path.join(prefix, "performance.log")
                    logging_func(log_file, message)
                    
        except IOError as e:
            error = e
            tries -= 1
            logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
            if not tries:
                raise error
        else:
            break

def adjust_learning_rate(opt, optimizer, epoch):
    logger = logging.getLogger(__name__)
    lr_schedules = opt.lr_update
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.3
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
