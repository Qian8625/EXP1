import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.temperature = temperature

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s, temperature=self.temperature)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


def get_sim(images, captions, temperature=1.0):
    """
    Compute similarity matrix between images and captions.
    
    Args:
        images: Image embeddings [batch_size, embed_dim]
        captions: Caption embeddings [batch_size, embed_dim]
        temperature: Temperature scaling factor. Lower values amplify differences.
    
    Returns:
        Similarity matrix [batch_size, batch_size]
    """
    # Compute cosine similarity (dot product for normalized embeddings)
    similarities = images.mm(captions.t())
    # Apply temperature scaling to amplify similarity differences
    # Lower temperature makes the model more confident in its predictions
    # similarities = similarities / temperature
    return similarities

class DiversityRegularization(nn.Module):
    """
    Compute diversity regularization
    """
    def __init__(self, smry_k, batch_size):
        super(DiversityRegularization, self).__init__()
        self.smry_k = smry_k
        self.batch_size = batch_size
        self.I = torch.eye(smry_k).unsqueeze(0).repeat(batch_size, 1, 1).cuda() #(bs, k, k)

    def forward(self, smry_mat):
        bs = smry_mat.size(0)
        # smry_mat = smry_mat.permute(0, 2, 1)
        smry_mat = F.normalize(smry_mat, dim=1)   #(bs, num_r, k)
        diversity_loss = torch.matmul(smry_mat.transpose(1, 2), smry_mat)   #(bs, k, k)
        if bs != self.batch_size:
            I = torch.eye(self.smry_k).unsqueeze(0).repeat(bs, 1, 1).cuda()
        else:
            I = self.I
        diversity_loss = diversity_loss - I
        diversity_loss = (diversity_loss ** 2).sum()
        return diversity_loss