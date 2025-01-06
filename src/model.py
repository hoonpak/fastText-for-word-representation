import torch
from torch import nn

import numpy as np

from copy import deepcopy

from args import *

class SISG(nn.Module):
    def __init__(self, vocab_size, subwords_size, dimension):
        super(SISG, self).__init__()
        self.vocab_size = vocab_size
        self.subwords_size = subwords_size
        self.dimension = dimension
        self.word_in_emb = nn.Embedding(self.subwords_size, self.dimension, padding_idx=MAX_SUBWORD_VOCAB_SIZE).to("cuda:0") #padding for matrix operation
        self.word_out_emb = nn.Embedding(self.vocab_size, self.dimension).to("cuda:1")
        self.sigmoid = nn.Sigmoid().to("cuda:1")
        
        nn.init.uniform_(self.word_in_emb.weight, a=-1/300, b=1/300)
        nn.init.zeros_(self.word_out_emb.weight)
        
    # def forward(self, targets, subwords, subword_length, contexts, negs):
    def forward(self, targets, subwords, subword_length, samples):
        """ **SHAPE**
            subwords = (batchsize, subword_max)
            targets, contexts = (batchsize)
            negs = (batchsize, num samples)
            labels = (batchsize, 6)"""
        word_emb = self.word_in_emb(targets) #batch_size, dimension
        subword_emb = self.word_in_emb(subwords).sum(dim=1) #batch_size, subword, dimension => batch_size, dimension
        subword_emb = torch.div(subword_emb, subword_length.view(-1,1)) #average
        word_emb += subword_emb  #batch_size, dimension
        
        pos_neg_emb = self.word_out_emb(samples) #batch_size, negative+1, dimension
        
        score = torch.matmul(pos_neg_emb, word_emb.unsqueeze(2).to("cuda:1")).squeeze(2) #batch_size, negative+1
        predict = self.sigmoid(score) #batch_size, negative+1
        
        return predict

class CpuSisg:
    def __init__(self, subword_size, vocab_size, dimension):
        self.vocab_size = vocab_size
        self.in_emb_layer = (1/300)*(2*(torch.rand((subword_size, dimension),requires_grad=False)) - 1)
        self.out_emb_layer = torch.zeros((vocab_size, dimension),requires_grad=False)
        self.predict = None
        self.label = None
        
    def update(self, dictionary, target, context, num_sample, label, lr):
        subwords = list(map(lambda x: (x+self.vocab_size)%MAX_SUBWORD_VOCAB_SIZE, dictionary.word2subword[target]))
        target %= MAX_SUBWORD_VOCAB_SIZE
        
        # if len(subwords) == 0:
        #     target_emb = self.in_emb_layer[target]
        # else:
        #     target_emb = self.in_emb_layer[target] + self.in_emb_layer[subwords].mean(dim=0) #300
        subwords += [target]
        target_emb = self.in_emb_layer[subwords].mean(dim=0) #300
        neg_prob = deepcopy(dictionary.negative_probability)
        neg_prob[context] = 0.0
        denominator = sum(neg_prob)
        neg_prob = list(map(lambda x : x/denominator, neg_prob))
        neg_sam = np.random.choice(dictionary.ids, num_sample, replace=False, p=neg_prob)
        samples = [context] + list(neg_sam)
        samples_emb = self.out_emb_layer[samples] #6,300
        score = torch.matmul(target_emb, samples_emb.T) #6
        self.predict = torch.sigmoid(score) #6
        
        self.label = label
        d_socre = lr*(self.predict - label) #6
        d_sample_emb = torch.matmul(d_socre.view(-1,1), target_emb.view(1,-1)) #300*6 -> 6,300
        d_target_emb = torch.matmul(d_socre, samples_emb) #6*6,300 -> 300
        self.out_emb_layer[samples] -= d_sample_emb
        # self.in_emb_layer[target] -= d_target_emb
        # if len(subwords) != 0:
        #     self.in_emb_layer[subwords] -= d_target_emb/(len(subwords))
        self.in_emb_layer[subwords] -= d_target_emb/(len(subwords)+1)
        return None
    
    def loss(self):
        return -(torch.log(self.predict[0] + 1e-6) + torch.sum(torch.log(1 - self.predict[1:] + 1e-6)))