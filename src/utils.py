import torch
import csv
import numpy as np

from tqdm import tqdm
from args import *

def load_rw():
    rw = []
    with open("../corr_test/rw.csv", "r", encoding='utf-8-sig') as file:
        rw_csv = csv.reader(file)
        for i in rw_csv:
            rw.append(i)
    w1, w2, corr = zip(*rw)
    return w1, w2, corr
    
def load_ws353_en():
    ws353 = []
    with open("../corr_test/wordsim353_agreed.txt", 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line[0] == '#':
                continue
            line = line.lower().split()
            ws353.append(line)
    _, w1, w2, corr = zip(*ws353)
    return w1, w2, corr

def load_gur(number):
    gur = []
    with open(f"../corr_test/Gur{number}_DE.csv", "r", encoding='utf-8-sig') as file:
        gur_csv = csv.reader(file)
        for i in gur_csv:
            gur.append(i)
    w1, w2, corr, _1, _2 = zip(*gur)
    return w1, w2, corr

def load_zg222():
    zg222 = []
    with open(f"../corr_test/ZG222_DE.csv", "r", encoding='utf-8-sig') as file:
        zg222_csv = csv.reader(file)
        for i in zg222_csv:
            zg222.append(i)
    w1, w2, corr, _1, _2 = zip(*zg222)
    return w1, w2, corr

def get_subword(word, ngram, dictionaries, test = True):
    wholeword = "<"+word+">"
    subword_list = []
    for n in ngram:
        if len(wholeword) <= n:
            break
        else:
            num_subword = len(wholeword) - n + 1
            for si in range(num_subword):
                subword = wholeword[si:si+n]
                if subword not in dictionaries.subword2id.keys():
                    continue
                subword_list.append(dictionaries.subword2id[subword])
    return subword_list

def make_embedding(best_model, dictionaries, device):
    vocab_num = len(dictionaries.ids)
    word_vec = torch.zeros((vocab_num, 300)).to(device=device)
    print("start calculate word embedding with vocab")
    with torch.no_grad():
        best_model.eval()
        for iii in dictionaries.ids:
            subword_id = list(map(lambda x: (x+vocab_num)%MAX_SUBWORD_VOCAB_SIZE, dictionaries.word2subword[iii]))
            subword_id = torch.LongTensor(subword_id).to(device=device)
            si = best_model.word_in_emb(subword_id).to(device=device)
            si = si.mean(dim=0).to(device=device)
            si += best_model.word_in_emb(torch.LongTensor([iii])).view(-1).to(device=device)
            l2norm = si.pow(2).sum().sqrt().to(device=device) #L2Norm
            word_vec[iii] = si/l2norm
    return word_vec

def get_word_vec(word, dictionaries, in_emb_layer, vocab_size, sisg, ngram = [3,4,5,6], device = "cuda:0"):
    with torch.no_grad():
        if word in dictionaries.word2id.keys():
            word_id = dictionaries.word2id[word]
            word_subword = list(map(lambda x:(x+vocab_size)%MAX_SUBWORD_VOCAB_SIZE, dictionaries.word2subword[word_id]))
            word_id %= MAX_SUBWORD_VOCAB_SIZE
            if len(word_subword) == 1:
                word_e = in_emb_layer[word_id]
                l2norm = word_e.pow(2).sum().sqrt()
                word_emb = word_e/(l2norm)
            else:
                word_subword = word_subword
                word_subword_emb = in_emb_layer[word_subword].mean(dim=0)
                word_emb = in_emb_layer[word_id]
                word_emb += word_subword_emb
                l2_norm = word_emb.pow(2).sum().sqrt()
                word_emb /= (l2_norm)
        else:
            if sisg :
                word_subword = get_subword(word, ngram, dictionaries)
                word_subword = list(map(lambda x: (x+vocab_size)%MAX_SUBWORD_VOCAB_SIZE, word_subword))
                word_emb = in_emb_layer[word_subword].mean(dim=0)
                l2_norm = word_emb.pow(2).sum().sqrt()
                word_emb = word_emb/(l2_norm + 1e-7)
            else :
                word_emb = torch.zeros(300)
    return word_emb.to(device)
