import time
import pickle
import argparse
import numpy as np

from functools import partial
from scipy import stats

import torch
from torch.utils.tensorboard import SummaryWriter

from args import *
from utils import get_word_vec, load_gur
from data import Dictionary
from model import CpuSisg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    lang = args.lang
    name = f"{lang}_cpu"
    training_path = f"../data/train_pair_TwoBil{lang}.txt"
    
    if lang == "en":
        max_iter = MAX_TRAIN_EN_DATA_SIZE
    elif lang == "de":
        max_iter = MAX_TRAIN_DE_DATA_SIZE
    
    st = time.time()
    file_path = f"../src/{lang}_dictionary.pkl"
    with open(file_path,"rb") as file:
        dictionaries = pickle.load(file)
    print(f"finish making {lang} dictionary {time.time()-st:.2f}")
    
    data_file = open(training_path, "r")
    
    dimension = 300
    vocab_size = len(dictionaries.ids)
    subwords_size = MAX_SUBWORD_VOCAB_SIZE
    if (vocab_size + len(dictionaries.subword2id.keys())) <= MAX_SUBWORD_VOCAB_SIZE:
        subwords_size = vocab_size + len(dictionaries.subword2id.keys())
    
    print(f"data size: {dictionaries.data_size}  || train size: {dictionaries.train_size}  ||   vocab size: {vocab_size} ")
    
    model = CpuSisg(subwords_size, vocab_size, dimension)
    
    writer = SummaryWriter(log_dir=f"./runs/{name}")
    
    st = time.time()
    print("start learning")
    
    label = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    lr = 0.05
    w1, w2, base_corr = load_gur(350)
    
    for iter in range(1,max_iter+1):
        # breakpoint()
        line = data_file.readline()
        if not line:
            data_file.close()
            break
        
        target, context = map(int, line.strip().split())
        train_lr = lr*(1 - ((iter-1)/max_iter))
        model.update(dictionaries, target, context, 5, label, train_lr)
        
        # if iter%100000 == 0:
        if iter%10 == 0:
            train_loss = model.loss()
            print(f"iter:{iter:<10} lr:{train_lr:<10.5f} loss:{train_loss:<10.5f} time:{(time.time()-st)/3600:>6.4f} Hour")
            writer.add_scalars('loss', {'train_loss':train_loss}, iter)
            
        # if iter%3901818 == 0:
        if iter%10 == 0:
            corr_list_false = []
            corr_list_true = []
            b_corr_list = []
            for word1, word2, bc in zip(w1, w2, base_corr):
                word1, word2 = word1.lower(), word2.lower()
                word1_emb_false = get_word_vec(word1, dictionaries, model.in_emb_layer, vocab_size, False)
                word2_emb_false = get_word_vec(word2, dictionaries, model.in_emb_layer, vocab_size, False)
                word1_emb_true = get_word_vec(word1, dictionaries, model.in_emb_layer, vocab_size, True)
                word2_emb_true = get_word_vec(word2, dictionaries, model.in_emb_layer, vocab_size, True)
                corr_list_false.append(torch.sum(word1_emb_false* word2_emb_false).item())
                corr_list_true.append(torch.sum(word1_emb_true* word2_emb_true).item())
                b_corr_list.append(bc)
            b_corr_list = np.array(list(map(float,b_corr_list)))
            l2_norm = np.sqrt(np.sum(np.power(b_corr_list,2)))
            b_corr_list /= (l2_norm + 1e-7)
            corr_false = stats.spearmanr(corr_list_false, b_corr_list)[0]*100
            corr_true = stats.spearmanr(corr_list_true, b_corr_list)[0]*100
            writer.add_scalars('corr', {'sisg-':corr_false}, iter)
            writer.add_scalars('corr', {'sisg':corr_true}, iter)
            print(f"{corr_false:<10.5f}, {corr_true:<10.5f}")

        if iter%390181830 == 0:
            torch.save({
                'iter':iter,
                'word_embedding': model.in_emb_layer,
            }, f"./save_model/sisg_{name}.pth")
            
    torch.save({
                'iter':iter,
                'model_embedding': model.in_emb_layer,
            }, f"./save_model/sisg_{name}_final.pth")
    
    writer.close()
    data_file.close()