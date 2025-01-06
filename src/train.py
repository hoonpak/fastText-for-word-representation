import time
import torch
import pickle
import argparse
import numpy as np

from scipy import stats
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from args import *
from data import Dictionary, CustumDataset, custom_collate_fn
from utils import get_word_vec, load_gur
from model import SISG
from functools import partial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    lang = args.lang
    name = f"{lang}_all_non_shuffle_v2"
    training_path = f"../data/train_pair_TwoBil{lang}.txt"
    
    batch_size = 1024 #4096
    if lang == "en":
        max_iter = MAX_TRAIN_EN_DATA_SIZE//batch_size + 1
    elif lang == "de":
        max_iter = MAX_TRAIN_DE_DATA_SIZE//batch_size + 1
    
    st = time.time()
    file_path = f"../src/{lang}_dictionary.pkl"
    with open(file_path,"rb") as file:
        dictionaries = pickle.load(file)
    print(f"finish making {lang} dictionary {time.time()-st:.2f}")
    
    # dataset = CustumDataset(training_path, dictionaries)
    dataset = CustumDataset(training_path, dictionaries.word2subword, dictionaries.max_subword_length, len(dictionaries.ids))
    custom_collate_fn = partial(custom_collate_fn, ids = dictionaries.ids, batch_size = batch_size, negative_probability = dictionaries.negative_probability)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=2, collate_fn=custom_collate_fn)
    
    dimension = 300
    vocab_size = len(dictionaries.ids)
    subwords_size = MAX_SUBWORD_VOCAB_SIZE + 1
    if (vocab_size + len(dictionaries.subword2id.keys())) <= MAX_SUBWORD_VOCAB_SIZE:
        subwords_size = vocab_size + len(dictionaries.subword2id.keys())
    
    print(f"data size: {dictionaries.data_size}  || train size: {dictionaries.train_size}  ||   vocab size: {vocab_size} ")
    
    model = SISG(vocab_size, subwords_size, dimension)
    with torch.no_grad():
        model.word_in_emb.weight[MAX_SUBWORD_VOCAB_SIZE] = torch.zeros(300).to("cuda:0")
    loss_function = torch.nn.BCELoss().to("cuda:1")
    
    optim = torch.optim.SGD(model.parameters(), lr=0.05*batch_size)
    lambda_func = lambda iter: max(0.00005, (1 - (iter / max_iter)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim,lr_lambda=lambda_func)
    
    writer = SummaryWriter(log_dir=f"./runs/{name}")
    st = time.time()

    print("start learning")
    train_loss = 0
    w1, w2, base_corr = load_gur(350)
    x_axis = 0
    for iter, cache in enumerate(dataloader):
        # breakpoint()
        iter += 1
        targets, subwords, length, samples, labels = cache # target, contexts = (batchsize) // subwords = (batchsize, subword_max) // negs = (batchsize, num samples) // labels = (batchsize, 6)
        targets = targets.to("cuda:0")
        subwords = subwords.to("cuda:0")
        length = length.to("cuda:0")
        samples = samples.to("cuda:1")
        labels = labels.to("cuda:1")
        
        optim.zero_grad()
        predict = model.forward(targets, subwords, length, samples)
        loss = loss_function.forward(predict, labels)
        loss.backward()
        optim.step()
        lr_scheduler.step()
        
        train_loss += loss.detach().cpu().item()
        
        if iter%500 == 0:
            train_loss /= 500
            print(f"iter:{iter:<10} lr:{optim.param_groups[0]['lr']:<10.4f} loss:{train_loss:<10.5f} time:{(time.time()-st)/3600:>6.4f} Hour")
            writer.add_scalars('loss', {'train_loss':train_loss}, iter)
            train_loss = 0
            
        if iter%3810 == 0:
            x_axis += 0.1
            model.eval()
            with torch.no_grad():
                corr_list_false = []
                corr_list_true = []
                b_corr_list = []
                for word1, word2, bc in zip(w1, w2, base_corr):
                    word1, word2 = word1.lower(), word2.lower()
                    word1_emb_false = get_word_vec(word1, dictionaries, model.word_in_emb.weight, vocab_size, False)
                    word2_emb_false = get_word_vec(word2, dictionaries, model.word_in_emb.weight, vocab_size, False)
                    word1_emb_true = get_word_vec(word1, dictionaries, model.word_in_emb.weight, vocab_size, True)
                    word2_emb_true = get_word_vec(word2, dictionaries, model.word_in_emb.weight, vocab_size, True)
                    corr_list_false.append(torch.sum(word1_emb_false * word2_emb_false).item())
                    corr_list_true.append(torch.sum(word1_emb_true * word2_emb_true).item())
                    b_corr_list.append(bc)
            b_corr_list = np.array(list(map(float,b_corr_list)))
            l2_norm = np.sqrt(np.sum(np.power(b_corr_list,2)))
            b_corr_list /= (l2_norm + 1e-7)
            corr_false = round(stats.spearmanr(corr_list_false, b_corr_list)[0]*100)
            corr_true = round(stats.spearmanr(corr_list_true, b_corr_list)[0]*100)
            writer.add_scalars('corr', {'sisg-':corr_false}, iter)
            writer.add_scalars('corr', {'sisg':corr_true}, iter)
            print(f"process:{x_axis:.1f}%     sisg-:{corr_false:<6} sisg:{corr_true:<6}")
            model.train()
            
        if iter%500000 == 0:
            torch.save({
                'iter':iter,
                'model_state_dict': model.word_in_emb.weight,
            }, f"./save_model/sisg_{name}.pth")
            
    torch.save({
        'iter':iter,
        'model_state_dict': model.word_in_emb.weight,
    }, f"./save_model/sisg_{name}_final.pth")
    
    writer.close()
