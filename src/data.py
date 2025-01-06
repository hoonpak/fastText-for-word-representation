import re
import numpy as np
from collections import Counter
from copy import deepcopy

from args import *

import torch 
from torch.utils.data import IterableDataset, get_worker_info
from torch import LongTensor
from torch import IntTensor
from torch import Tensor

class Dictionary:
    def __init__(self, file_path, language, minimum, threshold, ngram):
        self.file_path = file_path
        self.language = language
        self.minimum = minimum
        self.ngram = ngram #list
        # self.vocab_num = vocab_num
        
        self.id2word = dict()
        self.word2id = dict()
        self.id2subword = dict()
        self.subword2id = dict()
        self.frequency = None
        self.data_size = None
        self.train_size = None
        self.max_subword_length = None
        
        self.word2subword = dict()
        
        self.dictionary()
        print("finish make dictionary")
        self.subword_dictionary()
        print("finish make subword dictionary")
        
        self.negative_probability = self.calculate_negative_probability(self.frequency)
        self.subsampling_probability = self.calculate_subsampling_probability(self.frequency, threshold)
        
        self.ids = list(self.id2word.keys())
        
    def clean_sentence(self, sentence, lang):
        if lang == "en":
            sentence = re.sub(r"[^A-Za-z()\.\,\!\?\"\']", " ", sentence)
            sentence = re.sub(r"\(", " ( ", sentence)
            sentence = re.sub(r"\)", " ) ", sentence)
            sentence = re.sub(r"\.", " . ", sentence)
            sentence = re.sub(r"\,", " , ", sentence)
            sentence = re.sub(r"\!", " ! ", sentence)
            sentence = re.sub(r"\?", " ? ", sentence)
            sentence = re.sub(r"\"", " \" ", sentence)
            sentence = re.sub(r"\'", " \' ", sentence)
            sentence = re.sub(r"\s{2,}", " ", sentence)
        if lang == "de":
            sentence = re.sub(r"[^A-Za-zÄäÖöÜüß()\.\,\!\?\"\']", " ", sentence)
            sentence = re.sub(r"\(", " ( ", sentence)
            sentence = re.sub(r"\)", " ) ", sentence)
            sentence = re.sub(r"\.", " . ", sentence)
            sentence = re.sub(r"\,", " , ", sentence)
            sentence = re.sub(r"\!", " ! ", sentence)
            sentence = re.sub(r"\?", " ? ", sentence)
            sentence = re.sub(r"\"", " \" ", sentence)
            sentence = re.sub(r"\'", " \' ", sentence)
            sentence = re.sub(r"\s{2,}", " ", sentence)
        return sentence.lower()
    
    def dictionary(self):
        counter = Counter()
        with open(self.file_path, "rb") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.decode(errors="replace")
                # line = self.clean_sentence(line, self.language)
                tokens = [token.strip() for token in line.split()]
                counter.update(tokens)
                
        id = 0
        frequency = list()
        for k,v in counter.items():
        # for k,v in counter.most_common(self.vocab_num):
            if len(k) >= 30:
                continue
            if v >= self.minimum:
                self.id2word[id] = k
                self.word2id[k] = id
                frequency += [v]
                id += 1
                
        self.frequency = np.array(frequency)
        self.data_size = counter.total()
        self.train_size = np.sum(frequency)
    
    def subword_dictionary(self):
        self.subword2id["0"] = 0
        self.id2subword[0] = "0"
        index = 1
        for word in self.id2word.values():
            wholeword = "<"+word+">"
            subword_list = []
            for n in self.ngram:
                if len(wholeword) <= n:
                    break
                else:
                    num_subword = len(wholeword) - n + 1
                    for si in range(num_subword):
                        subword = wholeword[si:si+n]
                        subword_list.append(subword)
                        if subword not in self.subword2id.keys():
                            self.subword2id[subword] = index
                            self.id2subword[index] = subword
                            index += 1
            self.word2subword[self.word2id[word]] = [self.subword2id[sub] for sub in subword_list]
        self.max_subword_length = max([len(subwords) for subwords in self.word2subword.values()])
        
    def calculate_negative_probability(self, frequency):
        pow_fre = np.float_power(frequency, 1/2) # 3/4 in word2vec
        Z = pow_fre.sum()
        return pow_fre / Z

    def calculate_subsampling_probability(self, frequency, threshold):
        f = frequency/self.train_size
        sqrt = np.sqrt(threshold/f)
        return 1 - sqrt
    
class CustumDataset(IterableDataset):
    def __init__(self, file_path, word2subword, max_subword_length, len_ids):
        self.file_path = file_path
        # self.dictionaries = dictionaries
        self.word2subword = word2subword
        self.max_subword_length = max_subword_length
        self.vocab_num = len_ids

    def _reset(self):
        self.file = open(self.file_path, 'r')
    
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self._reset()
            return self
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            self._reset()
            self.file.seek(0, 2)
            file_size = self.file.tell()
            chunk_size = file_size // num_workers
            start_offset = worker_id * chunk_size
            end_offset = None if worker_id == num_workers - 1 else (worker_id + 1) * chunk_size

            self.file.seek(start_offset)
            if start_offset != 0:
                self.file.readline()

            self.end_offset = end_offset
            return self

    def __next__(self): 
        if self.end_offset is not None and self.file.tell() >= self.end_offset:
            self.file.close()
            raise StopIteration

        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration

        target, context = map(int, line.strip().split())
        sub = list(map(lambda x: (x+self.vocab_num)%MAX_SUBWORD_VOCAB_SIZE, self.word2subword[target]))
        if len(sub) == 0:
            sub_length = 1
        else:
            sub_length = len(sub)
        subwords = sub + [MAX_SUBWORD_VOCAB_SIZE] * (self.max_subword_length - len(self.word2subword[target]))
        labels = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        target %= MAX_SUBWORD_VOCAB_SIZE
        return [target, subwords, sub_length, context, labels]

def custom_collate_fn(batch, ids, batch_size, negative_probability):
    targets, subwords, sub_lengths, contexts, labels = zip(*batch)
    negs = np.random.choice(ids, (batch_size,5), replace=True, p=negative_probability)
    negs = LongTensor(negs)
    contexts = LongTensor(contexts).view(-1,1)
    samples = torch.cat((contexts, negs), dim=1)
    targets = LongTensor(targets)
    subwords = LongTensor(subwords)
    sub_lengths = Tensor(sub_lengths)
    labels = Tensor(labels)
    return targets, subwords, sub_lengths, samples, labels