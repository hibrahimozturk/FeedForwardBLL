from gensim.models import KeyedVectors
import gensim
# import fasttext
import json
import codecs
import pickle

import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def vectors_cache():
    wordvectors = {}
    with codecs.open('../data/wikicomp_dataset/val/wikicomp_val_set.json', 'r', "ISO-8859-1") as fp:
        en_it_pairs = json.load(fp)
    
    en_model = gensim.models.KeyedVectors.load_word2vec_format('word2vec/en/GoogleNews-vectors-negative300.bin', binary=True)
    
    it_model = gensim.models.KeyedVectors.load('word2vec/it/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m')
    
    filtered_it_words = []
    for it_word in en_it_pairs['it_words']:
        if it_word in it_model.wv.vocab:
            wordvectors[it_word] = it_model[it_word]
            filtered_it_words.append(it_word)
        else:
            print(it_word)

    filtered_en_words = []
    for en_word in en_it_pairs['en_words']:
        if en_word in en_model.wv.vocab:
            wordvectors[en_word] = en_model[en_word]
            filtered_en_words.append(en_word)
        else:
            print(en_word)
            
    filtered_input_outpus = []
    
    for input_output in en_it_pairs['input_outputs']:
        if input_output['italian_word'] in filtered_it_words:
            if input_output['english_word'] in filtered_en_words:
                filtered_input_outpus.append(input_output)
    
    filtered_annonatations = {'en_words':filtered_en_words, 'it_words':filtered_it_words, 'input_outputs':filtered_input_outpus}
    with codecs.open('wikicomp_val_filtered_set.json', 'w', "ISO-8859-1") as fp:
        fp.write(json.dumps(filtered_annonatations, sort_keys=True, indent=4, ensure_ascii=False))
    
    with open('val_vectors.pickle', 'wb') as handle:
        pickle.dump(wordvectors, handle, protocol=pickle.HIGHEST_PROTOCOL)


# -*- coding: utf-8 -*-

class BLLDataset(Dataset):

    def __init__(self, pairs, word_vectors):
        self.pairs      = pairs
        self.word_vectors = word_vectors
        
        print("Number of pairs {}".format(len(pairs)))
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        src_word2vec = self.word_vectors[self.pairs[idx]['english_word']]
#         src_word2vec = np.interp(src_word2vec, (src_word2vec.min(), src_word2vec.max()), (-1, 1))
        target_word2vec = self.word_vectors[self.pairs[idx]['italian_word']]
#         target_word2vec = np.interp(target_word2vec, (target_word2vec.min(), target_word2vec.max()), (-1, 1))

        sample = {'src_word': self.pairs[idx]['english_word'] , 'target_word':self.pairs[idx]['italian_word'],
                  'src_word2vec': src_word2vec, 'target_word2vec': target_word2vec ,
                  'output': float(self.pairs[idx]['output'])}

        return sample
    

    
if __name__ == '__main__':
    vectors_cache()
    
#     os.environ["CUDA_VISIBLE_DEVICES"]= '0'
#  
#     with open('../data/wikicomp_dataset/train/train_vectors.pickle', 'rb') as handle:
#         word_vectors = pickle.load(handle)
#          
#     with codecs.open('../data/wikicomp_dataset/train/train_filtered_set.json', 'r', "ISO-8859-1") as fp:
#         en_it_pairs = json.load(fp) 
#           
#     train_data = BLLDataset(en_it_pairs['input_outputs'], word_vectors)
#  
#     # show a batch
#     batch_size = 4
#     for i in range(batch_size):
#         sample = train_data[i]
#         print(i, sample['src_word'], sample['src_word2vec'].shape, sample['target_word'], sample['target_word2vec'].shape)
# 
#     dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
#   
#     for i, batch in enumerate(dataloader):
#         print(i, batch['src_word'], batch['src_word2vec'], batch['target_word'], batch['target_word2vec'], batch['output'])
    
    
    print("finish")  