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

# en_fasttext = fasttext.load_model('fasttext/en/wiki-news-300d-1M.vec')



def vectors_cache():
    wordvectors = {}
    with codecs.open('dataprocessing/train_wikicomp_pairs.json', 'r', "ISO-8859-1") as fp:
        en_it_pairs = json.load(fp)
    
    en_model = gensim.models.KeyedVectors.load_word2vec_format('word2vec/en/GoogleNews-vectors-negative300.bin', binary=True)
    
    it_model = gensim.models.KeyedVectors.load('word2vec/it/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m')
    
    for it_word in en_it_pairs['it_words']:
        if it_word in it_model.wv.vocab:
            wordvectors[it_word] = it_model[it_word]
            
    for en_word in en_it_pairs['en_words']:
        if en_word in en_model.wv.vocab:
            wordvectors[en_word] = en_model[en_word]
            
    with open('train_vectors.pickle', 'wb') as handle:
        pickle.dump(wordvectors, handle, protocol=pickle.HIGHEST_PROTOCOL)


# -*- coding: utf-8 -*-

class BLLDataset(Dataset):

    def __init__(self, pairs, word_vectors):
        self.pairs      = pairs
        self.word_vectors = word_vectors
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
#         print(self.pairs[idx]['english_word'])
        src_word2vec = self.word_vectors[self.pairs[idx]['english_word']]
#         print(src_word2vec)
        target_word2vec = self.word_vectors[self.pairs[idx]['italian_word']]
#         print(target_word2vec)
#         print(self.pairs[idx]['italian_word'])
        
        sample = {'src_word': self.pairs[idx]['english_word'] , 'target_word':self.pairs[idx]['italian_word'], 'src_word2vec': src_word2vec, 'target_word2vec': target_word2vec , 'output': self.pairs[idx]['output']}

        return sample
    

    
if __name__ == '__main__':
#     vectors_cache()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'

    with open('train_vectors.pickle', 'rb') as handle:
        word_vectors = pickle.load(handle)
        
    with codecs.open('dataprocessing/wikicomp_train_set.json', 'r', "ISO-8859-1") as fp:
        en_it_pairs = json.load(fp) 
         
    train_data = BLLDataset(en_it_pairs['input_outputs'], word_vectors)

    # show a batch
    batch_size = 4
    for i in range(batch_size):
        sample = train_data[i]
        print(i, sample['src_word'], sample['src_word2vec'].shape, sample['target_word'], sample['target_word2vec'].shape)

#     dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
# 
#     for i, batch in enumerate(dataloader):
#         print(i, sample['src_word'], sample['src_word2vec'].shape, sample['target_word'], sample['target_word2vec'].shape)
    
    
print("finish")  