from gensim.models import KeyedVectors
import gensim
# import fasttext
import json
import codecs
import pickle
# en_fasttext = fasttext.load_model('fasttext/en/wiki-news-300d-1M.vec')

# with open('filename.pickle', 'rb') as handle:
#     unserialized_data = pickle.load(handle)

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

print("finish")  