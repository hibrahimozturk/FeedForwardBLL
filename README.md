## Download Links
* Wikipedia comparable corporas https://linguatools.org/tools/corpora/wikipedia-comparable-corpora/
 
* Wikipedia corpora english-italian download link: https://www.dropbox.com/s/jjkfn7v19tnxnww/wikicomp-2014_enit.xml.bz2?dl=0

* Python gensim word2vec italian language model http://hlt.isti.cnr.it/wordembeddings/skipgram_wiki_window10_size300_neg-samples10.tar.gz

* Python gensim word2vec english language model https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

* Python fasttext english language model https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

* Python fasttext italian language model  https://www.dropbox.com/s/orqfu6mb9cj9ewr/it.tar.gz?dl=0

## Python packages

* nltk
* google-cloud-translate
* numpy
* gensim
* fasttext
* pytorch

## Dataset

[English word, italian word]

* Total words:    15278
  Training set:   11590
  Validation set: 2939
  Test set:       749

One positive and one negative translation pair are selected from words in sets.

* After filtering words not included in word2vec
   Training pairs: 20977

## Todo

* [x] Tokens will be extracted from corpora
* [x] Pairs will be generated by google translate
* [x] Training, validation and test sets will be splitted
* [x] python word2vec for it and en will be installed 
* [x] pytorch special dataloader will be implemented  
* [ ] models will be trained on pytorch  
* [ ] evaluation will be implemented  
