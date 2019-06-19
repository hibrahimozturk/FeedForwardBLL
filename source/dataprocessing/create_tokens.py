import nltk
import codecs
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string 
from nltk.corpus import stopwords
from googletrans import Translator
import numpy as np
import json

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def preprocess_text(input_str):
    input_str = input_str.lower()
    input_str = re.sub(r'\d+', '', input_str)
    input_str = input_str.translate(str.maketrans('', '', string.punctuation))
    input_str = input_str.strip()
    return input_str


def get_words():
    lemmatizer = WordNetLemmatizer() 
    stop_words = set(stopwords.words('english'))
    lemmas = set()
    is_content = False
    is_english = False
    with  codecs.open('../../data/wikicomp-2014_enit.xml', 'r', encoding='UTF-8') as f:
        for line in f:
            
            if '</article>' in line: #end of article
                is_english = False
            
            if is_english:
                if '</content' in line: # end of content
                    is_content = False
                    
                if is_content:
                    content = cleanhtml(line)
                    content = preprocess_text(content)
                    words = word_tokenize(content)
                    words = [i for i in words if not i in stop_words]
                    for word in words:
                        stem = lemmatizer.lemmatize(word)
#                         print("{} : {}".format(word, stem))
                        if stem != "":
                            lemmas.add(stem)
                        
#                         if len(lemmas) > 100: #test on small portion
#                             return lemmas
                    
                if '<content' in line: # start of content
                    is_content = True 
                
            if '<article lang="en"' in line: # only english articles
                is_english = True
    return lemmas

def translate_en_it(en_words):
    it_words = set()
    en_it_pairs = []
    translator = Translator()
    for i,en_word in enumerate(en_words):
        translated = translator.translate(en_word, dest='it', src='en')
        if translated.text == en_word: # not translated
            continue
        
        if len(translated.text.split()) > 1: # translated should be word not phrase
            for possible_translation in  translated.extra_data['possible-translations'][0][2]:
                if len(possible_translation[0].split()) == 1:
                    it_words.add(possible_translation[0])
                    en_it_pairs.append([en_word, possible_translation[0].lower() ])
                    break
        elif len(translated.text.split()) == 1:
            it_words.add(translated.text)
            en_it_pairs.append([en_word, translated.text.lower() ])
            
        if i%100 == 0: # backup translations, there is a limit of translations in google translate
            with open('wikicomp_pairs.json', 'w') as fp:
                fp.write(json.dumps(train_set))   
        
    return it_words, en_it_pairs

def split_dataset(word_list, val_ratio=0.2, test_ratio=0.3):
    
    dataset_size = len(word_list)
    indexes = np.arange(dataset_size)
    np.random.shuffle(indexes)
    
    test_size = int(test_ratio*dataset_size)
    train_size = dataset_size - test_size
    test_indexes = indexes[:test_size]
    
    # validation set size is calculated from training set size
    trainval_indexes = indexes[test_size:]
    val_size = int(val_ratio*train_size)
    val_indexes = trainval_indexes[:val_size]
    train_indexes = trainval_indexes[val_size:]
    
    train_set = [word_list[i] for i in train_indexes]
    val_set = [word_list[i] for i in val_indexes]
    test_set = [word_list[i] for i in test_indexes]
    
    return train_set, val_set, test_set


if __name__ == '__main__':
    
#     nltk.download()
    en_words = get_words()
    print("Number of english words: {}".format(len(en_words)))

    it_words, en_it_pairs = translate_en_it(en_words)
    print("Number of pairs: {}".format(len(en_it_pairs)))

    train_set, val_set, test_set = split_dataset(en_it_pairs)
    
    print("Training set size: {}".format(len(train_set)))
    print("Validation set size: {}".format(len(val_set)))
    print("Test set size: {}".format(len(test_set)))

    
    with open('wikicomp_train_set.json', 'w') as fp:
        fp.write(json.dumps(train_set))
        
    with open('wikicomp_val_set.json', 'w') as fp:
        fp.write(json.dumps(val_set))
        
    with open('wikicomp_test_set.json', 'w') as fp:
        fp.write(json.dumps(test_set))
            
    print("finish")