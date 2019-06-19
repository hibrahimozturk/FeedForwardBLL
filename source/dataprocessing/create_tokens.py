import nltk
import codecs
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string 
from nltk.corpus import stopwords
import numpy as np
import json
import random
from google.cloud import translate

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def preprocess_text(input_str):
    input_str = input_str.lower()
    input_str = re.sub(r'\d+', '', input_str)
    input_str = re.sub(r'–', '', input_str)
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
                        
                        if len(lemmas) >= 100000: #test on small portion
                            return lemmas
                    
                if '<content' in line: # start of content
                    is_content = True 
                
            if '<article lang="en"' in line: # only english articles
                is_english = True
    return lemmas

def translate_en_it(en_words, split_set = 'train'):
    train_pairs = {'en_words':set(), 'it_words':set(), 'translation_pairs':[]}
#     translator= Translator(from_lang="english",to_lang="italian")
    translate_client = translate.Client()
    num_of_chars = 0
    for i,en_word in enumerate(en_words):
        translation = translate_client.translate(en_word, target_language='it')
        num_of_chars += len(en_word)
        
        if translation['translatedText'].lower() == en_word: # not translated
            continue        
        if len(translation['translatedText'].split()) == 1:
            train_pairs['it_words'].add(translation['translatedText'].lower().strip())
            train_pairs['translation_pairs'].append([en_word, translation['translatedText'].lower().strip() ])
            train_pairs['en_words'].add(en_word)
                
        if i%100 == 0: # backup translations, there is a limit of translations in google translate
            with open(split_set + '_'+'wikicomp_pairs.json', 'w') as fp:
                save_pairs = {'en_words':list(train_pairs['en_words']), 'it_words':list(train_pairs['it_words']), 'translation_pairs':train_pairs['translation_pairs']}
                fp.write(json.dumps(save_pairs))   
                print("{} translations completed in {}".format(i, split_set))
                print("{} num of chars translated in {}".format(num_of_chars, split_set))
                print()

        
    return train_pairs

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

def create_annotations(set_pair, split_set='train'):
    
    annotations = {'en_words':list(set_pair['en_words']), 'it_words':list(set_pair['it_words']), 'input_outputs':[]}
    italian_words = list(set_pair['it_words'])
    for pair in set_pair['translation_pairs']:
        positive_sample = {'english_word': pair[0], 'italian_word':pair[1], 'output':1}
        wrong_translation = random.choice(italian_words)
        while wrong_translation == pair[1]: # true translation could be selected
            wrong_translation = random.choice(italian_words)
        negative_sample = {'english_word': pair[0], 'italian_word':wrong_translation, 'output':0}
        
        annotations['input_outputs'].append(positive_sample)
        annotations['input_outputs'].append(negative_sample)

    return annotations

if __name__ == '__main__':
    
#     nltk.download()
    en_words = get_words()
    print("Number of english words: {}".format(len(en_words)))

    train_set, val_set, test_set = split_dataset(list(en_words))

    
    train_pairs = translate_en_it(train_set, split_set='train')
    print("Number of train pairs: {}".format(len(train_pairs['translation_pairs'])))

    val_pairs = translate_en_it(val_set, split_set='val')
    print("Number of val pairs: {}".format(len(val_pairs['translation_pairs'])))
     
    test_pairs = translate_en_it(test_set, split_set='test')
    print("Number of test pairs: {}".format(len(test_pairs['translation_pairs'])))
 
    train_annotations = create_annotations(train_pairs)
    val_annotations = create_annotations(val_pairs)
    test_annotations = create_annotations(test_pairs)
    

    with open('wikicomp_train_set.json', 'w') as fp:
        fp.write(json.dumps(train_annotations))
         
    with open('wikicomp_val_set.json', 'w') as fp:
        fp.write(json.dumps(val_annotations))
         
    with open('wikicomp_test_set.json', 'w') as fp:
        fp.write(json.dumps(test_annotations))
            
    print("finish")