import nltk
import codecs
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string 
from nltk.corpus import stopwords
from googletrans import Translator
import numpy as np

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
                        lemmas.add(stem)
                        
                        if len(lemmas) > 100:
                            return lemmas
                    
                if '<content' in line: # start of content
                    is_content = True 
                
            if '<article lang="en"' in line: # only english articles
                is_english = True
    return lemmas

def translate_en_it(en_words):
    it_words = set()
    en_it_pairs = {}
    translator = Translator()
    for en_word in en_words:
        translated = translator.translate(en_word, 'it', 'en')
        if translated.text == en_word: # not translated
            continue
        
        if len(translated.text.split()) > 1: # translated should be word not phrase
            for possible_translation in  translated.extra_data['possible-translations'][0][2]:
                if len(possible_translation[0].split()) == 1:
                    it_words.add(possible_translation[0])
                    en_it_pairs[en_word] = possible_translation[0]
                    break
        elif len(translated.text.split()) == 1:
            it_words.add(translated.text)
            en_it_pairs[en_word] = translated.text
    return it_words, en_it_pairs

def split_dataset(word_list, val_ratio=0.2, test_ratio=0.3):
    
    num_of_instance = len(word_list)
    indexes = np.arange(num_of_instance)
    np.random.shuffle(indexes)
    num_of_vals = int(split_ratio*num_of_instance)
    val_indexes = indexes[:num_of_vals]
    train_indexes = indexes[num_of_vals:]
    
    return


if __name__ == '__main__':
    
#     nltk.download()
    en_words = get_words()

            
            
    print("finish")