import nltk
import codecs
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string 
from nltk.corpus import stopwords

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

if __name__ == '__main__':
    
#     nltk.download()
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stems = set()
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
                    words = result = [i for i in words if not i in stop_words]
    #                 print(words)
                    for word in words:
                        stem = porter.stem(word)
                        print("{} : {}".format(word, stem))
                        stems.add(stem)
                    
                if '<content' in line: # start of content
                    is_content = True 
                
            if '<article lang="en"' in line: # only english articles
                is_english = True
    print("finish")