import nltk
import codecs
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string 
from nltk.corpus import stopwords

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

    is_content = False
    with  codecs.open('../../data/wikicomp-2014_enit.xml', 'r', encoding='UTF-8') as f:
        for line in f:

            if '</content' in line:
                is_content = False
                
            if is_content:
#                 print(line)
#                 print(porter.stem(line))
                content = line.strip().split('>')[1].split('<')[0]
                content = preprocess_text(content)
                words = word_tokenize(content)
                words = result = [i for i in words if not i in stop_words]
            if '<content' in line:
                is_content = True 
    print("finish")