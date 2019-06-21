import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import argparse
from bll_model import BLLModel
import time
import pickle
import json
import codecs

def parse_args():
    parser = argparse.ArgumentParser(description='Bilingual Lexicon Learning')
    
    parser.add_argument("-g", "--gpu", type=str, default="0",
    help="run on the gpu")

    parser.add_argument(
        '--exp', '-e',
        dest='exp_name', type=str, default="adam/shallow2-3/bn3-nodropout"
    )
    
    parser.add_argument(
        '--exp_dir',
        dest='exp_dir', type=str, default='../experiments')
    
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint', type=str , default="epoch_9.checkpoint"
        )
        
    return parser.parse_args()


def test(args, top_n=1):
    
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    
    bll_model = BLLModel()
    
    if use_gpu:
        ts = time.time()
        bll_model = bll_model.cuda()
        bll_model = nn.DataParallel(bll_model, device_ids=num_gpu)
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
    
    checkpoint = torch.load(os.path.join(args.exp_dir, args.exp_name, 'checkpoints', args.checkpoint))
    bll_model.load_state_dict(checkpoint['model_state_dict'])

    bll_model.eval()
    
    with open('../data/wikicomp_dataset/test/test_vectors.pickle', 'rb') as handle:
        word_vectors = pickle.load(handle)
         
    with codecs.open('../data/wikicomp_dataset/test/test_filtered_set.json', 'r', "ISO-8859-1") as fp:
        en_it_pairs = json.load(fp)
        
    num_of_true_prediction = 0
    num_of_total_prediction = 0
        
    for translation_pair in en_it_pairs['input_outputs']:
        if translation_pair['output'] == 1:
            probs = []
            en_word = translation_pair['english_word']
            true_it_word = translation_pair['italian_word']
            true_index = -1
            for idx,it_word in enumerate(en_it_pairs['it_words']):
                if it_word == true_it_word:
                    true_index = idx
                    
                src_word2vec = Variable( torch.Tensor( np.expand_dims(word_vectors[en_word], axis=0) ).float().cuda() )
                target_word2vec = Variable( torch.Tensor(np.expand_dims(word_vectors[it_word], axis=0) ).float().cuda() )
                output = bll_model(src_word2vec, target_word2vec)
                probs.append( output.data.cpu().numpy()[0,0] )
                
            sorted_probs = np.argsort(np.array(probs), axis=0)
            if true_index !=-1:
                num_of_total_prediction += 1
                if true_index in sorted_probs[-top_n:]:
                    print("True prediction")
                    num_of_true_prediction += 1
                else:
                    print("Wrong prediction")
            else:
                print("Warning! true index not found")
                return
    
    print("Top-{} Num of true prediction: {} Num of total prediction: {} Accuracy:{}".\
          format(top_n, num_of_true_prediction, num_of_total_prediction, num_of_true_prediction/num_of_total_prediction))
            
if __name__ == "__main__":

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    test(args, top_n=1)


    print("finish")