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
from dataloader import BLLDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

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

    parser.add_argument(
        '--topk', action='store_true',
        dest='topk'
        )
        
    return parser.parse_args()


def top_k(args,bll_model,word_vectors,en_it_pairs, top_n=1):
    

        
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
            
            
def validate(args, bll_model, val_loader, use_gpu):
    
    bll_model.eval()
    print()
    print("####################################################")

    buffer_losses = []
    epoch_start = time.time()
    
    all_targets = []
    all_outputs = []
    
    # Epoch of Training
    for iter, batch in enumerate(val_loader):

        if use_gpu:
            src_word2vec = Variable(batch['src_word2vec'].float().cuda())
            target_word2vec = Variable(batch['target_word2vec'].float().cuda())
            labels = Variable(batch['output'].float().cuda())
        else:
            src_word2vec = Variable(batch['src_word2vec'])
            target_word2vec = Variable(batch['target_word2vec'])
            labels = Variable(batch['output'])

        # Train one iteration
        outputs = bll_model(src_word2vec, target_word2vec)
        all_outputs += outputs.data.cpu().numpy().squeeze().tolist()
        all_targets += labels.cpu().numpy().squeeze().tolist()

        
        if iter % 10 == 0:
            print("Test  iter{}/{}, elapsed {}"\
                  .format(iter, val_loader.__len__(), time.time()-epoch_start))
        
    # End of epoch info
    
    prediction = np.array(all_outputs) >= 0.5
    correct = prediction == np.array(all_targets)
    accuracy = ( np.sum(correct) / len(all_targets) ) *100
    [pr, rc, f1, _] = precision_recall_fscore_support(np.array(all_targets), prediction, average='binary')
    
    print("Test accuracy: {} precision: {} recall:{} f1: {}".format(accuracy, pr, rc, f1))
    print("####################################################")
    
    
    return

if __name__ == "__main__":

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


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


    test_data = BLLDataset(en_it_pairs['input_outputs'], word_vectors)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=1)
    
    
    validate(args, bll_model, test_loader, use_gpu)
    
    if args.topk:
        top_k(args, top_n=1)


    print("finish")