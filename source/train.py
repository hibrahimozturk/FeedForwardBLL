import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from dataloader import BLLDataset
from torch.utils.data import DataLoader

from bll_model import BLLModel
import argparse
import os
import pickle
import codecs
import json
import time

def train(args):
    
        
    for epoch in range(0, args.epochs):
        fcn_model.train()
        print("####################################################")
        print('Training Epoch: ' + str(epoch))
        scheduler.step()

        buffer_losses = []
        epoch_start = time.time()
        
        # Epoch of Training
        for iter, batch in enumerate(train_loader):
            cur_iter = iter + start_iter
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['smap_Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['smap_Y'])

            # Train one iteration
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            buffer_losses.append(loss.item())
            
            if cur_iter % 10 == 0:
                print("train epoch {}, iter{}/{}, loss: {}, elapsed {}"\
                      .format(epoch, cur_iter, train_loader.__len__(), loss.item(), time.time()-epoch_start))
                draw_graph(iter_losses, "Iteration", "Loss", output_dir, "train_iter_loss")
            
            # Save checkpoint
            if cur_iter % 50 == 0:
                torch.save({'epoch':epoch,
                    'model_state_dict': fcn_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_losses,
                    'val_results': val_results,
                    'iter_losses': iter_losses, 
                    'epoch_continue': True,
                    'start_iter': cur_iter
                    }, os.path.join(output_dir, 'checkpoints', 'epoch_' + str(epoch) + '_iter_' + str(cur_iter) + '.checkpoint'))
        
    
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Bilingual Lexicon Learning')
    
    parser.add_argument("-g", "--gpu", type=str, default="0",
    help="run on the gpu")

    parser.add_argument(
        '--exp', '-e',
        dest='exp_name', type=str, default="exp"
    )
    
    parser.add_argument(
        '--exp_dir',
        dest='exp_dir', type=str, default='../experiments')
    
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint', type=str)
    
    parser.add_argument(
        "--batchsize", type=int, default=64,
        dest='batchsize', help="batchsize")
    
    parser.add_argument(
        "--epochs", type=int, default=10,
        dest='epochs', help="batchsize")
    
    parser.add_argument(
        '--validate', dest='validate', action='store_true',
        help='evaluate model on validation set')
    
    parser.add_argument(
        '--test', dest='test', action='store_true',
        help='evaluate model on test set')
        
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    output_dir=os.path.join(args.exp_dir, args.exp_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    with open('../data/wikicomp_dataset/train/train_vectors.pickle', 'rb') as handle:
        word_vectors = pickle.load(handle)
         
    with codecs.open('../data/wikicomp_dataset/train/train_filtered_set.json', 'r', "ISO-8859-1") as fp:
        en_it_pairs = json.load(fp) 

    train_data = BLLDataset(en_it_pairs['input_outputs'], word_vectors)
    train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=1)

    bll_model = BLLModel()
    if use_gpu:
        ts = time.time()
        bll_model = bll_model.cuda()
        bll_model = nn.DataParallel(bll_model, device_ids=num_gpu)
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

    criterion = nn.MSELoss()

    print("finished")
