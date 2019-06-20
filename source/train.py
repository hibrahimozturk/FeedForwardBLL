import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from dataloader import BLLDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from bll_model import BLLModel
import argparse
import os
import pickle
import codecs
import json
import time

import numpy as np
from validation import validate

def draw_graph(values, x_label, y_label , output_dir, graph_name):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(values)
    plt.savefig(os.path.join(output_dir, graph_name + '.png'))
    plt.clf()

def train(args, bll_model, train_loader, val_loader, scheduler, optimizer, epochs, criterion, use_gpu):
    
    epoch_losses = []
    val_losses = []
    val_accs = []
       
    for epoch in range(0, epochs):
        
        iter_losses = []
        
        bll_model.train()
        print("####################################################")
        print('Training Epoch: ' + str(epoch))
        scheduler.step()

        buffer_losses = []
        epoch_start = time.time()
        
        # Epoch of Training
        for iter, batch in enumerate(train_loader):
#             cur_iter = iter + start_iter
            optimizer.zero_grad()

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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_losses.append(loss.item())
            buffer_losses.append(loss.item())
            
            if iter % 10 == 0:
                print("train epoch {}, iter{}/{}, loss: {}, elapsed {}"\
                      .format(epoch, iter, train_loader.__len__(), loss.item(), time.time()-epoch_start))
                draw_graph(iter_losses, "Iteration", "Loss", output_dir, "train_iter_loss")
            
        # End of epoch info
        epoch_end = time.time()
        hours, rem = divmod(epoch_end-epoch_start, 3600)
        minutes, seconds = divmod(rem, 60)
           
        epoch_losses.append(np.mean(np.array(buffer_losses)))
        val_losses, val_accs = validate(args, bll_model, val_loader, criterion, use_gpu, val_losses, val_accs, epoch)

        buffer_losses = []
        print("##### Finish epoch {}, time elapsed {}h {}m {}s #####".format(epoch, hours, minutes, seconds))
        print("####################################################")
        
        # Save epoch checkpoint
        if (epoch+1)%10 == 0:
            torch.save({'epoch':epoch,
                        'model_state_dict': bll_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': epoch_losses,
    #                     'val_results': val_results,
                        'iter_losses': iter_losses,
                        'epoch_continue': False,
                        'start_iter': 0
                }, os.path.join(output_dir, 'checkpoints', 'epoch_' + str(epoch) + '.checkpoint'))

        draw_graph(epoch_losses, "Epoch", "Loss", output_dir, "train_epoch_loss")
        draw_graph(val_losses, "Epoch", "Loss", output_dir, "val_epoch_loss")
        draw_graph(val_accs, "Epoch", "Loss", output_dir, "val_epoch_accs")

    
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Bilingual Lexicon Learning')
    
    parser.add_argument("-g", "--gpu", type=str, default="0",
    help="run on the gpu")

    parser.add_argument(
        '--exp', '-e',
        dest='exp_name', type=str, default="exp/exp1"
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

    lr         = 1e-4
    momentum   = 0.9
    epochs     = 100
    batchsize  = 64
    w_decay    = 0
    step_size  = 50
    gamma      = 0.2

    output_dir=os.path.join(args.exp_dir, args.exp_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
        os.makedirs(os.path.join(output_dir, 'checkpoints'))
        
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    with open('../data/wikicomp_dataset/train/train_vectors.pickle', 'rb') as handle:
        word_vectors = pickle.load(handle)
         
    with codecs.open('../data/wikicomp_dataset/train/train_filtered_set.json', 'r', "ISO-8859-1") as fp:
        en_it_pairs = json.load(fp) 

    train_data = BLLDataset(en_it_pairs['input_outputs'], word_vectors)
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=1)
    
    with open('../data/wikicomp_dataset/val/val_vectors.pickle', 'rb') as handle:
        word_vectors = pickle.load(handle)
         
    with codecs.open('../data/wikicomp_dataset/val/val_filtered_set.json', 'r', "ISO-8859-1") as fp:
        en_it_pairs = json.load(fp) 

    val_data = BLLDataset(en_it_pairs['input_outputs'], word_vectors)
    val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=1)

    bll_model = BLLModel()
    if use_gpu:
        ts = time.time()
        bll_model = bll_model.cuda()
        bll_model = nn.DataParallel(bll_model, device_ids=num_gpu)
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

    criterion = nn.BCELoss()
#     optimizer = optim.SGD(bll_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    optimizer = optim.Adam(bll_model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    train(args, bll_model, train_loader, val_loader, scheduler, optimizer, epochs, criterion, use_gpu)

    print("finished")
