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


def validate(args, bll_model, val_loader, criterion, use_gpu, val_losses, val_accs, epoch):
    
    bll_model.eval()
    print()
    print("####################################################")
    print('Validation Epoch: ' + str(epoch))

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

        loss = criterion(outputs, labels)
        buffer_losses.append(loss.item())
        
        if iter % 10 == 0:
            print("validation epoch {}, iter{}/{}, loss: {}, elapsed {}"\
                  .format(epoch, iter, val_loader.__len__(), loss.item(), time.time()-epoch_start))
        
    # End of epoch info
    
    prediction = np.array(all_outputs) >= 0.5
    correct = prediction == np.array(all_targets)
    accuracy = ( np.sum(correct) / len(all_targets) ) *100
    print("validation epoch {}, accuracy: {}".format(epoch, accuracy))
    val_losses.append(np.mean(np.array(buffer_losses)))
    val_accs.append(accuracy)
    print("####################################################")
    
    
    return val_losses, val_accs