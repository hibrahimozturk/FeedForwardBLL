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


def validate(args, bll_model, val_loader, criterion, use_gpu, val_losses, epoch):
    
    bll_model.eval()
    print()
    print("####################################################")
    print('Validation Epoch: ' + str(epoch))

    buffer_losses = []
    epoch_start = time.time()
    
    # Epoch of Training
    for iter, batch in enumerate(val_loader):

        if use_gpu:
            src_word2vec = Variable(batch['src_word2vec'].cuda())
            target_word2vec = Variable(batch['target_word2vec'].cuda())
            labels = Variable(batch['output'].float().cuda())
        else:
            src_word2vec = Variable(batch['src_word2vec'])
            target_word2vec = Variable(batch['target_word2vec'])
            labels = Variable(batch['output'])

        # Train one iteration
        outputs = bll_model(src_word2vec, target_word2vec)
        loss = criterion(outputs, labels)
        buffer_losses.append(loss.item())
        
        if iter % 10 == 0:
            print("validation epoch {}, iter{}/{}, loss: {}, elapsed {}"\
                  .format(epoch, iter, val_loader.__len__(), loss.item(), time.time()-epoch_start))
        
    # End of epoch info
       
    val_losses.append(np.mean(np.array(buffer_losses)))
    print("####################################################")
    
    
    return val_losses