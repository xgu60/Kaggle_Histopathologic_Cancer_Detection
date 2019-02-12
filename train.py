from dataset import HistoDataset
from net import CNN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, os.path
import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import roc_curve, auc

def train_model(epoch=1, batchSize=64):
    EPOCH_NUM = epoch
    BATCH_SIZE = batchSize

    train_dataset = HistoDataset("data/train.csv", "data/train")
    val_dataset = HistoDataset("data/valid.csv", "data/train")
    train_data = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=2)
    val_data = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    
    net = CNN().cuda()
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.05, betas=(0.9, 0.99))

    for epoch in range(EPOCH_NUM):
        print("epoch: {}".format(epoch))
        for i, (samples, labels) in tqdm(enumerate(train_data)):
            net.train()
            inputs, labels = samples.view((samples.size(0),3, 96, 96)).cuda(),labels.view(labels.size(0), 1).float().cuda()
            #print(labels)
            y_pred = net(inputs)
            #print(y_pred)
            loss = criterion(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #evaluate on train set
        print(compute_auc(train_data, net))
        print(compute_auc(val_data, net))
        
    
def compute_auc(data, net):
    net.eval()
    pred_values = []
    true_values = []
    for j, (samples, labels) in enumerate(data):
        inputs = samples.view((samples.size(0),3, 96, 96)).cuda()
        pred_values += [x[0] for x in net(inputs).data.cpu().numpy()]
        true_values += [x for x in labels.numpy()]
    
    fpr, tpr, thres = roc_curve(np.array(true_values), np.array(pred_values))
    auc_score = auc(fpr, tpr)
    return auc_score


if __name__ == "__main__":
    train_model(10, 64)
    


