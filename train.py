import torch
import numpy as np
from os import listdir
from Unet import Model
from test import *
import torch.optim as optim
from torch import nn
import os
model = Model(3,3).getModel()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print("---------------------------Model loaded--------------------------------")


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


def train_model(model, criterion, optimizer, epochs = 20):
    since = time.time()
    model.train()
    valLosses = []
    losses = []
    val_acc = 0
    for epoch in range(epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 15)
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            values, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
    return model, losses

def getData(root):
    trainPaths = os.listdir(root)
    return trainPaths

    
trainPaths = getData('dataset')
print(trainPaths)
trainImages = [extractFrames('dataset/' + path) for path in trainPaths]
truthLabels = []
trainLabels = []
for i in range(len(trainImages)):
    print(len(trainImages[i]))
    warped, label = getPredictedFrame(trainImages[i])
    trainLabels.append(warped)
    truthLabels.append(label)

for i in range(len(truthLabels)):
    print(truthLabels[i].shape)



    