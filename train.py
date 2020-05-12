# import segmentation_models_pytorch as smp
#
#
# def UnetModel():
#     model = smp.Unet('resnet34')
#     return model

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
import os
import cv2
import torchvision
from Unet import Model
import torch.optim as optim
from torch import nn
import torchvision
from torchvision import transforms

'''
train images -> 11,000

image size -> (256, 256, 3)
'''

trainImagesPath = './predicted_images'
trueImagesPath = './true_images'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


root = './outputs'
predictedPath = root + '/predicted_images'
truePath = root + '/true_images'
def getFileNames(path):
    files = os.listdir(path)
    return files



class PredictedImages(Dataset):
    def __init__(self, predicted_path, true_path, transform):
        self.predictedImageList = getFileNames(predicted_path)
        self.trueImageList = getFileNames(true_path)
        self.transform = transform

    def __len__(self):
        return len(self.predictedImageList)

    def __getitem__(self, idx):
        predictedImageName = self.predictedImageList[idx]
        trueImageName = self.trueImageList[idx]
        predictedImage = cv2.imread(predictedPath + '/' + predictedImageName)
        trueImage = cv2.imread(truePath + '/' + trueImageName)
        return self.transform(predictedImage), self.transform(trueImage)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


print('---------loading data----------------')
dataset = PredictedImages(predictedPath, truePath, transform)
trainLoader  = DataLoader(dataset, 8)
print('------------------Data is loaded-----------------------')


model = Model(3, 3).getModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)



model = model.to(device)

def train_model(model, criterion, optimizer, epochs = 100):
    model.train()
    losses = []
    for epoch in range(epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 70)
        running_loss = 0.0
        for inputs, labels in trainLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            values, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print('Loss after {} epoch is {}'.format(epoch, running_loss))
        losses.append(running_loss)
    return model, losses

print('---------------model training---------------------')
model, losses = train_model(model, criterion, optimizer, 2)
