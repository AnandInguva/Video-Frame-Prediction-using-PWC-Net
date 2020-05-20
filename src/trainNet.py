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
from post_processing import ResNet
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

files = getFileNames(predictedPath)

class PredictedImages(Dataset):
    def __init__(self, predicted_path, true_path):
        self.predictedImageList = getFileNames(predicted_path)
        self.trueImageList = getFileNames(true_path)

    def __len__(self):
        return len(self.predictedImageList)

    def __getitem__(self, idx):
        predictedImageName = self.predictedImageList[idx]
        trueImageName = self.trueImageList[idx]
        predictedImage = cv2.imread(predictedPath + '/' + predictedImageName)
        trueImage = cv2.imread(truePath + '/' + trueImageName)
        predictedImage = torch.from_numpy(predictedImage)
        predictedImage = predictedImage.permute(2,0,1)
        trueImage = torch.from_numpy(trueImage)
        trueImage = trueImage.permute(2,0,1)
        return predictedImage.float(), trueImage.float()

def save_weights(model,loss, epoch, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss}, path)


print('------------------loading data----------------------')
print('device is {}'.format(device))
dataset = PredictedImages(predictedPath, truePath)
trainLoader  = DataLoader(dataset, 8)
print('------------------Data is loaded-----------------------')


# model = Model(3, 3).getModel()
model = ResNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
model = model.to(device)

def train_model(model, criterion, optimizer, epochs = 20):
    model.train()
    losses = []
    checkpoint = 0
    for epoch in range(epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 70)

        running_loss = 0.0
        for inputs, labels in trainLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print('Loss after {} epoch is {}'.format(epoch, running_loss / len(files)))
        losses.append(running_loss / len(files))

        if epoch % 4 == 0:
            save_weights(model, running_loss / len(files), epoch, './checkpoints/checkpoint{}.pt'.format(checkpoint))
            checkpoint += 1
    return model, losses

print('---------------model training---------------------')
model, losses = train_model(model, criterion, optimizer, 20)

PATH = 'C:/Users/Anand/Desktop'
torch.save(model.state_dict(), PATH + '/myweights.pt')
