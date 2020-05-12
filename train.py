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

'''
train images -> 11,000

image size -> (256, 256, 3)
'''

trainImagesPath = './predicted_images'
trueImagesPath = './true_images'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


temp = './dataset'

def getImages(path):
    files = os.listdir(path)
    data = np.zeros((len(files),256,256,3))
    for i, file in enumerate(files):
        image = cv2.imread(path + '/' + file)
        # mean = np.mean(image)
        # image = (image - mean) / np.std(image)
        image = image / np.max(image)
        data[i, :, :, :] = image
    return data

# predictedFrames = getImages(trainImagesPath)
# trueFrames = getImages(trueImagesPath)

t = getImages(temp)
p = getImages(temp)
tensor_y = torch.Tensor(p)
tensor_x = torch.Tensor(t)
tensor_x = tensor_x.permute(0, 3, 1, 2)
tensor_y = tensor_y.permute(0, 3, 1, 2)

x = torchvision.transforms.Normalize(mean = 0, std = 0.1)

dataset = data.TensorDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset, batch_size = 4)
images, labels = next(iter(dataloader))


model = Model(3, 3).getModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

model = model.to(device)

def train_model(model, criterion, optimizer, epochs = 20):
    model.train()
    losses = []
    for epoch in range(epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 70)
        running_loss = 0.0
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
        print('Loss after {} epoch is {}'.format(epoch, running_loss))
        losses.append(running_loss)
    return model, losses

model, losses = train_model(model, criterion, optimizer, 2)
