import torch
import torchvision
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from post_processing import ResNet
import numpy as np

def getFileNames(path):
    files = os.listdir(path)
    return files


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


print('weights are loaded')

#Place the outputs in the directory as described below
predictedPath = './outputs/predicted_images'
truePath = './outputs/true_images'


transform = transforms.Compose([transforms.ToTensor()])

testData = PredictedImages(predictedPath, truePath)

testLoader = DataLoader(testData, batch_size = 128)

images, labels = next(iter(testLoader))

model1 = ResNet()
model = model1.cuda()

model1.load_state_dict(torch.load('resnet_weights.pt'))
model1.eval()

for images, labels in testLoader:
    images = images
    labels  = labels
    model = model.cpu()
    print(images.shape)
    outputs = model(images)
    break

labels = labels.cpu()
for i in range(outputs.size()[0]):
    out = outputs[i]
    out = out.permute(1,2,0)
    out = out.detach().numpy()
    cv2.imwrite('predictions/predictedframe{}.png'.format(i + 1), out)
    label = labels[i]
    label = label.permute(1,2,0)
    label = label.numpy()
    cv2.imwrite('predictions/trueframe{}.png'.format(i+ 1), label)
