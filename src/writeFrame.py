from test import *
from run import *
import cv2
import torch
import torchvision
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from post_processing import ResNet
import numpy as np

path = 'v_BrushingTeeth_g01_c01.avi'
trainImages = extractFrames(path, 2, 5)

for i in range(len(trainImages)):
    cv2.imwrite('frame{}.jpg'.format(i), trainImages[i])


warped, label = getPredictedFrame(trainImages)
print(warped.shape)

cv2.imwrite('WARPED_IMAGE.jpg', warped)
cv2.imwrite('Predicted_image.jpg', label)

model1 = ResNet()
# model = model1.cuda()
model1.load_state_dict(torch.load('resnet_weights.pt'))
model1.eval()


image = torch.from_numpy(warped)
image = image.unsqueeze(0)
image = image.permute(0,3,1,2)

print(image)
output = model1(image)

output = torch.squeeze(output)
output = output.permute(1,2,0)
output = output.numpy()
cv2.imwrite('refined_fifth_frame.jpg', output)
