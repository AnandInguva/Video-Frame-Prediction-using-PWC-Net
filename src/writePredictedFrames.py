import numpy as np
import os
import torch
import cv2
from utils import extractFrames, getPredictedFrame

#File to save all the warped and truth label to the folders 'predicted path' and 'true_path'.


root = './ucf/ucf/'

predicted_path = './predicted_images/'
true_path = './true_images/'

files = os.listdir(root)

totalVideos = len(files) #13320
print(totalVideos)

for i in range(totalVideos):
    frames = extractFrames(root + files[i], 2, 5)
    print('------------------------------------------------------')
    warped, label = getPredictedFrame(frames)
    # cv2.imwrite(os.path.join(predicted_path + 'predicted' + str(i), '.jpg'), warped)
    # cv2.imwrite(os.path.join(predicted_path + 'true' + str(i), '.jpg'), label)
    cv2.imwrite(os.path.join(predicted_path, '{}.jpg'.format(i)), warped)
    cv2.imwrite(os.path.join(true_path, '{}.jpg'.format(i)), label)

    print("Image {} is done".format(i + 1))
