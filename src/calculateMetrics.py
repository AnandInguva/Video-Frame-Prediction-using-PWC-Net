import numpy as np
import os
import torch

import cv2

ssim_values = []
psnr_values = []

predicted_image_path = './predictions/predictedframe'
true_image_path = './predictions/trueframe'



def calculatePSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse))
def calculateSSIM(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


length  = len(os.listdir('./predictions')) // 2

for i in range(length - 1):
    img1 = cv2.imread(predicted_image_path + '{}.png'.format(i + 1))
    img2 = cv2.imread(true_image_path + '{}.png'.format(i + 1))

    psnr_values.append(calculatePSNR(img1, img2))
    ssim_values.append(calculateSSIM(img1, img2))

print('Average PSNR over 2000 images :' ,np.average(psnr_values))
print('Average SSIM over 2000 images :', np.average(ssim_values))
