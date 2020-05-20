import cv2
import time
import torch
import pickle
import numpy as np

from forward_Warp import forward_warp


#im0 = cv2.imread("im0.png")[np.newaxis, :, :, :]
#im1 = cv2.imread("im1.png")[np.newaxis, :, :, :]
#with open("flow.pkl", "rb+") as f:
#   flow = pickle.load(f)


def warp_Forward(im0,im1,flow):

    im0 = torch.FloatTensor(im0).permute(0, 3, 1, 2)
    im1 = torch.FloatTensor(im1).permute(0, 3, 1, 2)

    flow = torch.FloatTensor(flow).permute(0,2,3,1)
    # flow = np.reshape(flow, (1, flow.shape[1], flow.shape[2], flow.shape[0]))
    # flow = torch.from_numpy(flow)

    fw = forward_warp("Nearest")

    since = time.time()
    im1_python = fw(im0, flow)
    print("python version forward cost time: {}".format(time.time()-since))

    # im0 = im0.cuda()
    # flow = flow.cuda()
    # since = time.time()
    # im1_cuda = fw(im0, flow)
    # print("cuda version forward cost time: {}".format(time.time()-since))


    loss_fn = torch.nn.MSELoss()
    python_loss = loss_fn(im1_python, im1)
    print("python loss: {}".format(python_loss))
    # cuda_loss = loss_fn(im1_cuda, im1.cuda())
    # print("cuda loss: {}".format(cuda_loss))

    im1_python = im1_python.permute(0, 2, 3, 1)[0]
    cv2.imwrite("im1_python_1.png", im1_python.numpy().astype(np.uint8))
    # im1_cuda = im1_cuda.permute(0, 2, 3, 1)[0]
    # cv2.imwrite("im_final_cuda_1.png", im1_cuda.cpu().numpy().astype(np.uint8))

    return im1_python
