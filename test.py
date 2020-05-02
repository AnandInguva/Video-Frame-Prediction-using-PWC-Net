import cv2
from run import Run
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from IPython.display import Image, display


path = 'v_BrushingTeeth_g01_c01.avi'
# path = 'v_FrisbeeCatch_g01_c01.avi'

def resize(img):
	new_h = 128
	new_w = 128
	return cv2.resize(img, (new_h, new_w), interpolation = cv2.INTER_AREA)


'''
frames per video -> 16

'''

def extractFrames(path):

	'''
	Extracted 5 frames with 
	'''

	print(path)
	count = 0
	cap = cv2.VideoCapture(path)   # capturing the video from the given path
	frameRate = cap.get(5) #frame rate
	x=1
	frames = []
	while(cap.isOpened() and count // 5 < 5):
		frameId = cap.get(1)
		ret, frame = cap.read()
		if count % 5 == 0:
			frames.append(resize(frame))
		count = count + 1
	cap.release()
	print('Frames of the video are extracted')
	return frames
        
frames = extractFrames(path)
print(len(frames))


# for i in range(len(frames)):
# 	cv2.imwrite('frame{}.jpg'.format(i), frames[i])


def runModel(path1, path2):
	output = Run(path1, path2).getOutput()
	print('done')
	return output


# def readFlow(name):
#     if name.endswith('.pfm') or name.endswith('.PFM'):
#         return readPFM(name)[0][:,:,0:2]
#     f = open(name, 'rb')
#     header = f.read(4)
#     if header.decode("utf-8") != 'PIEH':
#         raise Exception('Flow file header does not contain PIEH')
#     width = np.fromfile(f, np.int32, 1).squeeze()
#     height = np.fromfile(f, np.int32, 1).squeeze()
#     flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
#     return flow.astype(np.float32)


def vel_acc(flow14,flow13,flow12):
    flow34 = np.subtract(flow14,flow13)
    flow23 = np.subtract(flow13,flow12)
    v4 = np.subtract(flow34,flow23)
    v3 = np.subtract(flow23,flow12)
    a4 = np.subtract(v4,v3)
    v2 = np.add(a4,v3)
    flow01 = np.add(v2,flow12)
    flow10 =  flow01
    return flow10



def findFlows(frames):
	flows = []
	# frames -> [0,1,2,3,4]
	for i in range(len(frames) - 2):
		flows.append(runModel(frames[i], frames[len(frames) - 2]))

	'''
	output : flows, label

	flows -> [flow03, flow13, flow23]
	anchor -> frame4 -> frames[len(frames) -2]
	label -> frame 5 -> frames[len(frames) -1]
	'''
	return flows, frames[len(frames) -2], frames[len(frames) - 1]




# paths = ['frame1.jpg', 'frame2.jpg', 'frame3.jpg', 'frame4.jpg']
# flows = []
# for i in range(0, len(paths) - 1):
# 	print(i)
# 	flows.append(Run(paths[i], paths[len(paths) - 1]).getOutput())

print('Flows are computed')
print('----------------------------------------------------------------------------------------')




# tenOutput = tenOutput.permute(2,0,1)

def warp(x, flo): #(ref_frame, flow)
        B,C,H,W = x.shape
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        print('xx shape is :', xx.shape)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        print('yy shape is :', yy.shape)

        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone()/ max(H - 1, 1) - 1.0

        
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid).clone() 
        
        if not torch.cuda.is_available():
            mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        else:
            mask = torch.autograd.Variable(torch.ones(x.size()))
            
        mask = nn.functional.grid_sample(mask, vgrid).clone()

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

# print('Image size :', image.size())
# print(tenOutput.size())

def getPredictedFrame(frames):
	flows, anchor, label = findFlows(frames)
	velacc = vel_acc(flows[0], flows[1], flows[2])
	image = anchor
	image=image.astype(float)
	image = np.reshape(image,(1,image.shape[2],image.shape[0],image.shape[1]))
	image = torch.from_numpy(image)
	result = warp(image.float(), velacc)
	print('--------------------------------------Got the predicted frame-------------------------------------')
	warped = result.numpy()
	warped = np.reshape(warped,(result.shape[2], result.shape[3],result.shape[1]))
	return warped, label

# tenOutput = vel_acc(flows[0], flows[1], flows[2])
# tenOutput = torch.from_numpy(tenOutput)


# image = anchor
# image=image.astype(float)
# image = np.reshape(image,(1,image.shape[2],image.shape[0],image.shape[1]))
# image = torch.from_numpy(image)


# result = warp(image.float(), tenOutput)

# warped = result.numpy()
# warped = np.reshape(warped,(result.shape[2], result.shape[3],result.shape[1]))
# # plt.imshow(warped.astype(int)/np.max(warped))

# cv2.imwrite('Warped_Image.jpg', warped.astype(float)) 

'''
Output -> warped image

warped Image -> NN -> remove occlusion holes -> clear Image

size = 128 * 128 * 3

frames -> labels , warped images -> input, loss = output - frames

'''



