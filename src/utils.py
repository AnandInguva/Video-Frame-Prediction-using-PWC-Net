import cv2
from run import Run
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from forw.warp import *

def resize(img, new_h = 256, new_w = 256):
	'''
	input: image
	output: resized image
	'''
	return cv2.resize(img, (new_h, new_w), interpolation = cv2.INTER_AREA)

def extractFrames(path, period = 5, numOfFrames = 5):

	'''
	input: video path
	output: Frames extracted from the vide
	'''

	count = 0
	cap = cv2.VideoCapture(path)   # capturing the video from the given path
	frameRate = cap.get(5) #frame rate
	x=1
	frames = []
	while(cap.isOpened() and count // period < numOfFrames):
		frameId = cap.get(1)
		ret, frame = cap.read()
		if count % period == 0:
			frames.append(resize(frame))
		count = count + 1
	cap.release()

	return frames

def runModel(path1, path2):
	output = Run(path1, path2).getOutput()
	return output

def vel_acc(flow14,flow13,flow12):
    flow34 = np.subtract(flow14,flow13)
    flow23 = np.subtract(flow13,flow12)
    v4 = np.subtract(flow34,flow23)
    v3 = np.subtract(flow23,flow12)
    a4 = np.subtract(v4,v3)
    v2 = np.add(a4,v3)
    flow01 = np.add(v2,flow12)
    flow10 =  -flow01
    return flow10


def findFlows(frames):
	'''
	output : flows, label

	flows -> [flow03, flow13, flow23]
	anchor -> frame4 -> frames[len(frames) -2]
	label -> frame 5 -> frames[len(frames) -1]
	'''
	# frames -> [0,1,2,3,4]
	flows = []
	for i in range(len(frames) - 2):
		flows.append(runModel(frames[len(frames) - 2],frames[i]))
	return flows, frames[len(frames) -2], frames[-1]


def getPredictedFrame(frames):
	flows, anchor, label = findFlows(frames)
	flow = vel_acc(flows[0], flows[1], flows[2])
	anchor = torch.from_numpy(anchor).float()
	label = torch.from_numpy(label).float()
	anchor = anchor.unsqueeze(0)
	label = label.unsqueeze(0)
	flow = flow.unsqueeze(0)
	warped = warp_Forward(anchor, label, flow)
	warped = warped.squeeze(0)
	label = label.squeeze(0)
	warped = warped.numpy()
	label = label.numpy()
	return warped, label
