#Python code to extract the frames

import numpy as np
import cv2
import matplotlib.pyplot as plt
path = 'C:/Users/Anand/Desktop/IVPCode/UCF101/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'


def framesPerVideo(video_path):

	'''
	input: String -> The path of the video

	output: numpy array -> return all the frames in the video
	'''
	frames = []
	cap = cv2.VideoCapture(video_path)
	while(cap.isOpened()):
		ret, frame = cap.read();
		if not ret:
			break
		frames.append(frame)
	cap.release()
	cv2.destroyAllWindows()

	return np.array(frames)

plt.subplot(2, 3, 1)

plt.imshow(frames[1])
plt.subplot(2, 3, 2)
plt.imshow(frames[10])
plt.subplot(2,3,3)
plt.imshow(frames[19])