import cv2
import numpy as np
import dataset_helper

sceneflow = dataset_helper.DatasetHelper()

max_disparity = 192.0
width = 512
height = 256
while(True):
	aIL, aIR, aDL, aDR = sceneflow.sample_from_training_set_better(width=width,height=height)
	aDL = aDL / max_disparity
	aDR = aDR / max_disparity
	cv2.imshow('TrainIL', aIL*0.5+0.5)
	cv2.imshow('TrainIR', aIR*0.5+0.5)
	cv2.imshow('TrainDL', aDL)
	cv2.imshow('TrainDR', aDR)

	aIL, aIR, aDL, aDR = sceneflow.sample_from_test_set(width=width,height=height)
	aDL = aDL / max_disparity
	aDR = aDR / max_disparity
	cv2.imshow('TestIL', aIL*0.5+0.5)
	cv2.imshow('TestIR', aIR*0.5+0.5)
	cv2.imshow('TestDL', aDL)
	cv2.imshow('TestDR', aDR)

	shuffled = np.transpose(aIL, (2, 0, 1))
	print(shuffled.shape)
	
	cv2.imwrite("./test_l.png", (255.0*(aIL*0.5+0.5)).astype(np.uint8))
	cv2.imwrite("./test_r.png", (255.0*(aIR*0.5+0.5)).astype(np.uint8))
	cv2.imwrite("./test_disp.png", (255.0*aDR).astype(np.uint8))
	key = cv2.waitKey(0)
	if key == ord('q'):
		break