# Imports
import numpy as np
import pfm
import cv2
import file_helper

img_dir = "D:/Datasets/SceneFlow/flyingthings3d_frames_finalpass_webp/frames_finalpass_webp/TEST/"
disp_dir = "D:/Datasets/SceneFlow/flyingthings3d_disparity/disparity/TEST/"

disps_and_imgs_files = file_helper.get_imgs_and_disps(img_dir, disp_dir)

for disp_img in disps_and_imgs_files:
	dispL, scaleL = pfm.load_pfm(disp_img.disparityLeftFilename)
	dispR, scaleR = pfm.load_pfm(disp_img.disparityRightFilename)
	dispL = cv2.flip(dispL, 0)
	dispR = cv2.flip(dispR, 0)
	imgL = cv2.imread(disp_img.imageLeftFilename)
	imgR = cv2.imread(disp_img.imageRightFilename)
	#import pdb; pdb.set_trace()
	dispL = dispL / np.max(np.max(dispL))
	dispR = dispR / np.max(np.max(dispR))

	cv2.imshow('ImgLeft', imgL)
	cv2.imshow('ImgRight', imgR)
	cv2.imshow('DisparityLeft', dispL)
	cv2.imshow('DisparityRight', dispR)
	
	key = cv2.waitKey(0)
	if key == ord('q'):
		break