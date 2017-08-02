import numpy as np
import pfm
import cv2
import file_helper
import random

class DatasetHelper:
	def __init__(self):
		img_dir_test = "D:/Datasets/SceneFlow/flyingthings3d_frames_finalpass_webp/frames_finalpass_webp/TEST/"
		disp_dir_test = "D:/Datasets/SceneFlow/flyingthings3d_disparity/disparity/TEST/"
		img_dir_train = "D:/Datasets/SceneFlow/flyingthings3d_frames_finalpass_webp/frames_finalpass_webp/TRAIN/"
		disp_dir_train = "D:/Datasets/SceneFlow/flyingthings3d_disparity/disparity/TRAIN/"
		self.train_set = file_helper.get_imgs_and_disps(img_dir_train, disp_dir_train)
		self.test_set = file_helper.get_imgs_and_disps(img_dir_test, disp_dir_test)
		random.shuffle(self.train_set)
		random.shuffle(self.test_set)
		self.train_idx = 0
		self.test_idx = 0

	def load_image(self, img_data):
		dispL, scaleL = pfm.load_pfm(img_data.disparityLeftFilename)
		dispR, scaleR = pfm.load_pfm(img_data.disparityRightFilename)
		dispL = cv2.flip(dispL, 0)
		dispR = cv2.flip(dispR, 0)
		imgL = cv2.imread(img_data.imageLeftFilename).astype(np.float32) / 255.0
		imgR = cv2.imread(img_data.imageRightFilename).astype(np.float32) / 255.0
		imgL = imgL * 2.0 - 1.0
		imgR = imgR * 2.0 - 1.0
		return imgL, imgR, dispL, dispR

	def apply_augmentations(self, iL, iR, dL, dR, width, height):
		randX = np.random.randint(0,iL.shape[1] - 1 - width)
		randY = np.random.randint(0,iL.shape[0] - 1 - height)
		iLC = iL[randY:randY+height,randX:randX+width]
		iRC = iR[randY:randY+height,randX:randX+width]
		dLC = dL[randY:randY+height,randX:randX+width]
		dRC = dR[randY:randY+height,randX:randX+width]
		return iLC, iRC, dLC, dRC

	def hslAug(self, inImg, hueShift, lightness, saturation, gamma):
		rescaled = np.minimum(1, np.maximum(0,inImg*0.5+0.5))
		degamma = np.power(rescaled, 2.2)
		hsv = cv2.cvtColor(degamma*255.0, cv2.COLOR_BGR2HSV)
		hue = hsv[:,:,0]
		hue += hueShift
		hue[(hue > 359)] -= 359
		hsv[:,:, 0] = hue
		hsv[:,:, 1] *= saturation
		hsv[:,:, 2] *= lightness
		outImg = np.minimum(1, np.maximum(0, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)/255.0))
		outImg = np.minimum(1, np.maximum(0, np.power(outImg, 1.0/gamma)))
		return outImg*2.0-1.0


	def apply_augmentations_better(self, iL, iR, dL, dR, width, height):
		randX = np.random.randint(0,iL.shape[1] - 1 - width)
		randY = np.random.randint(0,iL.shape[0] - 1 - height)
		iLC = iL[randY:randY+height,randX:randX+width]
		iRC = iR[randY:randY+height,randX:randX+width]
		dLC = dL[randY:randY+height,randX:randX+width]
		dRC = dR[randY:randY+height,randX:randX+width]
		randHue = np.random.uniform(0,359)
		randSat = np.random.uniform(0.3, 1)
		randLight = np.random.uniform(0.3, 1.1)
		randGamma = np.random.uniform(1.1,2.6)
		iLC = self.hslAug(iLC, randHue, randSat, randLight, randGamma)
		iRC = self.hslAug(iRC, randHue, randSat, randLight, randGamma)
		return iLC, iRC, dLC, dRC

	def sample_from_training_set(self, width, height):
		img_data = self.train_set[self.train_idx]
		iL, iR, dL, dR = self.load_image(img_data)
		self.train_idx = (self.train_idx + 1) % len(self.train_set)
		return self.apply_augmentations(iL, iR, dL, dR, width, height)

	def sample_from_training_set_better(self, width, height):
		img_data = self.train_set[self.train_idx]
		iL, iR, dL, dR = self.load_image(img_data)
		self.train_idx = (self.train_idx + 1) % len(self.train_set)
		return self.apply_augmentations_better(iL, iR, dL, dR, width, height)

	def sample_from_test_set(self, width, height):
		img_data = self.test_set[self.test_idx]
		iL, iR, dL, dR = self.load_image(img_data)
		self.test_idx = (self.test_idx + 1) % len(self.test_set)
		return self.apply_augmentations(iL, iR, dL, dR, width, height)

	def sample_from_test_set_better(self, width, height):
		img_data = self.test_set[self.test_idx]
		iL, iR, dL, dR = self.load_image(img_data)
		self.test_idx = (self.test_idx + 1) % len(self.test_set)
		return self.apply_augmentations_better(iL, iR, dL, dR, width, height)

