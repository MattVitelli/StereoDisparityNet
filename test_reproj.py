# Imports
import numpy as np
import pfm
import cv2
import file_helper
import tensorflow as tf
def make_cost_volume(imgL, imgR, max_disparity):
	#Ld, Ly,Lx = np.mgrid[0:max_disparity, 0:imgR.shape[0], 0:imgR.shape[1]]
	Ld, Ly, Lx, Lc = np.meshgrid(range(max_disparity),range(imgR.shape[0]),range(imgR.shape[1]),range(imgR.shape[2]), indexing="ij")
	Lx = np.maximum(0, np.minimum(imgR.shape[1]-1, np.round(Lx - Ld))).astype(np.int32)

	imgRReshaped = np.reshape(imgR,[1,imgR.shape[0],imgR.shape[1], imgR.shape[2]])
	imgLReshaped = np.reshape(imgL,[1,imgL.shape[0],imgL.shape[1], imgL.shape[2]])

	imgL_extended = np.tile(imgLReshaped, [max_disparity,1,1,1])
	imgR_extended = np.tile(imgRReshaped, [max_disparity,1,1,1])
	print(Ld.shape)
	print(Ly.shape)
	print(Lx.shape)
	print(Lc.shape)
	print(imgR_extended.shape)
	cost_volume = np.concatenate((imgL_extended, imgR_extended[Ld, Ly, Lx, Lc]), axis=-1)

	num_channels = imgL.shape[2]
	cost_volume_naive = np.tile(imgL_extended, [1,1,1,2])
	for d in range(max_disparity):
		for y in range(imgL.shape[0]):
			for x in range(imgL.shape[1]):
				d_idx = max(0, min(imgR.shape[1]-1, x - d))
				cost_volume_naive[d,y,x,num_channels:] = imgR[y, d_idx,:]

	error = np.mean(np.mean(np.mean(np.mean(np.abs(cost_volume_naive - cost_volume)))))
	print("Cost Volume Error: " + str(error))
	print("Cost volume shapes:" + str(cost_volume.shape) +"\t" + str(cost_volume_naive.shape))

	#import pdb; pdb.set_trace()

	x = tf.placeholder('float64', (None, None, None, None))
	y = tf.placeholder('float64', (None, None, None, None))
	idx1 = tf.placeholder('int32', (None, None, None, None))
	idx2 = tf.placeholder('int32', (None, None, None, None))
	idx3 = tf.placeholder('int32', (None, None, None, None))
	idx4 = tf.placeholder('int32', (None, None, None, None))
	result = tf.gather_nd(y, tf.stack((idx1, idx2, idx3, idx4), -1))
	cv_result = tf.concat((x,result),axis=-1)
	with tf.Session() as sess:
		r = sess.run(cv_result, feed_dict={
			x: imgL_extended,
			y: imgR_extended,
			idx1: Ld,
			idx2: Ly,
			idx3: Lx,
			idx4: Lc
		})
		print("TF Shape: " + str(r.shape))
		tf_error = np.sum(np.sum(np.sum(np.sum(np.abs(cost_volume - r)))))
		print("TensorFlow error: " + str(tf_error))

	exit()



def resample_img(imgL, imgR, disparityL, disparityR, max_disparity):
	newXsL = np.tile(np.array(np.array(range(imgR.shape[1]))), (imgR.shape[0],1))
	newXsL = np.maximum(0, np.minimum(imgR.shape[1]-1, np.round(newXsL - disparityL)))
	newXsL = newXsL.astype(int)
	newYsL = np.tile(np.array(np.array(range(imgR.shape[0]))), (1, imgR.shape[1]))
	newYsL = np.array(range(imgR.shape[0]))

	newXsR = np.tile(np.array(np.array(range(imgL.shape[1]))), (imgL.shape[0],1))
	newXsR = np.maximum(0, np.minimum(imgL.shape[1]-1, np.round(newXsR + disparityR)))
	newXsR = newXsR.astype(int)
	newYsR = np.tile(np.array(np.array(range(imgL.shape[0]))), (1, imgL.shape[1]))
	newYsR = np.array(range(imgL.shape[0]))

	Ly,Lx = np.meshgrid(range(imgR.shape[0]),range(imgR.shape[1]), indexing="ij")
	#Ly,Lx = np.mgrid[0:imgR.shape[0], 0:imgR.shape[1]]
	Lx = np.maximum(0, np.minimum(imgR.shape[1]-1, np.round(Lx - disparityL))).astype(np.int32)
	Ry,Rx = np.meshgrid(range(imgL.shape[0]),range(imgL.shape[1]),indexing="ij")
	#Ry,Rx = np.mgrid[0:imgL.shape[0], 0:imgL.shape[1]]
	Rx = np.maximum(0, np.minimum(imgL.shape[1]-1, np.round(Rx + disparityR))).astype(np.int32)

	newImgL = imgR[Ly, Lx, :]
	newImgR = imgL[Ry, Rx, :]

	newImgL_slow = np.zeros(imgL.shape, dtype=np.uint8)
	newImgR_slow = np.zeros(imgR.shape, dtype=np.uint8)
	#newImg[:,:,:] = imgR[newXs[:],:]
	for y in range(imgL.shape[0]):
		newImgL_slow[y,:,:] = imgR[y, newXsL[y], :]
		newImgR_slow[y,:,:] = imgL[y, newXsR[y], :]
		#newImgR[y,newXs[y],:] = imgL[y,:,:]
		#for x in range(imgL.shape[1]):
		#	disIdx = int(round(x - disparity[y,x]))
		#	if disIdx < 0 or disIdx >= imgL.shape[1]:
		#		newImgL[y,x,:] = 0
		#	else:
		#		newImgL[y,x,:] = imgR[y,disIdx,:]
	errL = np.mean(np.mean(np.mean(np.abs(newImgL_slow - newImgL))))
	errR = np.mean(np.mean(np.mean(np.abs(newImgR_slow - newImgR))))
	print("Error = " + str(errL) + " " + str(errR))
	#newImg = imgR[:,newXs,:]
	#import pdb; pdb.set_trace()
	cv2.imshow("ErrorL", np.abs(imgL.astype(np.float32) - newImgL.astype(np.float32))/255.0)
	cv2.imshow("ErrorR", np.abs(imgR.astype(np.float32) - newImgR.astype(np.float32))/255.0)
	cv2.imshow("ImgL", imgL)
	cv2.imshow("ImgR", imgR)
	cv2.imshow("DispL", disparityL / max_disparity)
	cv2.imshow("DispR", disparityR / max_disparity)
	cv2.imshow("NewL", newImgL)
	cv2.imshow("NewR", newImgR)
	cv2.waitKey(0)
	exit()

w = 128
h = 128
c = 4
tmpL = np.random.rand(h,w,c)
tmpR = np.random.rand(h,w,c)
make_cost_volume(tmpL, tmpR, int(192/2))

img_dir = "D:/Datasets/SceneFlow/flyingthings3d_frames_finalpass_webp/frames_finalpass_webp/TEST/"
disp_dir = "D:/Datasets/SceneFlow/flyingthings3d_disparity/disparity/TEST/"

disps_and_imgs_files = file_helper.get_imgs_and_disps(img_dir, disp_dir)

max_disparity = 192.0
for disp_img in disps_and_imgs_files:
	dispL, scaleL = pfm.load_pfm(disp_img.disparityLeftFilename)
	dispR, scaleR = pfm.load_pfm(disp_img.disparityRightFilename)
	dispL = cv2.flip(dispL, 0)
	dispR = cv2.flip(dispR, 0)
	imgL = cv2.imread(disp_img.imageLeftFilename)
	imgR = cv2.imread(disp_img.imageRightFilename)
	resample_img(imgL, imgR, dispL, dispR, max_disparity)	
	#import pdb; pdb.set_trace()
	dispL = dispL / max_disparity
	dispR = dispR / max_disparity

	cv2.imshow('ImgLeft', imgL)
	cv2.imshow('ImgRight', imgR)
	cv2.imshow('DisparityLeft', dispL)
	cv2.imshow('DisparityRight', dispR)
	
	key = cv2.waitKey(0)
	if key == ord('q'):
		break