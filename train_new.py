from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dataset_helper
import cv2
import math
import model_new
import numpy as np
import sys

def update_console(text):
	sys.stdout.write("\r" + text)
	sys.stdout.flush() # important
def end_console():
	print("\n")

# model I/O settings
model_path = "./models_new/best_model.tfmodel"
model_base_path = "./models_new/disparity_"
last_model_path = "./models_new/last_model.tfmodel"
old_model_path = last_model_path#model_path
force_save_number = 10000

# training settings
num_iterations = 150000
iterations_per_epoch = 10
num_epochs = math.ceil(float(num_iterations) / float(iterations_per_epoch))
num_eval_images_per_epoch = 10

# learning settings
learning_rate = 1.0e-3

# model settings
num_features = 32
num_disparities = 192
width = 512
height = 256
channels = 3

# tf Graph input
left_input = tf.placeholder(tf.float32, shape=[1, channels, height, width])
right_input = tf.placeholder(tf.float32, shape=[1, channels, height, width])
true_disparity = tf.placeholder(tf.float32, shape=[1, height, width])
isTraining = tf.placeholder(tf.bool)

predicted_disparity = model_new.make_disparity_model(left_input, right_input, num_features, num_disparities, isTraining)

cost = tf.losses.absolute_difference(labels=true_disparity, predictions=predicted_disparity)
#supposedly these two commands should make BN work at test time
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
best_val_cost = float('Inf')

# Initializing the variables
init = tf.global_variables_initializer()
print("Model initialized!")

sceneflow = dataset_helper.DatasetHelper()
print("Dataset Initialized")

# Running first session
print("Starting session...")
with tf.Session() as sess:
	# Initialize variables
	sess.run(init)

	if old_model_path is not None:
		print("Loaded from old file: " + old_model_path)
		saver.restore(sess, old_model_path)

	iters_till_next_save = force_save_number
	# Training cycle
	for epoch in range(num_epochs):
		print("Starting epoch " + str(epoch + 1) + "/" + str(num_epochs))
		# train the model on a small batch
		best_train_data_cost = float('Inf')
		best_train_data = None
		avg_cost = 0.
		acc_1 = 0.
		acc_5 = 0.
		for it in range(iterations_per_epoch):
			aIL, aIR, aDL, aDR = sceneflow.sample_from_training_set_better(width=width,height=height)
			aILShuffled = np.transpose(aIL, (2, 0, 1))
			aIRShuffled = np.transpose(aIR, (2, 0, 1))
			left_img = np.reshape(aILShuffled, [1, aILShuffled.shape[0], aILShuffled.shape[1], aILShuffled.shape[2]])
			right_img = np.reshape(aIRShuffled, [1, aIRShuffled.shape[0], aIRShuffled.shape[1], aIRShuffled.shape[2]])
			disparity_gt = np.reshape(aDL, [1, aDL.shape[0], aDL.shape[1]])
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c, prediction, _ = sess.run([optimizer, cost, predicted_disparity, extra_update_ops], feed_dict={left_input: left_img,
														right_input: right_img,
														true_disparity: disparity_gt,
														isTraining: True})
			if c < best_train_data_cost:
				best_train_data_cost = c
				best_train_data = (aIL, aIR, aDL, np.squeeze(prediction))
			avg_cost += (c - avg_cost) / float(it+1)
			absDiff = np.abs(prediction - disparity_gt)
			acc_1 += (np.mean(absDiff < 1) - acc_1) / float(it+1)
			acc_5 += (np.mean(absDiff < 5) - acc_5) / float(it+1)
			update_console("Train Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + 
				"\tProgress: " + str(float(it+1)/float(iterations_per_epoch)) +
				"\tLoss: " + str(avg_cost) + "\tAcc 1: " + str(acc_1) + "\tAcc 5: " + str(acc_5))
		end_console()

		# test the model on the validation set
		print("Validating on validation set...")
		best_val_data_cost = float('Inf')
		best_val_data = None
		val_cost = 0.
		val_acc_1 = 0.
		val_acc_5 = 0.
		for it in range(num_eval_images_per_epoch):
			aIL, aIR, aDL, aDR = sceneflow.sample_from_test_set(width=width,height=height)
			aILShuffled = np.transpose(aIL, (2, 0, 1))
			aIRShuffled = np.transpose(aIR, (2, 0, 1))
			left_img = np.reshape(aILShuffled, [1, aILShuffled.shape[0], aILShuffled.shape[1], aILShuffled.shape[2]])
			right_img = np.reshape(aIRShuffled, [1, aIRShuffled.shape[0], aIRShuffled.shape[1], aIRShuffled.shape[2]])
			disparity_gt = np.reshape(aDL, [1, aDL.shape[0], aDL.shape[1]])
			prediction, c = sess.run([predicted_disparity, cost], feed_dict={left_input: left_img,
														right_input: right_img,
														true_disparity: disparity_gt,
														isTraining: False})
			if c < best_val_data_cost:
				best_val_data_cost = c
				best_val_data = (aIL, aIR, aDL, np.squeeze(prediction))
			val_cost += (c - val_cost) / float(it+1)
			absDiff = np.abs(prediction - disparity_gt)
			val_acc_1 += (np.mean(absDiff < 1) - val_acc_1) / float(it+1)
			val_acc_5 += (np.mean(absDiff < 5) - val_acc_5) / float(it+1)
			update_console("Test Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + 
				"\tProgress: " + str(float(it+1)/float(num_eval_images_per_epoch)) +
				"\tLoss: " + str(val_cost) + "\tAcc 1: " + str(val_acc_1) + "\tAcc 5: " + str(val_acc_5))
		end_console()

		if best_train_data is not None:
			cv2.imwrite("./models_new/train_l.png", (255.0*(best_train_data[0]*0.5+0.5)).astype(np.uint8))
			cv2.imwrite("./models_new/train_r.png", (255.0*(best_train_data[1]*0.5+0.5)).astype(np.uint8))
			cv2.imwrite("./models_new/train_disp.png", (255.0*(best_train_data[2]/float(num_disparities))).astype(np.uint8))
			cv2.imwrite("./models_new/train_predict.png", (255.0*(best_train_data[3]/float(num_disparities))).astype(np.uint8))
		if best_val_data is not None:
			cv2.imwrite("./models_new/test_l.png", (255.0*(best_val_data[0]*0.5+0.5)).astype(np.uint8))
			cv2.imwrite("./models_new/test_r.png", (255.0*(best_val_data[1]*0.5+0.5)).astype(np.uint8))
			cv2.imwrite("./models_new/test_disp.png", (255.0*(best_val_data[2]/float(num_disparities))).astype(np.uint8))
			cv2.imwrite("./models_new/test_predict.png", (255.0*(best_val_data[3]/float(num_disparities))).astype(np.uint8))

		# save the new model (if needed)
		if val_cost < best_val_cost:
			best_val_cost = val_cost
			save_path = saver.save(sess, model_path)
			print("Model saved in file: %s" % save_path)

		# save the last model
		save_path = saver.save(sess, last_model_path)

		iters_till_next_save -= iterations_per_epoch
		iteration = epoch * iterations_per_epoch
		if iters_till_next_save < 0 or epoch == num_epochs-1:
			iters_till_next_save = force_save_number
			save_fn = model_base_path + str(iteration) + ".tfmodel"
			save_path = saver.save(sess, save_fn)
			print("Model saved in file: %s" % save_path)
		
		print("Epoch: " + str(epoch + 1) + "/" + str(num_epochs) + "\t Train Cost: " + str(avg_cost) \
			 + "\t Validation Cost: " + str(val_cost))