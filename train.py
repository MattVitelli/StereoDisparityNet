from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dataset_helper
import cv2
import math
import model
import numpy as np

# model I/O settings
model_path = "./models/best_model.tfmodel"
model_base_path = "./models/disparity_"
force_save_number = 10000

# training settings
num_iterations = 150000
iterations_per_epoch = 500
num_epochs = math.ceil(float(num_iterations) / float(iterations_per_epoch))
num_eval_images_per_epoch = 50

# learning settings
learning_rate = 1.0e-3

# model settings
num_features = 32
num_disparities = 192
width = 512
height = 256
channels = 3

# tf Graph input
left_input = tf.placeholder(tf.float32, shape=[1, height, width, channels])
right_input = tf.placeholder(tf.float32, shape=[1, height, width, channels])
true_disparity = tf.placeholder(tf.float32, shape=[1, height, width])
isTraining = tf.placeholder(tf.bool, shape=[])

predicted_disparity = model.make_disparity_model(left_input, right_input, num_features, num_disparities, isTraining)

cost = tf.reduce_mean(tf.abs(true_disparity[0] - predicted_disparity[0]))
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

	# Training cycle
	for epoch in range(num_epochs):
		print("Starting epoch " + str(epoch + 1) + "/" + str(num_epochs))
		# train the model on a small batch
		avg_cost = 0.
		for it in range(iterations_per_epoch):
			print("Iteration: " + str(it+1) + "/" + str(iterations_per_epoch))
			aIL, aIR, aDL, aDR = sceneflow.sample_from_training_set(width=width,height=height)
			left_img = np.reshape(aIL, [1, aIL.shape[0], aIL.shape[1], aIL.shape[2]])
			right_img = np.reshape(aIR, [1, aIR.shape[0], aIR.shape[1], aIR.shape[2]])
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict={left_input: left_img,
														right_input: right_img,
														true_disparity: [aDL],
														isTraining: True})
			avg_cost += c / float(iterations_per_epoch)

		# test the model on the validation set
		print("Validating on validation set...")
		val_cost = 0
		best_tmp_cost = float('Inf')
		bestIL = None
		bestIR = None
		bestDL = None
		bestP = None
		for it in range(num_eval_images_per_epoch):
			aIL, aIR, aDL, aDR = sceneflow.sample_from_test_set(width=width,height=height)
			left_img = np.reshape(aIL, [1, aIL.shape[0], aIL.shape[1], aIL.shape[2]])
			right_img = np.reshape(aIR, [1, aIR.shape[0], aIR.shape[1], aIR.shape[2]])
			prediction, c = sess.run([predicted_disparity[0], cost], feed_dict={left_input: left_img,
														right_input: right_img,
														true_disparity: [aDL],
														isTraining: False})
			tmp_cost = c / float(num_eval_images_per_epoch)
			if tmp_cost < best_tmp_cost:
				best_tmp_cost = tmp_cost
				bestIL = aIL
				bestIR = aIR
				bestDL = aDL
				bestP = np.squeeze(prediction)
			val_cost += tmp_cost

		if bestIL is not None:
			cv2.imwrite("./result_l.png", (255.0*(bestIL*0.5+0.5)).astype(np.uint8))
			cv2.imwrite("./result_r.png", (255.0*(bestIR*0.5+0.5)).astype(np.uint8))
			cv2.imwrite("./result_disp.png", (255.0*(bestDL/float(num_disparities))).astype(np.uint8))
			cv2.imwrite("./result_predict.png", (255.0*(bestP/float(num_disparities))).astype(np.uint8))

		# save the new model (if needed)
		if val_cost < best_val_cost:
			save_path = saver.save(sess, model_path)
			print("Model saved in file: %s" % save_path)

		iteration = epoch * iterations_per_epoch
		if (iteration % force_save_number) == 0 or iteration == num_iterations:
			save_fn = model_base_path + str(iteration) + ".tfmodel"
			save_path = saver.save(sess, save_fn)
			print("Model saved in file: %s" % save_path)
		
		print("Epoch: " + str(epoch) + "/" + str(num_epochs) + "\t Train Cost: " + str(avg_cost) \
			 + "\t Validation Cost: " + str(val_cost))