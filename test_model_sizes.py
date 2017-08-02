import tensorflow as tf
#import model
import model_new

num_features = 32
num_disparities = 192
width = 512
height = 256
channels = 3
imgL = tf.Variable(tf.constant(0.0, shape=[1, channels, height, width]))
imgR = tf.Variable(tf.constant(0.0, shape=[1, channels, height, width]))
training = tf.Variable(True)
#result = model.make_unitary_model([imgL, imgR], 3, num_features, training)
result = model_new.make_disparity_model(imgL, imgR, num_features, num_disparities, training)