import tensorflow as tf

def conv2d(x, filter_size, stride, num_channels, num_features):
	W = tf.Variable(tf.random_normal([filter_size, filter_size, num_channels, num_features]))
	b = tf.Variable(tf.random_normal([num_features]))
	x_new = []
	for x_idx in range(len(x)):
		x_p = tf.nn.conv2d(x[x_idx], W, strides=[1, stride, stride, 1], padding='SAME',
			data_format='NHWC')
		x_p = tf.nn.bias_add(x_p, b, data_format='NHWC')
		x_new.append(x_p)
	return x_new

def conv2d_block(x, filter_size, stride, num_channels, num_features, training_scope):
	x = conv2d(x, filter_size, stride, num_channels, num_features)
	#BN
	x_new = []
	for x_idx in range(len(x)):
		x_p = tf.layers.batch_normalization(x[x_idx], center=True, scale=True,
			training=training_scope, axis=-1)
		x_p = tf.nn.relu(x_p)
		x_new.append(x_p)
	return x_new

def residual2d(x, old_x):
	x_new = []
	for x_idx in range(len(x)):
		x_new.append(x[x_idx] + old_x[x_idx])
	return x_new

def conv3d(x, filter_size, stride, num_channels, num_features):
	x_p = tf.layers.conv3d(inputs=x,
					filters=num_features,
					kernel_size=filter_size,
					strides=stride,
					padding='SAME',
					data_format='channels_last')
	return x_p

def conv3d_block(x, filter_size, stride, num_channels, num_features, training_scope):
	x_p = conv3d(x, filter_size, stride, num_channels, num_features)
	#BN
	x_p = tf.layers.batch_normalization(x_p,
			center=True, scale=True,
			training=training_scope, axis=-1)
	x_p = tf.nn.relu(x_p)
	return x_p

def conv3d_t(x, filter_size, upscale_factor, stride, num_channels, num_features):
	x_p = tf.layers.conv3d_transpose(inputs=x,
						filters=num_features,
						kernel_size=filter_size,
						strides=stride,
						padding='SAME',
						data_format='channels_last')
	return x_p

def conv3d_t_block(x, filter_size, upscale_factor, stride, num_channels, num_features, training_scope):
	x_p = conv3d_t(x, filter_size, upscale_factor, stride, num_channels, num_features)
	#BN
	x_p = tf.layers.batch_normalization(x_p,
			center=True, scale=True,
			training=training_scope, axis=-1)
	x_p = tf.nn.relu(x_p)
	return x_p

def residual3d(x, old_x):
	return x + old_x

def soft_argmax(costs):
	costs = tf.squeeze(costs, axis=-1)
	print(costs.shape)
	softmax = tf.nn.softmax(-costs, dim=1)
	disp_range = tf.range(0, int(costs.shape[1]), dtype=tf.float32)
	disp_range_sizes = [1, int(costs.shape[1]), 1, 1]
	height = int(costs.shape[2])
	width = int(costs.shape[3])
	disp_range = tf.reshape(disp_range, disp_range_sizes)
	disp_range_grid = tf.tile(disp_range, [1,1,height,width])
	soft_arg_max = tf.multiply(softmax,disp_range_grid)
	disparity = tf.reduce_sum(soft_arg_max, axis=1)
	return disparity

def make_unitary_model(x, num_channels, num_features, training_scope):
	print(x[0].shape)
	c0 = conv2d_block(x, 5, 2, num_channels, num_features, training_scope)
	print(c0[0].shape)
	c1 = conv2d_block(c0, 3, 1, num_features, num_features, training_scope)
	print(c1[0].shape)
	c2 = residual2d(conv2d_block(c1, 3, 1, num_features, num_features, training_scope), c0)
	print(c2[0].shape)
	for it in range(7):
		c1 = conv2d_block(c2, 3, 1, num_features, num_features, training_scope)
		print(c1[0].shape)
		c2 = residual2d(conv2d_block(c1, 3, 1, num_features, num_features, training_scope), c2)
		print(c2[0].shape)
	cFinal = conv2d(c2, 3, 1, num_features, num_features)
	print("Final Unary: " + str(cFinal[0].shape))
	return cFinal

def make_cost_volume_model(x, num_disparities):
	max_d = int((num_disparities + 1) / 2)
	width = int(x[0].shape[3])
	sizes = [int(x[0].shape[0]), 1, int(x[0].shape[1]), int(x[0].shape[2]), int(x[0].shape[3])]
	reshaped_left = tf.reshape(x[0], sizes)
	reshaped_right = tf.reshape(x[1], sizes)
	Lb, Ld, Ly, Lx, Lf = tf.meshgrid(tf.range(0, sizes[0]),
							tf.range(0, max_d),
							tf.range(0, sizes[2]),
							tf.range(0, sizes[3]),
							tf.range(0, sizes[4]),
							indexing='ij')
	Lx = tf.maximum(0, tf.minimum(width-1, Lx - Ld))

	tiled_left = tf.tile(reshaped_left, [1, max_d, 1, 1, 1])
	tiled_right = tf.tile(reshaped_right, [1, max_d, 1, 1, 1])

	g_right = tf.gather_nd(tiled_right, tf.stack((Lb, Ld, Ly, Lx, Lf), -1))
	cost_volume_left = tf.concat((tiled_left,g_right),axis=-1)
	return cost_volume_left

def make_disparity_model(left_img, right_img, num_features, num_disparities, training_scope):
	unitary_model = make_unitary_model([left_img, right_img], 3, num_features, training_scope)
	cost_volume = make_cost_volume_model(unitary_model, num_disparities)
	print("Cost volume: " + str(cost_volume.shape))
	num_features_x2 = num_features * 2
	num_features_x4 = num_features * 4
	#19
	c0 = conv3d_block(cost_volume, 3, 1, num_features_x2, num_features, training_scope)
	print(c0.shape)
	#20
	c1 = conv3d_block(c0, 3, 1, num_features, num_features, training_scope)
	print(c1.shape)
	#21
	c2 = conv3d_block(cost_volume, 3, 2, num_features_x2, num_features_x2, training_scope)
	print(c2.shape)
	#22
	c3 = conv3d_block(c2, 3, 1, num_features_x2, num_features_x2, training_scope)
	print(c3.shape)
	#23
	c4 = conv3d_block(c3, 3, 1, num_features_x2, num_features_x2, training_scope)
	print(c4.shape)
	#24
	c5 = conv3d_block(c2, 3, 2, num_features_x2, num_features_x2, training_scope)
	print(c5.shape)
	#25
	c6 = conv3d_block(c5, 3, 1, num_features_x2, num_features_x2, training_scope)
	print(c6.shape)
	#26
	c7 = conv3d_block(c6, 3, 1, num_features_x2, num_features_x2, training_scope)
	print(c7.shape)
	#27
	c8 = conv3d_block(c5, 3, 2, num_features_x2, num_features_x2, training_scope)
	print(c8.shape)
	#28
	c9 = conv3d_block(c8, 3, 1, num_features_x2, num_features_x2, training_scope)
	print(c9.shape)
	#29
	c10 = conv3d_block(c9, 3, 1, num_features_x2, num_features_x2, training_scope)
	print(c10.shape)
	#30
	c11 = conv3d_block(c8, 3, 2, num_features_x2, num_features_x4, training_scope)
	print(c11.shape)
	#31
	c12 = conv3d_block(c11, 3, 1, num_features_x4, num_features_x4, training_scope)
	print(c12.shape)
	#32
	c13 = conv3d_block(c12, 3, 1, num_features_x4, num_features_x4, training_scope)
	print(c13.shape)
	#33 - transposed conv
	c14 = residual3d(conv3d_t_block(c13, 3, 2, 2, num_features_x4, num_features_x2, training_scope), c10)
	print(c14.shape)
	#34 - transposed conv
	c15 = residual3d(conv3d_t_block(c14, 3, 2, 2, num_features_x2, num_features_x2, training_scope), c7)
	print(c15.shape)
	#35 - transposed conv
	c16 = residual3d(conv3d_t_block(c15, 3, 2, 2, num_features_x2, num_features_x2, training_scope), c4)
	print(c16.shape)
	#36 - transposed conv
	c17 = residual3d(conv3d_t_block(c16, 3, 2, 2, num_features_x2, num_features, training_scope), c1)
	print(c17.shape)
	costs = conv3d_t(c17, 3, 2, 2, num_features, 1)
	print(costs.shape)
	disparity = soft_argmax(costs)
	print("Final disparity map: " + str(disparity.shape))
	return disparity