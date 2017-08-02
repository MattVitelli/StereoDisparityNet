import numpy as np
import tensorflow as tf


m = np.random.rand(192,128,128,4)
#m = np.array([[1, 2, 3, 4], [5, 6, 7, 8]],dtype=np.float32)
mx, my, mz, mw = np.meshgrid(range(m.shape[0]),
		range(m.shape[1]),
		range(m.shape[2]),
		range(m.shape[3]), indexing="ij")

mz = np.minimum(m.shape[2]-1, mz+mx)


x = tf.placeholder('float64', (None, None, None, None))
idx1 = tf.placeholder('int32', (None, None, None, None))
idx2 = tf.placeholder('int32', (None, None, None, None))
idx3 = tf.placeholder('int32', (None, None, None, None))
idx4 = tf.placeholder('int32', (None, None, None, None))
result = tf.gather_nd(x, tf.stack((idx1, idx2, idx3, idx4), -1))

with tf.Session() as sess:
	r = sess.run(result, feed_dict={
		x: m,
		idx1: mx,
		idx2: my,
		idx3: mz,
		idx4: mw
	})
	rmat = m[mx,my,mz,mw]
	#print(rmat)
	#print(r)
	error = np.sum(np.sum(np.sum(np.sum(np.abs(rmat-r)))))
	print("Error: " + str(error))