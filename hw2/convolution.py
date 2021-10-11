from __future__ import absolute_import

import tensorflow as tf
import numpy as np

def conv2d(inputs, filters, strides, padding):
	if type(inputs) is not np.ndarray:
		inputs = inputs.numpy()
        
	if type(filters) is not np.ndarray:
		filters = filters.numpy()
        
	num_examples = inputs.shape[0]
	input_height = inputs.shape[1]
	input_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	stride_vertical = strides[1]
	stride_horizontal = strides[2]

	assert input_in_channels == filter_in_channels

	# Padding Input
	if padding == 'SAME':
        
		padding_width = (filter_width - 1) // 2
		padding_height = (filter_height - 1) // 2

	else:
        
		padding_height = 0
		padding_width = 0

	inputs = np.pad(inputs, ((0, 0), (padding_height, padding_width), (padding_height, padding_width), (0, 0)), 'constant')
    
	# Calculate Output Dimensions
	output_height = (input_height - filter_height + padding_height*2) // stride_vertical + 1
	output_width = (input_width - filter_width + padding_width*2) // stride_horizontal + 1
	output_channels = filter_out_channels
	outputs = np.zeros((num_examples, output_height, output_width, output_channels))

	for i in range(0, num_examples):
        
		if (i % 200 == 0):
                 print('Test Image ',i, ' of 2000')
        
		for j in range(0, output_height):
            
			for u in range(0, output_width):
                
				for v in range(0, output_channels):
                    
					tmp = inputs[i, j: filter_height + j , u: filter_width + u , :]
					kernel = filters[:, :, :, v]
					outputs[i, j, u, v] = np.tensordot(tmp, kernel, ((0, 1, 2), (0, 1, 2)))
                    

	return tf.convert_to_tensor(outputs, tf.float32)


def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())


def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.
	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())


def main():
	# TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
	same_test_0()
	valid_test_0()
	valid_test_1()
	valid_test_2()

if __name__ == '__main__':
	main()