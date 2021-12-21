import pickle
import numpy as np
import tensorflow as tf

def unpickle(file):
    
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(file_path, first_class, second_class):
    
	unpickled_file = unpickle(file_path)
	inputs = np.array(unpickled_file[b'data'], dtype=tf.uint8)
	labels = np.array(unpickled_file[b'labels'], dtype=tf.uint8)

	# TODO: Do the rest of preprocessing!
	indices = np.where((labels == first_class) | (labels == second_class))
    
	inputs = inputs[indices]
	inputs = np.reshape(inputs, (-1, 3, 32, 32))
	inputs = np.transpose(inputs,(0, 2, 3, 1))  
	inputs = (inputs / 255).astype('float64')

	labels = labels[indices]
	labels = (labels == second_class).astype('float64')
	labels = tf.one_hot(labels, depth=2, dtype=tf.uint8)


	return inputs, labels
