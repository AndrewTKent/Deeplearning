from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d
from random import randint

import tensorflow as tf
import numpy as np
import datetime


class Model(tf.keras.Model):
	def __init__(self):
        
		super(Model, self).__init__()

		self.batch_size = 200
		self.num_classes = 2
        
		# TODO: Initialize all hyperparameters
		self.learning_rate = 0.0075
		self.relu_alpha = 0.1
		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

		self.epoch = 10
		self.flatten_width = 1024 
		self.batch_normalization_varience = 0.001
		self.batch_moments = [0, 1, 2]
		self.dropout_rate = [0.1, 0.1, 0.1] 

		self.layer1_filter_num = 32
		self.layer1_stride_size = 1
		self.layer1_pool_ksize = 2
		self.layer1_pool_stride = 2
		self.dense1_output_width = 64
		self.dense1_dropout_rate = self.dropout_rate[0]
		self.filter1_stddev = 0.1

		self.layer2_filter_num = 128
		self.layer2_stride_size = 1
		self.layer2_pool_ksize = 2
		self.layer2_pool_stride = 2
		self.dense2_output_width = 32
		self.dense2_dropout_rate = self.dropout_rate[1]
		self.filter2_stddev = 0.1
                
		self.layer3_filter_num = 200
		self.layer3_stride_size = 1
		self.layer3_pool_ksize = 2
		self.layer3_pool_stride = 2
		self.dense3_output_width = 16
		self.dense3_dropout_rate = self.dropout_rate[2]    
		self.filter3_stddev = 0.1

		self.layer4_filter_num = 256
		self.layer4_stride_size = 1
		self.layer4_pool_ksize = 2
		self.layer4_pool_stride = 2   
		self.filter4_stddev = 0.1
        

		# TODO: Initialize all trainable parameters
		self.filter1 = tf.Variable(tf.random.truncated_normal([3, 3, 3, self.layer1_filter_num], stddev = self.filter1_stddev))
		self.stride1 = [1, self.layer1_stride_size, self.layer1_stride_size, 1]
        
		self.filter2 = tf.Variable(tf.random.truncated_normal([3, 3, self.layer1_filter_num, self.layer2_filter_num], stddev = self.filter2_stddev))
		self.stride2 = [1, self.layer2_stride_size, self.layer2_stride_size, 1]
        
		self.filter3 = tf.Variable(tf.random.truncated_normal([3, 3, self.layer2_filter_num, self.layer3_filter_num], stddev = self.filter3_stddev))
		self.stride3 = [1, self.layer3_stride_size, self.layer3_stride_size, 1]
        
		self.filter4 = tf.Variable(	tf.random.truncated_normal([3, 3, self.layer3_filter_num, self.layer4_filter_num], stddev = self.filter4_stddev))
		self.stride4 = [1, self.layer4_stride_size, self.layer4_stride_size, 1]

		self.w1 = tf.Variable(tf.random.truncated_normal([self.flatten_width, self.dense1_output_width], stddev=.1, dtype=tf.float32))
		self.w2 = tf.Variable(tf.random.truncated_normal([self.dense1_output_width, self.dense2_output_width], stddev=.1, dtype=tf.float32))
		self.w3 = tf.Variable(tf.random.truncated_normal([self.dense2_output_width, self.dense3_output_width], stddev=.1, dtype=tf.float32))
		self.w4 = tf.Variable(tf.random.truncated_normal([self.dense3_output_width, self.num_classes], stddev=.1, dtype=tf.float32))
        
		self.b1 = tf.Variable(tf.random.truncated_normal([1, self.dense1_output_width], stddev=.1, dtype=tf.float32))
		self.b2 = tf.Variable(tf.random.truncated_normal([1, self.dense2_output_width], stddev=.1, dtype=tf.float32))
		self.b3 = tf.Variable(tf.random.truncated_normal([1, self.dense3_output_width], stddev=.1, dtype=tf.float32))
		self.b4 = tf.Variable(tf.random.truncated_normal([1, self.num_classes], stddev=.1, dtype=tf.float32))


	def call(self, inputs, is_testing=False):
        
		# Filter 1
		layer1_convolution = tf.nn.conv2d(inputs, self.filter1, self.stride1, 'SAME')
		mean1, variance1 = tf.nn.moments(layer1_convolution, axes = self.batch_moments)
		layer1_norm = tf.nn.batch_normalization(layer1_convolution, mean1, variance1, offset=None, scale=None, variance_epsilon = self.batch_normalization_varience)
		layer1_relu = tf.nn.leaky_relu(layer1_norm, alpha = self.relu_alpha) 
		layer1_elu = tf.nn.elu(layer1_norm) 
		layer1_pool = tf.nn.max_pool(layer1_elu, self.layer1_pool_ksize, self.layer1_pool_stride, 'SAME')

		# Filter 2
		layer2_convolution = tf.nn.conv2d(layer1_pool, self.filter2, self.stride2, 'SAME')
		mean2, variance2 = tf.nn.moments(layer2_convolution, axes = self.batch_moments)
		layer2_norm = tf.nn.batch_normalization(layer2_convolution, mean2, variance2, offset=None, scale=None, variance_epsilon = self.batch_normalization_varience)
		layer2_relu = tf.nn.leaky_relu(layer2_norm, alpha = self.relu_alpha) 
		layer2_elu = tf.nn.elu(layer2_norm) 
		layer2_pool = tf.nn.max_pool(layer2_elu, self.layer2_pool_ksize, self.layer2_pool_stride, 'SAME')
        
		# Filter 3
		layer3_convolution = tf.nn.conv2d(layer2_pool, self.filter3, self.stride3, 'SAME')
		mean3, variance3 = tf.nn.moments(layer3_convolution, axes = self.batch_moments)
		layer3_norm = tf.nn.batch_normalization(layer3_convolution, mean3, variance3, offset=None, scale=None, variance_epsilon = self.batch_normalization_varience)
		layer3_relu = tf.nn.leaky_relu(layer3_norm, alpha = self.relu_alpha) 
		layer3_elu = tf.nn.elu(layer3_norm) 
		layer3_pool = tf.nn.max_pool(layer3_elu, self.layer3_pool_ksize, self.layer3_pool_stride, 'SAME')

		# Filter 4
		layer4_convolution = tf.nn.conv2d(layer3_pool, self.filter4, self.stride4, 'SAME')
		#layer4_convolution = conv2d(layer3_pool, self.filter4, self.stride4, 'SAME')
		mean4, variance4 = tf.nn.moments(layer4_convolution, axes = self.batch_moments)
		layer4_norm = tf.nn.batch_normalization(layer4_convolution, mean4, variance4, offset=None, scale=None, variance_epsilon = self.batch_normalization_varience)
		layer4_relu = tf.nn.leaky_relu(layer4_norm, alpha = self.relu_alpha) 
		layer4_elu = tf.nn.elu(layer4_norm) 
		layer4_pool = tf.nn.max_pool(layer4_elu, self.layer4_pool_ksize, self.layer4_pool_stride, 'SAME')

        # Flattened Fully Connected (FC) Input
		dense_input = tf.reshape(layer4_pool, [-1, self.flatten_width])

		# FC Layer 1
		dense_layer1 = tf.matmul(dense_input, self.w1) + self.b1
		dense_layer1_relu = tf.nn.leaky_relu(dense_layer1, alpha = self.relu_alpha)
		dense_layer1_elu = tf.nn.elu(dense_layer1) 
		dense_layer1_dropout = tf.nn.dropout(dense_layer1_relu, rate = self.dense1_dropout_rate)

		# FC Layer 2
		dense_layer2 = tf.matmul(dense_layer1_dropout, self.w2) + self.b2
		dense_layer2_relu = tf.nn.leaky_relu(dense_layer2, alpha = self.relu_alpha)
		dense_layer2_elu = tf.nn.elu(dense_layer2) 
		dense_layer2_dropout = tf.nn.dropout(dense_layer2_relu, rate = self.dense2_dropout_rate)

		# FC Layer 3
		dense_layer3 = tf.matmul(dense_layer2_dropout, self.w3) + self.b3   
		dense_layer3_relu = tf.nn.leaky_relu(dense_layer3, alpha = self.relu_alpha)
		dense_layer3_elu = tf.nn.elu(dense_layer3) 
		dense_layer3_dropout = tf.nn.dropout(dense_layer3_relu, rate = self.dense3_dropout_rate)

		# FC Layer 4
		dense_layer4 = tf.matmul(dense_layer3_dropout, self.w4) + self.b4
		dense_layer4_relu = tf.nn.leaky_relu(dense_layer4, alpha = self.relu_alpha)
		dense_layer4_elu = tf.nn.elu(dense_layer3) 

		logits = dense_layer4_relu
        
		return logits


	def loss(self, logits, labels):
        
		loss  = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        
		return loss



	def accuracy(self, logits, labels):
        
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        
		return accuracy



def train(model, train_inputs, train_labels):
    
	# Shuffle Batches
	indices = tf.range(0, train_inputs.shape[0])
	indices = tf.random.shuffle(indices)
	train_inputs = tf.gather(train_inputs, indices)
	train_labels = tf.gather(train_labels, indices)
	batch_num = int(train_inputs.shape[0]/model.batch_size)

	for batch in range(batch_num):
        
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
        
		if (batch + 1) * model.batch_size > train_inputs.shape[0]:  
        
			end = train_inputs.shape[0]
            
		inputs = tf.image.random_flip_left_right(train_inputs[start: end])  
		labels = train_labels[start: end]


		with tf.GradientTape() as tape:
            
			logits = model.call(inputs)
			loss = model.loss(logits, labels)
			train_acc = model.accuracy(logits, labels)

			if train_acc >= .80:
				#model.learning_rate = 0.01
				#model.relu_alpha = 0.01
				print("Learning Rate is" , model.learning_rate)
                
			if train_acc >= .90:
				#model.learning_rate = 0.001
				print("Learning Rate is" , model.learning_rate)

			if model.batch_size * batch % 1000 == 0: 
                
				train_acc = model.accuracy(logits, labels)
				image_num = model.batch_size * batch
                
				print("Accuracy on training set after {} images: {}".format(image_num, train_acc))

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test(model, test_inputs, test_labels):
    
	test_logits = model.call(test_inputs, is_testing = True)
	test_accuracy = model.accuracy(test_logits, test_labels)
    
	return test_accuracy



def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	NOTE: DO NOT EDIT
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


def main():
    
	# Print Current Time
	start_time = datetime.datetime.utcnow()
	print('\nStart Time = ', start_time)
    
	# Load in Test and Train Data from Data Folder
	cat_class = 3
	dog_class = 5
	test_inputs, test_labels = get_data('data/test', cat_class, dog_class)
	train_inputs, train_labels = get_data('data/train', cat_class, dog_class)

	# Initiate the Model
	model = Model()

	# Initie Training. Announce Accuracy and Epoch Values
	for epoch in range(0, model.epoch):
            
		print("\n       -------------     EPOCH {}     -------------       ".format(epoch))
		train(model, train_inputs, train_labels)
	print("\n   -------------     ALL EPOCHS END     -------------    \n")

	test_accuracy = test(model, test_inputs, test_labels)
	print("\nAccuracy on test set is: {}".format(test_accuracy), '\n')

	# Visualize 10 Random Images
	num_test = test_inputs.shape[0]
	random_num = randint(0, num_test - 10)
	sample_inputs = test_inputs[random_num : 10 + random_num]
	sample_labels = test_labels[random_num : 10 + random_num]
	sample_logits = model.call(sample_inputs, sample_labels)
	visualize_results(sample_inputs, sample_logits, sample_labels, 'cat', 'dog')
    
	# Print Current Time and Other Parameters
	end_time = datetime.datetime.utcnow()
	print('End Time = ', end_time)
	print('\nTime Difference = ', end_time - start_time)
	print('\nLearning Rate = ', model.learning_rate)
	print('\nRelu Alpha = ', model.relu_alpha)


if __name__ == '__main__':
	main()
    
