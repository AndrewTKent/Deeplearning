from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
from attenvis import AttentionVis

from preprocess import *

import tensorflow as tf
import numpy as np
import datetime
import sys

av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index):

	print('Training starting: \n')
    
	number_of_batches = train_french.shape[0] // model.batch_size
    
	for batch in range(number_of_batches):
        
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
        
		if (batch + 1) * model.batch_size > train_french.shape[0]: 
            
			end = train_french.shape[0]
            
		encoder_input = train_french[start: end, :]
        
		decoder_input = train_english[start: end, 0: ENGLISH_WINDOW_SIZE]
		decoder_label = train_english[start: end, 1: ENGLISH_WINDOW_SIZE + 1]
        
		mask = (decoder_label != eng_padding_index)

		with tf.GradientTape() as tape:
            
			prbs = model.call(encoder_input, decoder_input)
			loss = model.loss_function(prbs, decoder_label, mask)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		print('\r', 'Training Completion Percentage: {0:.4f} %'.format((batch + 1) * 100 / number_of_batches), end='')


@av.test_func
def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""
    
	print('\nTest starts:')
	number_of_batches = test_french.shape[0] // model.batch_size
	sum_loss = 0
	sum_true = 0
	sum_symbol = 0
    
	for batch in range(number_of_batches):
        
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
        
		if (batch + 1) * model.batch_size > test_french.shape[0]:
            
			end = test_french.shape[0]
            
		encoder_input = test_french[start: end, :]
		decoder_input = test_english[start: end, 0: ENGLISH_WINDOW_SIZE]
		decoder_label = test_english[start: end, 1: ENGLISH_WINDOW_SIZE + 1]
        
		mask = (decoder_label != eng_padding_index)

		prbs = model.call(encoder_input, decoder_input)
		loss = tf.reduce_mean(model.loss_function(prbs, decoder_label, mask)) * (end - start)
        
		sum_loss += loss
		batch_acc = model.accuracy_function(prbs, decoder_label, mask)
		batch_symbol = np.sum(tf.cast(mask, dtype=tf.float32))
        
		sum_symbol += batch_symbol
		sum_true += batch_acc * batch_symbol
        
		print('\r', 'testing process: {0:.1f} %'.format((batch + 1) * 100 / number_of_batches), end='')

	perplexity = np.exp(sum_loss / test_english.shape[0])
	accuracy = sum_true / sum_symbol

	return perplexity, accuracy

def main():	

	start_time = datetime.datetime.utcnow() 
	print('\nStart Time: {}\n'.format(start_time))
	    
	sys.argv = ["", "RNN"]

	
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
        
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
            
			exit()

	if sys.argv[1] == "TRANSFORMER":
        
		av.setup_visualization(enable=True)
    
	train_english, test_english, train_french, test_french, english_vocab, french_vocab, english_padding_index = get_data('data/fls.txt','data/els.txt','data/flt.txt','data/elt.txt')
	model_arguments = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))
    
	if sys.argv[1] == "RNN":
        
		model = RNN_Seq2Seq(*model_arguments)
        
	elif sys.argv[1] == "TRANSFORMER":
        
		model = Transformer_Seq2Seq(*model_arguments) 
	
	# Train and Test Model
	train(model, train_french, train_english, english_padding_index)
	perplexity, accuracy = test(model, test_french, test_english, english_padding_index)
    
	print('\n\nPerplexity = {0:.4f}'.format(perplexity))
	print('Accuracy = {0:.4f}'.format(accuracy))

	av.show_atten_heatmap()
    
	end_time = datetime.datetime.utcnow()
	run_time = end_time - start_time
	print('\nEnd Time: {}'.format(end_time))
	print('\nRun Time: {}'.format(run_time))  

if __name__ == '__main__':
	main()
