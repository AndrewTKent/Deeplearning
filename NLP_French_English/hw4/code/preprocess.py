import numpy as np

from attenvis import AttentionVis
av = AttentionVis()

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 14
##########DO NOT CHANGE#####################

def pad_corpus(french, english):
    
	french_padded_sentence = []
	english_padded_sentence = []
    
	for line in french:
        
		padded_french = line[:FRENCH_WINDOW_SIZE]
		padded_french += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_french)-1)
        
		french_padded_sentence.append(padded_french)
    
	for line in english:
        
		padded_english = line[:ENGLISH_WINDOW_SIZE]
		padded_english = [START_TOKEN] + padded_english + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_english)-1)
		english_padded_sentence.append(padded_english)

	return french_padded_sentence, english_padded_sentence

def build_vocab(sentences):
	"""
	DO NOT CHANGE

  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
    
	for s in sentences: 
        
		tokens.extend(s)
    
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab, vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE

  Convert sentences to indexed 

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE

  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text

@av.get_data_func
def get_data(french_training_file, english_training_file, french_test_file, english_test_file):
	
	# Read English and French Data for Training and Testing
	french_train = read_data(french_training_file)
	english_train = read_data(english_training_file)
    
	french_test = read_data(french_test_file)
	english_test = read_data(english_test_file)
    
	# Pad Training Data
	french_padded_train, english_padded_train = pad_corpus(french_train, english_train)

	# Pad Testing Data
	french_padded_test, english_padded_test = pad_corpus(french_test, english_test)

	# Build vocab
	french_vocab, french_pad_index = build_vocab(french_padded_train)
	english_vocab, english_pad_index = build_vocab(english_padded_train)

	# Convert Training and Testing English Sentences to list of IDS
	train_english = convert_to_id(english_vocab, english_padded_train)
	test_english = convert_to_id(english_vocab, english_padded_test)

	# Convert Training and Testing French Sentences to list of IDS
	train_french = convert_to_id(french_vocab, french_padded_train)
	test_french = convert_to_id(french_vocab, french_padded_test)

	return train_english, test_english, train_french, test_french, english_vocab, french_vocab, english_pad_index
	