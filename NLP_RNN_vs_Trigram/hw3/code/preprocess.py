import tensorflow as tf
import numpy as np
from functools import reduce

def get_data(train_file, test_file):

    # Loads and Concatenate Training Data from Training File
    with open(train_file, 'r') as f:
        train_data = f.read().split()

    # Load and Concatenates Testing Data from Testing File
    with open(test_file, 'r') as f:
        test_data = f.read().split()

    # Makes Vocab Dictionary
    vocab = set(train_data + test_data)
    vocab_dict = {j: i for i, j in enumerate(vocab)}

    # Reads in and Tokenizes Training Data
    train_token = []
    
    for s in train_data:
        
        train_token.append(vocab_dict.get(s))
        
    train_token = np.array(train_token)
    train_token = train_token.astype(dtype=np.int32)

    # Reads in and Tokenizes Testing Data
    test_token = []
    
    for s in test_data:
        
        test_token.append(vocab_dict.get(s))
        
    test_token = np.array(test_token)
    test_token = test_token.astype(dtype=np.int32)

    # Returns Tuple of Training Tokens, Testing Tokens, and the Vocab Dictionary
    return train_token, test_token, vocab_dict