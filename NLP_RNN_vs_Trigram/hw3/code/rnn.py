from preprocess import get_data

import tensorflow as tf
import numpy as np
import datetime


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        
        super(Model, self).__init__()

        # Initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.learning_rate_adam = 0.01 
        self.embedding_size = 300
        self.window_size = 20
        self.batch_size = 300
        self.rnn_size = 256  
        self.leaky_alpha = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.learning_rate_adam)

        # Initialize Embeddings and Forward Pass Weights (weights, biases)
        self.E = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.1))
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
         
        embedding = tf.nn.embedding_lookup(self.E, inputs)  
        output, state1, state2 = self.lstm(embedding, initial_state)
        dense = self.dense(output)

        return dense, (state1, state2)

    def loss(self, probs, labels):
        
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        
        return loss

def train(model, train_inputs, train_labels):
    
    initial_state = None
    
    indices = tf.range(0, train_inputs.shape[0])
    indices = tf.random.shuffle(indices)
    
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    train_batch_num = train_inputs.shape[0] // model.batch_size
    
    for batch in range(train_batch_num):
        
        first = batch * model.batch_size
        last = (batch + 1) * model.batch_size
        
        if (batch + 1) * model.batch_size > train_inputs.shape[0]:
            
            last = train_inputs.shape[0]
            
        inputs = train_inputs[first: last]
        labels = train_labels[first: last]

        with tf.GradientTape() as tape:
            
            probs, _ = model.call(inputs, initial_state)
            loss = model.loss(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_percent_complete = (batch + 1) * 100 // train_batch_num

        print('\r', "Training: {} %".format(train_percent_complete), end = '')

    print('\n\nTraining Complete')


def test(model, test_inputs, test_labels):
    
    print('\nTest starts: \n')
    
    initial_state = None
    
    indices = tf.range(0, test_inputs.shape[0])
    indices = tf.random.shuffle(indices)
    
    test_inputs = tf.gather(test_inputs, indices)
    test_labels = tf.gather(test_labels, indices)

    test_batch_num = test_inputs.shape[0] // model.batch_size

    loss = 0

    for batch in range(test_batch_num):
        
        first = batch * model.batch_size
        last = (batch + 1) * model.batch_size
        
        if (batch + 1) * model.batch_size > test_inputs.shape[0]:
            
            last = test_inputs.shape[0]
            
        inputs = test_inputs[first: last]
        labels = test_labels[first: last]

        probs, _ = model.call(inputs, initial_state)
        loss += model.loss(probs, labels)
        
        test_percent_complete = (batch + 1) * 100 // test_batch_num
        
        print('\r', "Testing: {} %".format(test_percent_complete), end = '')
    
    avg_loss = tf.reduce_mean(loss)/test_batch_num
    
    print('\n\nTesting Complete\n')
    
    return np.exp(avg_loss)


def generate_sentence(word1, length, vocab, model):
    
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    
    start_time = datetime.datetime.utcnow() 
    print('\nStart Time: {}\n'.format(start_time))   
    
    # Pre-process and vectorize the data
    train_token, test_token, vocab_dict = get_data('data/train.txt', 'data/test.txt')
    
    train_num = train_token.shape[0]
    test_num = test_token.shape[0]

    # Initialize Model and Tensorflow Variables
    vocab_size = 7342
    model = Model(vocab_size)

    # Separate your train and test data into inputs and labels
    train_num = (train_num - 1) // model.window_size
    test_num = (test_num - 1) // model.window_size
    
    train_inputs = np.zeros((train_num, model.window_size), dtype=np.int32)
    train_labels = np.zeros((train_num, model.window_size), dtype=np.int32)
    
    test_inputs = np.zeros((test_num, model.window_size), dtype=np.int32)
    test_labels = np.zeros((test_num, model.window_size), dtype=np.int32)
    
    for i in range(train_num):
        train_inputs[i] = train_token[i * model.window_size: (i + 1) * model.window_size]
        train_labels[i] = train_token[i * model.window_size + 1: (i + 1) * model.window_size + 1]

    for i in range(test_num):
        test_inputs[i] = test_token[i * model.window_size: (i + 1) * model.window_size]
        test_labels[i] = test_token[i * model.window_size + 1: (i + 1) * model.window_size + 1]

    # Set-up the training step
    train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps
    Perplexity = test(model, test_inputs, test_labels)
    
    print('Perplexity = {}'.format(Perplexity))

    print('\nFirst Generated Sentence: Donald')
    generate_sentence('donald', 10, vocab_dict, model)
    
    print('\nSecond Generated Sentence: computer')
    generate_sentence('computer', 10, vocab_dict, model)
    
    print('\nThird Generated Sentence: how')
    generate_sentence('how', 10, vocab_dict, model)
    
    end_time = datetime.datetime.utcnow()
    run_time = end_time - start_time
    print('\nEnd Time: {}'.format(end_time))
    print('\nRun Time: {}'.format(run_time))  
    
    pass
    
if __name__ == '__main__':
    main()
    
    
    
    
    