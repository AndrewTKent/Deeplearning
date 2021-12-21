from preprocess import get_data

import tensorflow as tf
import numpy as np
import datetime

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        
        super(Model, self).__init__()

        # Initialize vocab_size, embedding_size and etc.
        self.vocab_size = vocab_size
        self.learning_rate_adam = 0.001 
        self.embedding_size = 256
        self.batch_size = 500
        self.leaky_alpha = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate_adam)

        # Initializes Embeddings and Forward Pass Weights (weights, biases)
        self.E = tf.Variable(tf.random.truncated_normal(shape=[self.vocab_size, self.embedding_size], mean=0, stddev=0.1))
        self.W = tf.Variable(tf.random.truncated_normal(shape=[self.embedding_size * 2, self.vocab_size], mean=0, stddev=0.1))
        self.b = tf.Variable(tf.random.truncated_normal(shape=[1, self.vocab_size], mean=0, stddev=0.1))

    def call(self, inputs):
        
        embedding_1 = tf.nn.embedding_lookup(self.E, inputs[:, 0])
        embedding_2 = tf.nn.embedding_lookup(self.E, inputs[:, 1])
        embedding = tf.concat([embedding_1, embedding_2], 1)
        
        probs = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(embedding, self.W) + self.b, alpha = self.leaky_alpha))

        return probs

    def loss_function(self, probs, labels):
        
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        
        return loss

def train(model, train_inputs, train_labels):
    
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
            
            probs = model.call(inputs)
            loss = model.loss_function(probs, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_percent_complete = (batch + 1) * 100 // train_batch_num

        print('\r', "Training: {} %".format(train_percent_complete), end = '')

    print('\n\nTraining Complete\n')

def test(model, test_inputs, test_labels):
    
    indices = tf.random.shuffle(tf.range(0, test_inputs.shape[0]))
    
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

        probs = model.call(inputs)
        loss += model.loss_function(probs, labels)
        
        test_percent_complete = (batch + 1) * 100 // test_batch_num
        
        print('\r', "Testing: {} %".format(test_percent_complete), end = '')
    
    avg_loss = tf.reduce_mean(loss)/test_batch_num
    
    print('\n\nTesting Complete')
    
    return np.exp(avg_loss)

def generate_sentence(word1, word2, length, vocab, model):
    
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, 0: 2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]
    
    print(" ".join(text))


def main():

    start_time = datetime.datetime.utcnow() 
    print('\nStart Time: {}\n'.format(start_time))
    
    # Pre-process and vectorize the data using get_data from preprocess
    train_token, test_token, vocab_dict = get_data('data/train.txt', 'data/test.txt')
    
    train_num = train_token.shape[0]
    test_num = test_token.shape[0]

    # Separate Train and Test Data into Inputs and Labels
    train_inputs = np.zeros((train_num, 2), dtype=np.int32)
    train_labels = np.zeros((train_num, 1), dtype=np.int32)
    
    test_inputs = np.zeros((test_num, 2), dtype=np.int32)
    test_labels = np.zeros((test_num, 1), dtype=np.int32)
    
    for i in range(train_num - 2):
        train_inputs[i, :] = train_token[i: i + 2]
        train_labels[i] = train_token[i + 2]
        
    for i in range(test_num - 2):
        test_inputs[i, :] = test_token[i: i + 2]
        test_labels[i] = test_token[i + 2]

    # Initialize Model and Tensorflow Variables
    vocab_size = 7342
    model = Model(vocab_size)

    # Set-up the Training Step
    train(model, train_inputs, train_labels)

    # Set up the Testing Steps / Print out perplexity
    perplexity = test(model, test_inputs, test_labels)
    print('\nPerplexity = {}\n'.format(perplexity))

    word1 = 'how'
    word2 = 'many'
    length = 10
    generate_sentence(word1, word2, length, vocab_dict, model)
    
    end_time = datetime.datetime.utcnow()
    print('\nEnd Time = ', end_time)
    print('\nRun Time = ', end_time - start_time)  
    
if __name__ == '__main__':
    main()