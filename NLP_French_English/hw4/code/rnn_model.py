import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
		###### DO NOT CHANGE ##############
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
		self.embedding_size = 128
		self.batch_size = 100
		self.rnn_size = 80
        
		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.Embedding_English = tf.Variable(tf.random.truncated_normal(shape=[self.english_vocab_size, self.embedding_size], mean=0, stddev=0.01))
		self.Embedding_French = tf.Variable(tf.random.truncated_normal(shape=[self.french_vocab_size, self.embedding_size], mean=0, stddev=0.01))

		self.encoder = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
		self.decoder = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        
		self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

	@tf.function
	def call(self, encoder_input, decoder_input):
        
		# Pass French Sentence Embeddings to Encoder 
		french_embed = tf.nn.embedding_lookup(self.Embedding_French, encoder_input)
		encoder_out, state1, state2 = self.encoder(french_embed, initial_state=None)
        
		# Pass English Sentence Embeddings, and Final State of Encoder, to Decoder
		english_embed = tf.nn.embedding_lookup(self.Embedding_English, decoder_input)
		decoder_out, state3, state4 = self.decoder(english_embed, initial_state=(state1, state2))
        
		# Apply Dense layer to the Decoder out to Generate Probabilities
		prbs = self.dense(decoder_out)
        
		return prbs

	def accuracy_function(self, prbs, labels, mask):

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        
		return accuracy


	def loss_function(self, prbs, labels, mask):
        
		sparse = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		boolean = tf.boolean_mask(sparse, mask)
		reduced_mean = tf.reduce_mean(boolean)	

		return reduced_mean

