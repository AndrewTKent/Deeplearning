import transformer_funcs as transformer
import tensorflow as tf

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################

		# Hyperparameters
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.batch_size = 100
		self.embedding_size = 30

		# Define english and french embedding layers:
		self.Embedding_English = tf.Variable(tf.random.truncated_normal(shape=[self.english_vocab_size, self.embedding_size], mean=0, stddev=0.01))
		self.Embedding_French = tf.Variable(tf.random.truncated_normal(shape=[self.french_vocab_size, self.embedding_size], mean=0, stddev=0.01))        
		
		# Positional Encoder Layers
		self.position_french = transformer.Position_Encoding_Layer(french_window_size, self.embedding_size)
		self.position_english = transformer.Position_Encoding_Layer(english_window_size, self.embedding_size)

		# Encoder and Decoder Layers:
		self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False)
		self.decoder = transformer.Transformer_Block(self.embedding_size, is_decoder=True)
            
		# Dense Layer
		self.dense = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

	@tf.function
	def call(self, encoder_input, decoder_input):
        
		# Positional Embeddings for French/English Sentence Embeddings
		french_embed = self.position_french.call(tf.nn.embedding_lookup(self.Embedding_French, encoder_input))
		english_embed = self.position_english.call(tf.nn.embedding_lookup(self.Embedding_English, decoder_input))
        
		# French/English Sentence Embeddings to the Encoder
		encoder_out = self.encoder(french_embed)
		decoder_out = self.decoder(english_embed, encoder_out)
        
		# Dense Layer to the Decoder Out to Generate Probabilities
		prbs = self.dense(decoder_out)
    
		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
        
		sparse = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		boolean = tf.boolean_mask(sparse, mask)
		reduced_mean = tf.reduce_sum(boolean)	

		return reduced_mean

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)