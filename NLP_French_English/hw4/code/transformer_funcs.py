import tensorflow as tf
import numpy as np

from attenvis import AttentionVis  
av = AttentionVis()

@av.att_mat_func
def Attention_Matrix(K, V, Q, use_mask=False):
	
	window_size_queries = Q.get_shape()[1] 
	window_size_keys = K.get_shape()[1] 
    
	mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
	atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

	attention_weights = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / np.sqrt(K.shape[2])
    
	if use_mask:
		attention_weights += atten_mask
        
	attention_weights = tf.nn.softmax(attention_weights)
    
	attention = tf.matmul(attention_weights, V)

	return attention


class Atten_Head(tf.keras.layers.Layer):
    
	def __init__(self, input_size, output_size, use_mask):		
        
		super(Atten_Head, self).__init__()
        
		self.Weight_Q = self.add_weight(shape=[input_size, output_size], trainable=True)
		self.Weight_K = self.add_weight(shape=[input_size, output_size], trainable=True)
		self.Weight_V = self.add_weight(shape=[input_size, output_size], trainable=True)
        
		self.use_mask = use_mask
        
	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

		"""
		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""
        
		Q = tf.tensordot(inputs_for_queries, self.Weight_Q, axes=[[2], [0]])
		K = tf.tensordot(inputs_for_keys, self.Weight_K, axes=[[2], [0]])
		V = tf.tensordot(inputs_for_values, self.Weight_V, axes=[[2], [0]])
        
		attention = Attention_Matrix(K, V, Q, self.use_mask)

		return attention



class Multi_Headed(tf.keras.layers.Layer):
	def __init__(self, emb_sz, use_mask):
		super(Multi_Headed, self).__init__()
        
		# Initialize Multi Head Hyperparameters
		self.use_mask = use_mask
		self.d_model = emb_sz
		self.num_heads = 3
		self.depth = int(emb_sz/self.num_heads)
        
		assert self.d_model % self.num_heads == 0

		# Initialize heads
		self.head_1 = Atten_Head(emb_sz, self.depth, use_mask = self.use_mask)
		self.head_2 = Atten_Head(emb_sz, self.depth, use_mask = self.use_mask)
		self.head_3 = Atten_Head(emb_sz, self.depth, use_mask = self.use_mask)
        
		self.multi_headed_weights = self.add_weight(shape=[emb_sz, emb_sz], trainable=True)

	@tf.function
	def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
		"""
		:param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
		:return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
		"""
        
		self.head_1_attention = self.head_1.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
		self.head_2_attention = self.head_2.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
		self.head_3_attention = self.head_3.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
        
		self.multi_attention = tf.concat([self.head_1_attention, self.head_2_attention, self.head_3_attention], 2)

		multi_attention = tf.tensordot(self.multi_attention, self.multi_headed_weights, axes=[[2], [0]])

		return multi_attention


class Feed_Forwards(tf.keras.layers.Layer):
	def __init__(self, emb_sz):
		super(Feed_Forwards, self).__init__()

		self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
		self.layer_2 = tf.keras.layers.Dense(emb_sz)

	@tf.function
	def call(self, inputs):

		layer_1 = self.layer_1(inputs)
		layer_2 = self.layer_2(layer_1)
        
		return layer_2

class Transformer_Block(tf.keras.layers.Layer):
	def __init__(self, emb_sz, is_decoder, multi_headed=True):
		super(Transformer_Block, self).__init__()

		self.feed_forward_layer = Feed_Forwards(emb_sz)
		self.self_atten = Atten_Head(emb_sz,emb_sz,use_mask=is_decoder) if not multi_headed else Multi_Headed(emb_sz,use_mask=is_decoder)
		self.is_decoder = is_decoder
        
		if self.is_decoder:
			self.self_context_atten = Atten_Head(emb_sz,emb_sz,use_mask=False) if not multi_headed else Multi_Headed(emb_sz,use_mask=False)

		self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

	@tf.function
	def call(self, inputs, context=None):
		"""
		This functions calls a transformer block.

		There are two possibilities for when this function is called.
		    - if self.is_decoder == False, then:
		        1) compute unmasked attention on the inputs
		        2) residual connection and layer normalization
		        3) feed forward layer
		        4) residual connection and layer normalization

		    - if self.is_decoder == True, then:
		        1) compute MASKED attention on the inputs
		        2) residual connection and layer normalization
		        3) computed UNMASKED attention using context
		        4) residual connection and layer normalization
		        5) feed forward layer
		        6) residual layer and layer normalization

		If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)

		:param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
		:context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
			default=None, This is context from the encoder to be used as Keys and Values in self-attention function
		"""

		with av.trans_block(self.is_decoder):
			atten_out = self.self_atten(inputs,inputs,inputs)
		atten_out+=inputs
		atten_normalized = self.layer_norm(atten_out)

		if self.is_decoder:
			assert context is not None,"Decoder blocks require context"
			context_atten_out = self.self_context_atten(context,context,atten_normalized)
			context_atten_out+=atten_normalized
			atten_normalized = self.layer_norm(context_atten_out)

		ff_out=self.feed_forward_layer(atten_normalized)
		ff_out+=atten_normalized
		ff_norm = self.layer_norm(ff_out)

		return tf.nn.relu(ff_norm)

class Position_Encoding_Layer(tf.keras.layers.Layer):
	def __init__(self, window_sz, emb_sz):
		super(Position_Encoding_Layer, self).__init__()
		self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

	@tf.function
	def call(self, x):
		"""
		:param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
		:return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
		"""
		positional_embedding = x+self.positional_embeddings
        
		return positional_embedding
