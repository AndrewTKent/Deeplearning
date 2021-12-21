import numpy as np
import gzip

def get_data(inputs_file_path, labels_file_path, num_examples):
    
    # Reads Inputs and Labels, Not Including the Headers (16-byte Header for input and 8-byte Header for Label)
	with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
		bytefile = bytestream.read()
		inputs = np.frombuffer(bytefile, dtype=np.uint8, count=-1, offset=16)
		inputs = inputs.reshape((num_examples, 784))

    # Reads Labels, Not Including the Headers 8-byte Header for Label
	with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
		bytefile = bytestream.read()
		labels = np.frombuffer(bytefile, dtype=np.uint8, count=-1, offset=8)

    # Normalizes the Inputs, and Convers from float64 to float32
	inputs = inputs / 255.
	inputs = inputs.astype(np.float32)

	return inputs, labels


