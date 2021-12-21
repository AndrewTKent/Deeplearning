from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from random import randint
import numpy as np

class Model:

    def __init__(self):
        # Initializes the MNIST Parameters
        self.batch_size = 48
        self.learning_rate = 0.2
        self.num_classes = 10  
        self.input_size = 784  

        # Sets the W Matrix and Bias Vector for the Perceptron
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1, self.num_classes))

    def call(self, inputs):
        # Makes the Loss Function, then Returns the Probability for Each Class Using Softmax
        # loss = W * x + b
        loss = inputs @ self.W + self.b  
        exp_loss = np.exp(loss)
        sum_exp_loss = np.sum(exp_loss, 1)
        sum_exp_loss = sum_exp_loss.reshape(sum_exp_loss.shape[0], 1) 
        probabilities = exp_loss / sum_exp_loss  
        
        return probabilities

    def loss(self, probabilities, labels):
        
        # Calculates Average Cross Entropy Loss for a Batch
        prob = np.diag(probabilities[:, labels])
        loss = - np.log(prob)
        average_loss = np.mean(loss)
        
        return average_loss
    
    def back_propagation(self, inputs, probabilities, labels):
        
        # Calcualtes the Gradients for the Weights and the Bias w.r.t. Average Loss
        Y_vec = np.eye(self.batch_size) [labels, 0 : self.num_classes]
        grad_W = self.learning_rate * inputs.T @ (Y_vec - probabilities) / self.batch_size  
        grad_B = self.learning_rate * np.sum(Y_vec - probabilities) / self.batch_size      
        
        return grad_W, grad_B
    
    def accuracy(self, probabilities, labels):
        
        # Calculates the Models Accuracy 
        prediction = np.argmax(probabilities, axis=1)
        accuracy = sum(prediction == labels) / labels.shape[0]
        
        return accuracy

    def gradient_descent(self, gradW, gradB):
        
        # Updates the Weights and Biases of the Model for the Gradient Descent
        self.W += gradW
        self.b += gradB


def train(model, train_inputs, train_labels):
    
    # Trains the Model on all of the Inputs and Labels, then Decends the Gradients
    batch_train_size = int(train_inputs.shape[0]/model.batch_size)
    
    for i in range(0, batch_train_size):
        labels = train_labels[model.batch_size*i : model.batch_size*(i+1)]
        inputs = train_inputs[model.batch_size*i : model.batch_size*(i+1), :]
        probabilities = model.call(inputs)
        gradW, gradB = model.back_propagation(inputs, probabilities, labels)
        model.gradient_descent(gradW, gradB)


def test(model, test_inputs, test_labels):
    
    # Tests the model on the test inputs and labels and Returns accuracy across testing set
    probabilities = model.call(test_inputs)
    testing_set_accuracy = model.accuracy(probabilities, test_labels)
    
    return testing_set_accuracy


def visualize_results(image_inputs, probabilities, image_labels):

    # Uses Matplotlib to Visualize the Results of our Model
    
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main():
    
    # Reads in MNIST data, Initializes the Model, and Trains and Tests the Model for one Epoch
    num_train = 60000
    num_test = 10000
    random_num = randint(0, num_test - 10)
    train_inputs, train_labels = get_data('MNIST_data/train-images-idx3-ubyte.gz', 'MNIST_data/train-labels-idx1-ubyte.gz', num_train)
    test_inputs, test_labels = get_data('MNIST_data/t10k-images-idx3-ubyte.gz', 'MNIST_data/t10k-labels-idx1-ubyte.gz', num_test)

    # Creates and Trains Model
    model = Model()
    train(model, train_inputs, train_labels)

    # Tests Accuracy
    accuracy = test(model, test_inputs, test_labels)
    print('accuracy =', accuracy)

    # Randomly Take 10 Images for Visualization
    input = test_inputs[random_num : random_num + 10, :]
    label = test_labels[random_num : random_num + 10]
    visualize_results(input, model.call(input), label)
    
if __name__ == '__main__':
    main()
