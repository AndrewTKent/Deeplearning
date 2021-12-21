from contextlib import contextmanager
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

import random
import math


class Singleton(type):
    """Metaclass for Singleton objects

    From Python Cookbook, 3rd edition, by David Beazley and Brian K. Jones (O’Reilly).
    Copyright 2013 David Beazley and Brian Jones, 978-1-449-34037-7
    """
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class ReservoirSample1:
    """Randomly sample single item from data stream of indeterminate length
    
    Implementation after Algorithm L:
    Li, Kim-Hung. "Reservoir-sampling algorithms of time complexity O (n (1+ log (N/n)))."
    ACM Transactions on Mathematical Software (TOMS) 20, no. 4 (1994): 481-493.
    """
    
    def _update_s(self):
        """Update s based on w"""
        self.s = int(math.log(random.random()) / math.log(1.0 - self.w))
        
    def __init__(self):
        self.w = random.random()
        self._update_s()
        self.data = None
        self.stepno = 1
        self.target_step = 2 + self.s
        self.last_data_step = None # Just for debug
        
    def step(self, data):
        """Single time-step of data stream"""
        if self.stepno == 1:
            self.data = data
        elif self.stepno == self.target_step:
            self.data = data
            self.w = self.w * random.random()
            self._update_s()
            self.target_step = self.stepno + 1 + self.s
            self.last_data_step = self.stepno # Debug
        self.stepno += 1
            

class AttentionVis(metaclass=Singleton):
    """Global mechanism for saving and displaying attention-matrix visualization
    
    Store and later retrieve (and display) a single, randomly chosen attention matrix and
    its corresponding English sentence from the decoder block while the "test"
    operation is running.
    
    This class depends on the call structure (and argument order) of many stencil
    functions being preserved. It also depends on several function decorators
    and a "with" statement inside the transformer code to succesfully store and
    retrieve its data.
    """ 
    def __init__(self):
        self.enabled = False # Whether visualization even runs (decided by student)
        self.in_test = False # Shims are in the test function (as opposed to train)
        self.in_decoder = False # Shims are in the decoder (as opposed to encoder)
        self.rsample1 = ReservoirSample1() # Random sampling engine
        
        # randomly selected data within current batch
        # shared between the different shims
        self.cur_batch_data = { 
            "sent_ids": None,
            "att_mat": None,
            "index": None,
        }
        
        # Final data for showing heat map
        self.atten_matrix = None # The matrix itself (14x14)
        self.sentence = None # List of the words of the sentence
        self.rev_en_vocab = None # Reverse English vocabulary (id->word)

    def _setup_atten_heatmap(self, ax):
        """
        Create a heatmap from a numpy array and two lists of labels.
    
        This function derived from:
        https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

        ax - a "matplotlib.axes.Axes" instance to which the heatmap is plotted
        """
        
        data = self.atten_matrix
        row_labels = col_labels = self.sentence

        cbarlabel="Attention Score"
        cbar_kw={}

        # Plot the heatmap
        im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    def setup_visualization(self, enable=False):
        """Allow the student to turn visualization on or off"""
        self.enabled = enable

    def show_atten_heatmap(self):
        """Display heatmap from saved data after test run complete"""
        if self.enabled and self.atten_matrix is not None:
            fig, ax = plt.subplots()
            self._setup_atten_heatmap(ax)
            fig.tight_layout()
            plt.show()
        
    def test_func(self, func):
        """Shim for student's top-level test function
        
        Set a flag that signals to the other shims that we're inside the test function.
        Turn off graph execution if visualization is enabled.
        """
        def wrapper(*args, **kwargs):
            if self.enabled:
                self.in_test = True
                # The attention visualization is not compatible with tf.function. Oops.
                # Turn it off (but only if student requested visualizations)
                # The good news is: this won't affect "train"
                tf.config.run_functions_eagerly(True)
            ret = func(*args, **kwargs)
            # Save the data
            if self.enabled and self.rsample1.data:
                self.atten_matrix = self.rsample1.data["att_mat"]
                self.sentence = [self.rev_en_vocab[word] for word in self.rsample1.data["sent_ids"]]
                tf.config.run_functions_eagerly(False)
                self.in_test = False
                print("Collecting att matrix from batch", self.rsample1.last_data_step)
            return ret
        return wrapper
    
    def call_func(self, func):
        """Shim for student's main transformer call function
        
        If inside the test function, pick a random decoder sentence and remember its index
        """
        def wrapper(*args, **kwargs):
            if self.enabled and self.in_test:
                decoder_input = args[2]
                ridx = random.randint(0, len(decoder_input)-1)
                self.cur_batch_data["index"] = ridx
                self.cur_batch_data["sent_ids"] = list(decoder_input[ridx])
            ret = func(*args, **kwargs)
            if self.enabled and self.in_test:
                # Attention matrix should be collected at this point
                self.rsample1.step(self.cur_batch_data)
            return ret
        return wrapper
        
    def att_mat_func(self, func):
        """Shim for student's self-attention function
        
        If in the test function, and this is a decoder, store the attention matrix
        that corresponds with the saved index from the batch.
        """
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if self.enabled and self.in_test and self.in_decoder:
                # The return value of this function contains the to-be-visualized
                # attention matrix
                self.cur_batch_data["att_mat"] = ret[self.cur_batch_data["index"]].numpy()
            return ret
        return wrapper
        
    def get_data_func(self, func):
        """Shim for student's get-data function
        
        Collect the English vocab and reverse it
        """
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if self.enabled:
                en_vocab = ret[4]
                self.rev_en_vocab = {v:k for k,v in en_vocab.items()}
            return ret
        return wrapper
        
    @contextmanager
    def trans_block(self, is_decoder):
        """Shim for recording which transformer block we're in (encoder/decoder)"""
        try:
            self.in_decoder = is_decoder
            yield
        finally:
            self.in_decoder = False

