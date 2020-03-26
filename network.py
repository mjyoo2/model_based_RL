from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import tensorflow as tf

class Network(object):
    def __init__(self, layer_structure, input_shape, output_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate

        self.network = self.build_network(layer_structure)
        self.network.compile(loss='mse', optimizer='adam')

    def build_network(self, layer_structure):
        inputs = Input(shape=(self.input_shape, ), name='encoder_input')
        x = inputs
        for dims in layer_structure:
            x = Dense(dims, activation='relu')(inputs)
        output = Dense(self.output_shape)(x)
        outputs = Model(inputs, output, name='model')
        return outputs

    def train(self, input_data, output_data, training_epochs):
        input_data = np.array(input_data).reshape([-1, self.input_shape])
        output_data = np.array(output_data).reshape([-1, self.output_shape])
        self.network.fit(input_data, output_data, nb_epoch=training_epochs, batch_size=64)
        return

    def predict(self, input):
        self.network.predict(input)
        return input

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass

