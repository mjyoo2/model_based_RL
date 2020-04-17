from keras.layers import Input, Dense, BatchNormalization, ReLU, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

import numpy as np
import os


class Network(object):
    def __init__(self, layer_structure, action_shape, state_shape, output_shape, name, last_layer=None, loss='mse', metrics=None):
        self.action_shape = action_shape
        self.state_shape = state_shape
        self.output_shape = output_shape
        self.last_layer = last_layer

        self.network = self.build_network(layer_structure, last_layer)
        self.network_compile(loss, Adam(learning_rate=0.001), metrics)
        self.save_path = './weights/{}/init_weight.hdf5'.format(name)
        if not os.path.isdir('./weights/{}'.format(name)):
            os.mkdir('./weights/{}'.format(name))
        self.network.save_weights(self.save_path)

    def network_compile(self, loss, optimizer, metrics):
        if metrics is None:
            self.network.compile(loss=loss, optimizer=optimizer)
        else:
            self.network.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    def build_network(self, layer_structure, last_activation):
        actions_input = Input(shape=(self.action_shape, ), name='action_input')
        actions = Dense(32, activation='relu')(actions_input)
        actions = BatchNormalization()(actions)
        actions = ReLU()(actions)
        states_input = Input(shape=(self.state_shape, ), name='state_input')
        states = Dense(32, activation='relu', kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))(states_input)
        states = BatchNormalization()(states)
        x = Concatenate()([actions, states])
        x = Dense(64, activation='relu', kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))(x)
        for dims in layer_structure:
            x = Dense(dims, kernel_initializer='orthogonal', kernel_regularizer=l2(0.01))(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        if last_activation is None:
            output = Dense(self.output_shape)(x)
        else:
            output = Dense(self.output_shape, activation=last_activation)(x)
        outputs = Model([actions_input, states_input], output, name='model')
        return outputs

    def train(self, state_data, action_data, output_data, training_epochs):
        action_data = np.array(action_data).reshape([-1, self.action_shape])
        state_data = np.array(state_data).reshape([-1, self.state_shape])
        output_data = np.array(output_data).reshape([-1, self.output_shape])
        self.network.fit([action_data, state_data], output_data, nb_epoch=training_epochs, batch_size=1024, verbose=0)
        return

    def predict(self, state_data, action_data):
        action_data = np.array(action_data).reshape([-1, self.action_shape])
        state_data = np.array(state_data).reshape([-1, self.state_shape])
        input_data = [action_data, state_data]
        return self.network.predict(input_data)

    def reinit(self):
        self.network.load_weights(self.save_path)
