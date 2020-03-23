import tensorflow as tf
import numpy as np


class Network(object):
    def __init__(self, layer_structure, input_shape, output_shape, learning_rate=0.001, mode='next_state'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.sess = tf.Session()
        self.learning_rate = learning_rate
        self.mode = mode

        self.network_input, self.network_output = self.build_network(layer_structure)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.output_data = tf.placeholder(dtype=np.float32, shape=(self.output_shape, 1), name='output_data')
        self.loss = tf.reduce_sum(tf.square(self.network_output - self.output_data))

    def build_network(self, layer_structure):
        input_layer = tf.placeholder(dtype=np.float32, shape=(self.input_shape, 1), name='input_layer')
        temp_layer = input_layer
        for idx, units in enumerate(layer_structure):
            temp_layer = tf.layers.dense(inputs=temp_layer, units=units, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.orthogonal)
        output_layer = tf.layers.dense(inputs=temp_layer, units=self.output_shape, activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.orthogonal)
        return input_layer, output_layer

    def train(self, replay_batch):
        objective = self.optimizer.minimize(self.loss)
        loss_list = []
        for replay in replay_batch:
            input_data = np.concatenate([replay['state'].flatten(), replay['action'].flatten()])
            _, loss = self.sess.run(objective, feed_dict={self.network_input: input_data, self.output_data: replay[self.mode]})
            loss_list.append(loss)
        return np.mean(loss_list)

    def predict(self, state, action):
        input_data = np.concatenate([state.flatten(), action.flatten()])
        output = self.sess.run(self.network_output, feed_dict={self.network_input: input_data})
        return output

    def save(self, save_path):
        pass

    def load(self, load_path):
        pass

