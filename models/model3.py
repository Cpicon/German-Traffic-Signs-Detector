from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import numpy as np
import os


class LeNet5:
    SAVE_PATH = os.path.abspath('models/model3/saved/Letnet5')
    NUM_EPOCHS = 200
    # cantidad de minibach ultizados para actualizar los pesos
    BATCH_SIZE = 150
    # numero de categorias posibles 0 - 42
    CLASSES_SIZE = 43
    # tasa de aprendizaje optima
    LEARNING_RATE = 0.001  # 1e-4

    def __init__( self ):
        pass

    def _construct( self, x ):

        # Hyperparametros
        self.mu = 0
        self.sigma = 0.1

        # Layer 1 (Convolutional): Input = 32x32x1. Output = 28x28x6.
        self.filter1_width = 5
        self.filter1_height = 5
        self.input1_channels = 1
        self.conv1_output = 6

        tf.reset_default_graph()


        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        self.conv1_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter1_width, self.filter1_height, self.input1_channels, self.conv1_output), mean=self.mu,
            stddev=self.sigma))
        self.conv1_bias = tf.Variable(tf.zeros(self.conv1_output))
        self.conv1 = tf.nn.conv2d(x, self.conv1_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_bias
        # activation
        self.conv1 = tf.nn.relu(self.conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2 (Convolutional): Output = 10x10x16.
        self.filter2_width = 5
        self.filter2_height = 5
        self.input2_channels = 6
        self.conv2_output = 16
        # Weight and bias
        self.conv2_weight = tf.Variable(tf.truncated_normal(shape=(self.filter2_width, self.filter2_height, self.input2_channels, self.conv2_output),
            mean=self.mu, stddev=self.sigma))
        self.conv2_bias = tf.Variable(tf.zeros(self.conv2_output))
        # Apply Convolution
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_weight, strides=[1, 1, 1, 1],
                                  padding='VALID') + self.conv2_bias

        # Activation:
        self.conv2 = tf.nn.relu(self.conv2)

        # Pooling: Input = 10x10x16. Output = 5x5x16.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flattening: Input = 5x5x16. Output = 400.
        self.fully_connected0 = flatten(self.conv2)

        # Layer 3 (Fully Connected): Input = 400. Output = 120.
        self.connected1_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
        self.connected1_bias = tf.Variable(tf.zeros(120))
        self.fully_connected1 = (tf.matmul(self.fully_connected0, self.connected1_weights)) + self.connected1_bias

        # Activation:
        self.fully_connected1 = tf.nn.relu(self.fully_connected1)

        # Layer 4 (Fully Connected): Input = 120. Output = 84.
        self.connected2_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
        self.connected2_bias = tf.Variable(tf.zeros(84))
        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.connected2_weights)), self.connected2_bias)

        # Activation.
        self.fully_connected2 = tf.nn.relu(self.fully_connected2)

        # Layer 5 (Fully Connected): Input = 84. Output = 43.
        self.output_weights = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=self.mu, stddev=self.sigma))
        self.output_bias = tf.Variable(tf.zeros(43))
        self.logits = tf.add((tf.matmul(self.fully_connected2, self.output_weights)), self.output_bias)

        self.y = tf.placeholder(tf.int32, None)
        self.one_hot_y = tf.one_hot(self.y, self.CLASSES_SIZE)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Saving all variables
        self.saver = tf.train.Saver()

        self.keep_prob = tf.placeholder(tf.float32)  # For fully-connected layers
        self.keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers






    def train( self, X_data, Y_data ):
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        self._construct(x)
        with tf.Session() as se:
            se.run(tf.global_variables_initializer())
            X_t, Y_t = shuffle(X_data, Y_data)  # shuffle the training data to increase randomness and variety in training dataset, in order for the model to be more stable.
            print("Entrenando el modelo...")

            for i in range(self.NUM_EPOCHS):
                for offset in range(0, len(X_data), self.BATCH_SIZE):
                    batch_x = X_data[offset:offset + self.BATCH_SIZE],
                    batch_y = Y_data[offset:offset + self.BATCH_SIZE],
                    x = np.array(batch_x)
                    print(len(batch_x))
                    se.run(self.training_operation,
                           feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5, self.keep_prob_conv: 0.7})
                    print(" EPOCH {} Accuracy = {:.2f}%".format(i + 1, self.accuracy(X_data, Y_data) * 100))
            self._save()

    def predict( self, data ):

        return tf.argmax(self.logits, 1).eval(feed_dict={self.x: data}, session=self.session.as_default())

    def _save( self ):
        saved = self.saver.save(tf.get_default_session(), self.SAVE_PATH)
        print("Modelo guardado en: %s", saved)

    def accuracy( self, X_data, Y_data):

        num_examples = len(X_data)
        total_accuracy = 0
        se = tf.get_default_session()
        for offset in range(0, num_examples, self.BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset + self.BATCH_SIZE], Y_data[offset:offset + self.BATCH_SIZE]
            accuracy = se.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.5, self.keep_prob_conv: 0.7})
            total_accuracy += (accuracy * len(batch_x))

        return total_accuracy / num_examples
