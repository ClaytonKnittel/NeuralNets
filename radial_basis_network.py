import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def normal(tensor, variance):
    return tf.exp(tf.negative(tf.divide(tf.tensordot(tensor, tensor), variance * variance * 2)))


class radial_basis_network:

    def __init__(self, num_inputs, num_hidden, num_outputs, rbfunction=normal, learning_rate=.5, stddev=.03):
        self.__i_size = num_inputs
        self.__h_size = num_hidden
        self.__o_size = num_outputs
        self.__rbf = rbfunction
        self.__lr = learning_rate
        self.__sd = stddev
        self.__create()

    def __create(self):
        self.__input = tf.placeholder(tf.float32, [None, self.__i_size], name='input_layer')

        with tf.variable_scope('radial_basis_function'):
            self.__centers = tf.Variable(tf.random_normal([self.__h_size, 1, self.__i_size], stddev=self.__sd),
                                         name='rbf_centers')
            self.__stddevs = tf.Variable(tf.random_normal([self.__h_size], stddev=self.__sd), name='rbf_stddevs')
            norms = tf.transpose(tf.reduce_sum(tf.square(
                tf.subtract(self.__input, self.__centers, name='norms')), axis=2))
            self.__hidden_out = tf.divide(norms, tf.multiply(tf.constant(2, dtype=tf.float32),
                                                             tf.square(self.__stddevs)), name='hidden_out')

        with tf.variable_scope('linear_combination'):
            self.__weights = tf.Variable(tf.random_normal([self.__h_size, self.__o_size],
                                                          stddev=self.__sd), name='weights')
            self.__biases = tf.Variable(tf.random_normal([self.__o_size], stddev=self.__sd), name='biases')

            self.__z = tf.add(tf.matmul(self.__hidden_out, self.__weights), self.__biases)
            # self.__output = tf.clip_by_value(tf.sigmoid(self.__z), 1e-10, 0.9999999)
            self.__output = self.__z

        with tf.variable_scope('compute_cost'):
            self.__ex_output = tf.placeholder(tf.float32, [None, self.__o_size], name='expected_output')
            # exout = tf.clip_by_value(tf.sigmoid(self.__ex_output), 1e-10, 0.9999999)
            exout = self.__ex_output
            # v_cost = tf.reduce_sum(exout * tf.log(self.__output) +
            #                        (1 - exout) * tf.log(1 - self.__output), axis=1)
            v_cost = tf.reduce_sum(tf.square(exout - self.__output), axis=1)
            self.__cost = tf.reduce_mean(v_cost)
            self.__optimizer = tf.train.AdamOptimizer(learning_rate=self.__lr).minimize(self.__cost)

    def run_test(self, sess, inputs, ex_outputs, epochs, batch_size=1, print_epochs=True):
        inputs = np.array(inputs)
        ex_outputs = np.array(ex_outputs)

        batches = int(len(inputs) / batch_size)
        for epoch in range(0, epochs):
            np.random.shuffle(ex_outputs)

            tcost = 0
            for batch in range(0, batches):
                ins = inputs[batch * batch_size: (batch + 1) * batch_size]
                outs = ex_outputs[batch * batch_size: (batch + 1) * batch_size]
                o, cost = sess.run([self.__optimizer, self.__cost],
                                   feed_dict={self.__input: ins, self.__ex_output: outs})
                tcost += cost / batches

            if print_epochs:
                print('epoch ' + str(epoch + 1) + ':', '\tcost ' + str(tcost))

    def tot_cost(self, sess, inputs, outputs):
        return sess.run(self.__cost, feed_dict={self.__input: inputs, self.__ex_output: outputs})

    def input(self, sess, inp):
        res = sess.run([self.__z], feed_dict={self.__input: [inp]})
        return res[0][0].tolist()


def create_data(func, start, stop, steps):
    ret = []
    d = (stop - start) / steps
    for x in range(0, steps):
        ret.append(func(start + d * x))
    return ret


def cos(freq, offset):
    return lambda x: (1 + np.cos(x * freq + offset)) / 2


def convert(array):
    return str(array).replace('[', '{').replace(']', '}').replace(' ', '').replace('e', ' 10^')


data_ins = []
data_outs = []

size = 10

for x in range(0, 100):
    freq1 = .1 * 2 * np.pi
    freq2 = .1 * 2 * np.pi
    offset1 = np.random.random() * np.pi * 2
    offset2 = offset1 + np.pi
    ins = create_data(cos(freq1, offset1), 0, size - 1, size)
    outs = create_data(cos(freq2, offset2), 0, size - 1, size)
    data_ins.append(ins)
    data_outs.append(outs)

# fdat = create_data(cos(.01 * 2 * np.pi, np.pi), 0, 100, 101)
# print(str(data_ins).replace('[', '{').replace(']', '}').replace(' ', ''))


r = radial_basis_network(size, 50, size, learning_rate=.06, stddev=2)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    fw = tf.summary.FileWriter('./logs/radial_basis/train', sess.graph)
    r.run_test(sess, data_ins, data_outs, 150, batch_size=5, print_epochs=True)

    inp = create_data(cos(.1 * 2 * np.pi, 0), 0, size - 1, size)
    res = r.input(sess, inp)
    # print(convert(res))

    xe = np.arange(0, size, 1)
    plt.scatter(xe, res)
    plt.scatter(xe, inp)
    plt.show()

    fw.close()
