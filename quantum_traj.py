import numpy as np
import tensorflow as tf

import networks as n

import sys
sys.path.append('../qubits')

# noinspection PyUnresolvedReferences
import tester as t
# noinspection PyUnresolvedReferences
import qubit as q
# noinspection PyUnresolvedReferences
import evolutions as e
# noinspection PyUnresolvedReferences
import extramath as em


def convert(array):
    return str(array).replace('[', '{').replace(']', '}').replace(' ', '').replace('e', ' 10^')


sq2 = 1 / 2 ** .5


def test_safe(path):
    try:
        f = open(path, 'r')
    except IOError:
        return True
    try:
        return f.readline() == 'training data:\n'
    except IOError:
        return False


class data_generator:

    def __init__(self, measurements, trajectories, qubit=None):
        self.measures = measurements
        self.runs = trajectories
        self.r = []
        self.tr = []
        self.test_r = []
        self.test_tr = []

        self.qubit = None
        self.start_val = 0

        self.init_qubit(qubit)

    def init_qubit(self, qubit):
        if qubit is None:
            self.qubit = q.Qubit()
            self.qubit.init_psi(e.StochasticMeasurement(em.spinor(sq2, sq2), q.identity, 15))
            self.qubit.add_tracker('r', lambda psi: psi.last_r())
            self.qubit.add_tracker('z', lambda psi: psi.z(), measure_initially=True)
        else:
            self.qubit = qubit

    # training data
    def gen_training_data(self):
        for x in range(0, self.runs):
            qubit = t.individual(self.qubit, self.measures + 1)
            self.r.append(qubit.get_r())
            self.tr.append(np.array(qubit.get_traj())[:, 2].tolist())

    def save_training_data(self, path):
        txt = 'training data:\n' + str(self.measures) + '\n' + str(self.runs)
        for r_vals, tr_vals in zip(self.r, self.tr):
            txt += '\n\n'
            for r in r_vals:
                txt += '\n' + str(r)
            txt += '\n'
            for tr in tr_vals:
                txt += '\n' + str(tr)
        if test_safe(path):
            f = open(path, 'w')
            f.write(txt)
            f.close()
        else:
            raise IOError(path + ' not a training data file')

    @staticmethod
    def load_training_data(path, qubit=None, max_len=-1):
        f = open(path, 'r')
        title = f.readline()
        if title != 'training data:\n':
            raise IOError(path + ' not a training data file')
        data_gen = data_generator(int(f.readline()[:-1]), int(f.readline()[:-1]), qubit=qubit)
        r = []
        tr = []
        x = 0
        for runs in range(0, data_gen.runs):
            f.readline()
            f.readline()
            r.append([])
            for mea in range(0, data_gen.measures):
                r[-1].append(float(f.readline()[:-1]))
            f.readline()
            tr.append([])
            for mea in range(0, data_gen.measures + 1):
                tr[-1].append(float(f.readline()[:-1]))
            x += 1
            if x >= max_len and max_len != -1:
                break
        data_gen.r = r
        data_gen.tr = tr
        f.close()
        return data_gen

    def gen_testing_data(self, num_tests=20):
        for y in range(0, num_tests):
            qubit = self.qubit.__copy__()
            qubit = t.individual(qubit, self.measures)
            self.test_r.append(qubit.get_traj('r'))
            self.test_tr.append(np.array(qubit.get_traj('z')).tolist())

    def plot_tests(self, sess, plt, network):
        d = int(np.sqrt(len(self.test_r)))
        xe1 = range(0, self.measures + 1)
        avg_rms = 0

        for z in range(0, len(self.test_r)):
            get = network.input(sess, self.test_r[z], self.test_tr[z][0])
            rows = int(len(self.test_r) / d)
            cols = np.ceil(len(self.test_r) / rows)

            rms = n.get_rms(self.test_tr[z], get)
            avg_rms += rms

            ax = plt.subplot(rows, cols, z + 1)
            plt.plot(xe1, self.test_tr[z], color='red')
            plt.plot(xe1, get, color='blue')
            plt.text(.07, .83, 'rms: {:.4f}'.format(rms), transform=ax.transAxes)
        print('average rms: {:.4f}'.format(avg_rms / len(self.test_r)))
        plt.show()

    def avg_rms(self, sess, network):
        avg_rms = 0
        for z in range(0, len(self.test_r)):
            get = network.input(sess, self.test_r[z], self.test_tr[z][0])

            avg_rms += n.get_rms(get, self.test_tr[z])
        return avg_rms / len(self.test_r)

    def var_dist_end(self, sess, network):
        avg_dist_end = 0
        for z in range(0, len(self.test_r)):
            get = network.input(sess, self.test_r[z], self.test_tr[z][0])
            avg_dist_end += np.square(get[-1] - self.test_tr[z][-1])
        return np.sqrt(avg_dist_end / (len(self.test_r) - 1))


def run_test(dat_gen, v):
    with tf.Session() as sess:
        v += 1
        network = n.feedforward(dat_gen.measures, 2 * dat_gen.measures, dat_gen.measures + 1, tf=tf, learning_rate=.005,
                                decay_rate=.0002)
        init_op = tf.global_variables_initializer()

        sess.run(init_op)

        return network.train(sess, dat_gen.r, dat_gen.tr, 10, batch_size=5, print_=False)[-1]


def test_var(dat_gen, lista, avgs):
    ret = []
    for v in lista:
        t = 0
        for q in range(0, avgs):
            t += run_test(dat_gen, v)
        t /= avgs
        print(v, '\t', t)
        ret.append([t])
    return ret


# print(convert(test_var(range(10, 15), 5)))


def main(dat_gen, architecture=None, print_results=True, plot=False, save=False, lr=2e-5, dr=.0000):
    # simple
    tf.reset_default_graph()
    if architecture is None:
        lis = [1, 1]
    else:
        lis = []
        for xxe in architecture:
            lis.append(xxe)
    with tf.Session() as sess:

        for pso in range(0, len(lis)):
            lis[pso] = int(lis[pso] * dat_gen.measures)

        # network = n.feedforward(dat_gen.measures, *lis,
        #                         dat_gen.measures + 1, tf=tf, learning_rate=.005, decay_rate=.0002)
        network = n.feedforward(dat_gen.measures, *lis,
                                dat_gen.measures + 1, tf=tf, learning_rate=lr, decay_rate=dr)

        init_op = tf.global_variables_initializer()

        sess.run(init_op)
        # fw = tf.summary.FileWriter('./logs/qt/train', sess.graph)

        costs = network.train(sess, dat_gen.r, dat_gen.tr, 10, batch_size=5, print_=print_results)
        final_cost = costs[-1]

        if plot:
            import matplotlib.pyplot as plt
            dat_gen.plot_tests(sess, plt, network)

        # fw.close()
        if save:
            network.save(sess, '/users/claytonknittel/downloads/test.txt')

        avg_rms = dat_gen.avg_rms(sess, network)
        var_dist_end = dat_gen.var_dist_end(sess, network)
        return final_cost, avg_rms, var_dist_end


def main2(dat_gen, architecture=None, print_results=True, plot=False, save=False, lr=2e-5, dr=.0000):
    # iterating
    tf.reset_default_graph()
    with tf.Session() as sess:
        network = n.iterating_feedforward(2, *architecture, 1, tf=tf, learning_rate=lr, decay_rate=dr)

        init_op = tf.global_variables_initializer()

        sess.run(init_op)
        # fw = tf.summary.FileWriter('./logs/qt/train', sess.graph)

        costs = network.train(sess, dat_gen.r, dat_gen.tr, 10, batch_size=10, print_=print_results)
        final_cost = costs[-1]

        if plot:
            import matplotlib.pyplot as plt
            dat_gen.plot_tests(sess, plt, network)

        # fw.close()
        if save:
            network.save(sess, '/users/claytonknittel/downloads/test.txt')

        avg_rms = dat_gen.avg_rms(sess, network)
        var_dist_end = dat_gen.var_dist_end(sess, network)
        return final_cost, avg_rms, var_dist_end


def main3(dat_gen, print_results=True, plot=False):
    # recurrent
    tf.reset_default_graph()
    with tf.Session() as sess:
        network = n.recurrent_network(2, 4, 1, tf=tf, learning_rate=.005)

        init_op = tf.global_variables_initializer()

        sess.run(init_op)
        # fw = tf.summary.FileWriter('./logs/qt/train', sess.graph)

        costs = network.train(sess, dat_gen.r, dat_gen.tr, 10, print_=print_results)
        final_cost = costs[-1]

        if plot:
            import matplotlib.pyplot as plt
            dat_gen.plot_tests(sess, plt, network)

        # fw.close()
        # if save:
        #     network.save(sess, '/users/claytonknittel/downloads/test.txt')

        avg_rms = dat_gen.avg_rms(sess, network)
        var_dist_end = dat_gen.var_dist_end(sess, network)
        return final_cost, avg_rms, var_dist_end


def test(dat_gen):
    for lr in (.02, ):
        cost, avr, vardist = main2(dat_gen, lr=lr, print_results=True, plot=True)
        print(lr, '\tcost: ' + str(cost), '\tavr: ' + str(avr))


qubit = q.Qubit()
qubit.init_psi(e.StochasticMeasurement(em.spinor(sq2, sq2), q.identity, 15))
qubit.add_tracker('z', lambda psi: psi.z(), measure_initially=True)
qubit.add_tracker('r', lambda psi: psi.last_r())
# dat = data_generator(100, 10000)
# dat.gen_training_data(qubit)
# dat.save_training_data('/users/claytonknittel/downloads/data.txt')
dat = data_generator.load_training_data('/users/claytonknittel/downloads/data.txt', qubit=qubit, max_len=2000)
dat.gen_testing_data(num_tests=20)

# test(dat)

# REALLY GOOD
# main(dat, architecture=[2], lr=5e-4, dr=.00001, plot=True, print_results=True)
main2(dat, architecture=[8, 8], lr=5e-2, dr=0., plot=True, print_results=True)

# res = []
# for arch in ([1], [2], [4], [6], [1, 1], [2, 2], [4, 4], [6, 6]):
#     res.append([])
#     for lr in (1e-6, 5e-6, 2e-5, 1e-4, 5e-4, 2e-3, 1e-2, 5e-2, 2e-1, 1.0):
#         res[-1].append([])
#         for dr in (0., .00001, .0001, .001, .01, .1):
#             print('arch: ' + str(arch) + '\tlr: ' + str(lr) + '\tdr:' + str(dr))
#             cost, rms, varend = main2(dat, architecture=arch, lr=lr, dr=dr, plot=False, print_results=False)
#             print('cost:', cost, '\trms:', rms, '\tvarend:', varend)
#             res[-1][-1].append([cost, rms, varend])
#             print()
# print(convert(res))

# main3(dat, plot=True)

# fc = []
# rm = []
# vs = []
# for arch in ([1], [1, 1], [2, 1]):
#     tc = 0.0
#     tr = 0.0
#     tv = 0.0
#     for rr in range(0, 15):
#         c, r, v = main(dat, architecture=arch, print_results=False)
#         print('architecture: ' + str(arch), '\tfinal cost: {:.4f}\trms: {:.4f}\tvar end: {:.4f}'.format(c, r, v))
#         tc += c
#         tr += r
#         tv += v
#     fc.append(tc / 5)
#     rm.append(tr / 5)
#     vs.append(tv / 5)
#
# print(convert([fc, rm, vs]))


def test_data(dat_gen):
    with tf.Session() as sess:
        network = n.feedforward.fromfile('/users/claytonknittel/downloads/test.txt')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        xe1 = range(0, dat_gen.measures + 1)
        import matplotlib.pyplot as plt

        d = int(np.sqrt(len(dat_gen.test_r)))
        for z in range(0, len(dat_gen.test_r)):
            get = network.input(sess, dat_gen.test_r[z], dat_gen.start_val)
            rows = int(len(dat_gen.test_r) / d)
            cols = np.ceil(len(dat_gen.test_r) / rows)

            rms = 0
            for y, yh in zip(dat_gen.test_tr[z], get):
                dq = y - yh
                rms += dq * dq
            rms = np.sqrt(rms / len(get))

            ax = plt.subplot(rows, cols, z + 1)
            plt.scatter(xe1, dat_gen.test_tr[z], color='red')
            plt.scatter(xe1, get, color='blue')
            plt.text(.07, .83, 'rms: {:.4f}'.format(rms), transform=ax.transAxes)
        plt.show()
