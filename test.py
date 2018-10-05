import numpy as np


def calculate_total_loss(y):
    L = 0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]]), np.array([[0, 1, 0], [0, 0, 1]])
        # We only care about our prediction of the "correct" words
        print(np.arange(len(y[i])))
        correct_word_predictions = o[np.arange(len(y[i])), y[i]]
        print(correct_word_predictions)
        # Add to the loss based on how off we were
        L += -1 * np.sum(np.log(correct_word_predictions))
    return L


calculate_total_loss([[0, 1, 0], [1, 0, 0]])


def bptt(x, y):
    T = len(y)
    # Perform forward propagation
    o, s = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]]), np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])
            dLdU[:,x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
