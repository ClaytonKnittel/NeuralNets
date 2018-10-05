import tensorflow as tf


def convert(array):
    return str(array).replace('[', '{').replace(']', '}').replace(' ', '').replace('e', ' 10^')


x = tf.Variable(tf.random_normal((), stddev=2, mean=1))

one = tf.constant(1.0, dtype=tf.float32)

three = tf.constant(-3.0, dtype=tf.float32)

op = tf.pow(tf.add(tf.add(one, x), tf.add(tf.multiply(three, tf.pow(x, 2)), tf.pow(x, 3))), 2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=.05).minimize(op)

init_ops = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_ops)
    fw = tf.summary.FileWriter('./logs/w/train')
    xs = []

    for a in range(0, 30):
        m, x_ = sess.run([optimizer, x])

        xs.append(x_)

        print(x_)

    print(convert(xs))

    fw.close()


