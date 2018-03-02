import tensorflow as tf


def dqn(state, num_actions, num_hidden=[], scope='dqn'):
    net = state
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i, n in enumerate(num_hidden):
            net = tf.contrib.layers.fully_connected(net, n, scope='fc%d' % i)
        net = tf.contrib.layers.fully_connected(net, num_actions,
                                                activation_fn=None,
                                                scope='fc')
    return net
