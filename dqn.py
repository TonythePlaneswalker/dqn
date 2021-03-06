import tensorflow as tf


def get_model(state, num_actions, scope):
    num_hidden = [30, 30, 30]
    with tf.variable_scope(scope):
        net = state
        for i, n in enumerate(num_hidden):
            net = tf.contrib.layers.fully_connected(net, n, scope='fc_%d' % i)
        net = tf.contrib.layers.fully_connected(net, num_actions, activation_fn=None,
                                                scope='fc_%d' % len(num_hidden))
    return net
