import tensorflow as tf


def get_model(state, num_actions, scope):
    with tf.variable_scope(scope):
        net = tf.contrib.layers.fully_connected(state, num_actions, activation_fn=None, scope='fc')
    return net
