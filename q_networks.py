import tensorflow as tf


def dqn(state, num_actions, num_hidden, scope='dqn'):
    net = state
    with tf.variable_scope(scope):
        for i, n in enumerate(num_hidden):
            net = tf.contrib.layers.fully_connected(net, n, scope='fc%d' % i)
        net = tf.contrib.layers.fully_connected(net, num_actions,
                                                activation_fn=None,
                                                scope='fc%d' % len(num_hidden))
    return net


def dueling_dqn(state, num_actions, num_hidden, scope='dqn'):
    v_net = state
    a_net = state
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i, n in enumerate(num_hidden):
            v_net = tf.contrib.layers.fully_connected(v_net, n, scope='vnet/fc%d' % i)
        v_net = tf.contrib.layers.fully_connected(v_net, num_actions,
                                                  activation_fn=None,
                                                  scope='vnet/fc%d' % len(num_hidden))
        for i, n in enumerate(num_hidden):
            a_net = tf.contrib.layers.fully_connected(a_net, n, scope='anet/fc%d' % i)
        a_net = tf.contrib.layers.fully_connected(a_net, num_actions,
                                                  activation_fn=None,
                                                  scope='anet/fc%d' % len(num_hidden))
        q = v_net + a_net - tf.reduce_mean(v_net)
    return q
