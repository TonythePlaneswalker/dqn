import tensorflow as tf


def get_model(state, num_actions, scope):
    with tf.variable_scope(scope):
        feature = tf.contrib.layers.conv2d(state, 32, kernel_size=[8, 8], stride=4, padding='VALID', scope='conv_0')
        feature = tf.contrib.layers.conv2d(feature, 64, kernel_size=[4, 4], stride=2, padding='VALID', scope='conv_1')
        feature = tf.contrib.layers.conv2d(feature, 64, kernel_size=[3, 3], stride=1, padding='VALID', scope='conv_2')
        feature = tf.contrib.layers.flatten(feature)
        v_net = tf.contrib.layers.fully_connected(feature, 512, scope='v_net/fc_0')
        v_net = tf.contrib.layers.fully_connected(v_net, 1, scope='v_net/fc1')
        a_net = tf.contrib.layers.fully_connected(feature, 512, scope='a_net/fc_0')
        a_net = tf.contrib.layers.fully_connected(a_net, num_actions, scope='a_net/fc_1')
        q = v_net + a_net - tf.reduce_mean(a_net)
    return q
