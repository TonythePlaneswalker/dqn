import tensorflow as tf
import dqn


def get_model(state, num_actions, scope):
    with tf.variable_scope(scope):
        v_net = dqn.get_model(state, 1, scope='v_net')
        a_net = dqn.get_model(state, num_actions, scope='a_net')
        q = v_net + a_net - tf.reduce_mean(a_net)
    return q
