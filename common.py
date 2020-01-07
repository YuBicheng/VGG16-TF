import tensorflow as tf
import tensorflow.contrib as ct

def convolutional(input, shape, trainable, name, regularized=ct.layers.l2_regularizer(0.01)):
    with tf.variable_scope(name):
        weights = tf.get_variable(name='weights', shape=shape, trainable=True, dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
        tf.add_to_collection('loss', regularized(weights))
        conv = tf.nn.conv2d(input, weights, (1,1,1,1), 'SAME')
        biases = tf.get_variable(name='bias', shape=shape[-1], trainable=True, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(conv)
    return relu

def pool(input, shape=[1,2,2,1], strides=[1,2,2,1]):
    pool = tf.nn.max_pool(input, ksize=shape, strides=strides, padding="SAME")
    return pool

def fullconnect(input, out_dim, name, dropout=True, regularized=ct.layers.l2_regularizer(0.01)):
    input_dim = input.get_shape().as_list()[1]
    with tf.variable_scope(name):
        weights = tf.get_variable(name='weights', shape=[input_dim, out_dim], trainable=True, dtype=tf.float32, initializer=tf.random_normal_initializer(0.01))
        tf.add_to_collection('loss', regularized(weights))
        biases = tf.get_variable(name='biases', shape=[out_dim], trainable=True, dtype=tf.float32, initializer=tf.random_normal_initializer(0.01))
        fc = tf.matmul(input, weights) + biases
        relu = tf.nn.relu(fc)
        if dropout:
            relu = tf.nn.dropout(relu, rate=0.5)
    return relu
