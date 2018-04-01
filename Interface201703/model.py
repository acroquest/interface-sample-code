# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import tensorflow as tf

BATCH_SIZE = 16
NUM_CHANNELS = 1
IMAGE_SIZE = 28
SEED = 82
NUM_LABELS = 2


def define_model(images):
    """
    ニューラル・ネットワークのモデルを定義する。

    :return: 計算後の結果
    """

    # 畳み込みニューラル・ネットワークのパラメータの初期化を行う。
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],
                            stddev=0.1,
                            seed=SEED, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
    fc1_weights = tf.Variable(
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=tf.float32))

    # プレースホルダに対して、計算を行う。
    with tf.variable_scope("conv1") as scope:
        conv = tf.nn.conv2d(images,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    with tf.variable_scope("conv2") as scope:
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    pool_shape = pool.get_shape().as_list()
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    return tf.matmul(hidden, fc2_weights) + fc2_biases


def define_loss(logits, labels):
    """
    クロス・エントロピー誤差の定義

    :param logits: 入力値
    :param labels: ラベル
    :return: 誤差平均
    """
    labels = tf.to_int64(labels)
    # クロス・エントロピー誤差の計算
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    # 計算したクロス・エントロピー誤差の平均
    cross_entropy_loss_mean = tf.reduce_mean(cross_entropy_loss)
    return cross_entropy_loss_mean


def training(loss, learning_rate):
    """
    パラメータ更新方法を定義する

    :param loss: 計算した誤差
    :param learning_rate: 学習係数
    :return:
    """
    batch = tf.Variable(0, dtype=tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                        global_step=batch)
    return optimizer


def evaluation(logits, labels):
    """
    評価

    :param logits: 計算結果
    :param labels: ラベル
    :return: 評価
    """
    # 正しければ1, 誤っていれば0
    correct = tf.nn.in_top_k(logits, labels, 1)
    # 正解した数を計算する。
    return tf.reduce_sum(tf.cast(correct, tf.int32))
