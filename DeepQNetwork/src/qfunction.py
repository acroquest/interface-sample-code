# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import tensorflow as tf


def clipped_error(x):
    """
    誤差巻数

    :param x: 入力ベクトル
    :return: 計算結果
    """
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class QFunction(object):
    """
    Q関数を計算するネットワーク
    """

    def __init__(self, screen_height, screen_width, n_history, action_size):
        """

        :param screen_height: 画面サイズ（縦）
        :param screen_width: 画面サイズ（横）
        :param n_history: ヒストリー（数）
        :param action_size: 行動パタン数
        """
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.history_length = n_history
        self.action_size = action_size

        self.w = {}

        self.q = self.build_network()

    def build_network(self):
        """
        計算グラフの構築

        :return: 計算結果
        """
        initializer = tf.truncated_normal_initializer(0, 0.02)

        # (1) プレースホルダーの初期化
        self.s_t = tf.placeholder('float32',
                                  [None, self.screen_height, self.screen_width, self.history_length], name='s_t')

        # (2) 畳み込み演算の定義
        self.l1, self.w['l1_w'], self.w['l1_b'] = self.conv2d(self.s_t,
                                                              32, [8, 8], [4, 4], initializer, name='l1')
        self.l2, self.w['l2_w'], self.w['l2_b'] = self.conv2d(self.l1,
                                                              64, [4, 4], [2, 2], initializer, name='l2')
        shape = self.l2.get_shape().as_list()
        # (3) 全結合層の定義
        self.l2_flat = tf.reshape(self.l2, [-1, shape[1] * shape[2] * shape[3]])
        self.l3, self.w['l4_w'], self.w['l4_b'] = self.linear(self.l2_flat, 512, activation_fn=tf.nn.relu, name='l4')
        self.q, self.w['q_w'], self.w['q_b'] = self.linear(self.l3, self.action_size, name='q')

        return self.q

    def conv2d(self, x,
               output_dim,
               kernel_size,
               stride,
               initializer=tf.contrib.layers.xavier_initializer(),
               padding='VALID',
               name='conv2d'):
        """
        2次元畳み込み演算を計算する。

        :param x: 入力画像
        :param output_dim: 出力次元
        :param kernel_size: 畳み込みのカーネルサイズ
        :param stride: 畳み込みの幅
        :param initializer:
        :param padding: paddingの方式、本手法はVALIDを使う。
        :param name: tensorflowのscopeの名前
        :return: 出力、重み、バイアス
        """
        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

            w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding, data_format="NHWC")

            b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, b, "NHWC")
            out = tf.nn.relu(out)

        return out, w, b

    def linear(self, input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
        """
        全結合相を計算するための関数

        :param input_: 入力
        :param output_size: 出力サイズ
        :param stddev: 標準偏差
        :param bias_start: バイアス
        :param activation_fn: 活性化関数
        :param name: 変数のスコープ名
        :return: 計算グラフの出力
        """
        shape = input_.get_shape().as_list()

        with tf.variable_scope(name):
            w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('bias', [output_size],
                                initializer=tf.constant_initializer(bias_start))

            out = tf.nn.bias_add(tf.matmul(input_, w), b)

            # 活性化関数を利用するか否か
            if activation_fn is not None:
                return activation_fn(out), w, b
            else:
                return out, w, b


