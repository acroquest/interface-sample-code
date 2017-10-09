# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import tensorflow as tf


class Generator(object):
    """
    GANの生成器を定義する。
    """

    def __init__(self, generator_layers, s_size):
        self.generator_layers = generator_layers + [3]
        self.s_size = s_size
        self.reuse = False

    def build_network(self, inputs, training):
        with tf.variable_scope("generator", reuse=self.reuse):
            # (1)入力を変換するレイヤの作成
            with tf.variable_scope("g_first_layer"):
                outputs = tf.layers.dense(inputs, self.s_size * self.s_size * self.generator_layers[0])
                outputs = tf.reshape(outputs, shape=(-1, self.s_size, self.s_size, self.generator_layers[0]))
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training))
            # (2)逆畳み込みレイヤの作成
            with tf.variable_scope("g_dconv1"):
                outputs = tf.layers.conv2d_transpose(outputs, filters=self.generator_layers[1], kernel_size=[5, 5],
                                                     padding="SAME", strides=(2, 2))
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training))
            with tf.variable_scope("g_dconv2"):
                outputs = tf.layers.conv2d_transpose(outputs, filters=self.generator_layers[2], kernel_size=[5, 5],
                                                     padding="SAME", strides=(2, 2))
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training))
            with tf.variable_scope("g_dconv3"):
                outputs = tf.layers.conv2d_transpose(outputs, filters=self.generator_layers[3], kernel_size=[5, 5],
                                                     padding="SAME", strides=(2, 2))
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training))
            with tf.variable_scope("g_dconv4"):
                outputs = tf.layers.conv2d_transpose(outputs, filters=self.generator_layers[4], kernel_size=[5, 5],
                                                     padding="SAME", strides=(2, 2))
                outputs = tf.nn.tanh(tf.layers.batch_normalization(outputs, training))

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        return outputs


class Discriminator(object):
    """
    GANの識別器を定義する。
    """

    def __init__(self, discriminator_layer):
        self.discriminator_layer = discriminator_layer
        self.reuse = False

    def build_network(self, inputs, training):
        """
        ニューラルネットワークを構築する。
        :param inputs: 入力
        :return: 出力
        """

        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        with tf.variable_scope("discriminator", reuse=self.reuse):
            # (3)畳み込みレイヤの作成
            with tf.variable_scope("d_conv1"):
                outputs = tf.layers.conv2d(inputs, filters=self.discriminator_layer[0], kernel_size=[5, 5],
                                           strides=(2, 2),
                                           padding='SAME')
                # outputs = tf.nn.elu(tf.layers.batch_normalization(outputs, training))
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training), name="outputs")
            with tf.variable_scope("d_conv2"):
                outputs = tf.layers.conv2d(outputs, filters=self.discriminator_layer[1], kernel_size=[5, 5],
                                           strides=(2, 2),
                                           padding='SAME')
                # outputs = tf.nn.elu(tf.layers.batch_normalization(outputs, training))
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training), name="outputs")
            with tf.variable_scope("d_conv3"):
                outputs = tf.layers.conv2d(outputs, filters=self.discriminator_layer[2], kernel_size=[5, 5],
                                           strides=(2, 2),
                                           padding='SAME')
                # outputs = tf.nn.elu(tf.layers.batch_normalization(outputs, training))
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training), name="outputs")
            with tf.variable_scope("d_conv4"):
                outputs = tf.layers.conv2d(outputs, filters=self.discriminator_layer[3], kernel_size=[5, 5],
                                           strides=(2, 2),
                                           padding='SAME')
                # outputs = tf.nn.elu(tf.layers.batch_normalization(outputs, training))
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training), name="outputs")
            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 2, name='outputs')

        # (4)reuseの設定
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return outputs


class DCGAN(object):
    """
    DCGANを定義するソースコード
    """

    def __init__(self, discriminator_layer=[], generator_layers=[], batch_size=64, image_inputs=None):
        """

        :param discriminator_layer: 識別機のレイヤを定義する
        :param generator_layers: 生成器のレイヤを定義する
        :param batch_size: バッチサイズ
        :param inputs: 入力
        """
        # (5)クラスの初期化
        self.generator = Generator(
            generator_layers,
            s_size=4
        )

        self.discriminator = Discriminator(
            discriminator_layer
        )
        self.batch_size = batch_size
        self.z_dim = 100
        self.random_inputs = tf.random_uniform((self.batch_size, self.z_dim), minval=-1.0, maxval=1.0)
        self.image_inputs = image_inputs
        self.build_loss_network()
        self.optimize()

    def build_loss_network(self):
        """
        ニューラルネットワークを構築する
        生成画像を0 , オリジナル画像を1とする。

        :return: 誤差
        """
        # (6)画像生成用ネットワークの作成
        generate_output = self.generator.build_network(self.random_inputs, training=True)
        discriminator_output = self.discriminator.build_network(generate_output, training=True)

        # (7)画像学習用ネットワークの作成
        discriminator_output_orig = self.discriminator.build_network(self.image_inputs, training=True)

        # (8)Generatorが育つターン用の誤差を生成
        g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=discriminator_output,
            labels=tf.ones(
                [self.batch_size], dtype=tf.int32
            )
        ))

        # (9)Discriminatorが育つターン用の誤差を生成
        # 学習用画像と1(本物)の誤差
        d_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=discriminator_output_orig,
            labels=tf.ones(
                [self.batch_size], dtype=tf.int32
            )
        ))

        # 生成画像と0(本物)の誤差に、上の誤差を加える
        d_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=discriminator_output,
            labels=tf.zeros(
                [self.batch_size], dtype=tf.int32
            )
        ))

        self.g_loss = g_loss
        self.d_loss = d_loss

    def optimize(self, learning_rate=0.0002, beta=0.5):
        """
        最適化手法の定義
        :return: None
        """
        # (10)最適化手法の定義
        generator_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta)
        discriminator_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta)
        g_minimize = generator_opt.minimize(self.g_loss, var_list=self.generator.variables)
        d_minimize = discriminator_opt.minimize(self.d_loss, var_list=self.discriminator.variables)

        self.g_minimize = g_minimize
        self.d_minimize = d_minimize

    def fit_step(self, sess, image_inputs):
        """
        1 stepごとに学習する

        :param sess: TensorFlowのsession
        :param image_inputs: 入力画像データ
        :return: generatorの誤差、識別機の誤差、TensorFlowのサマリ結果
        """
        # (11)1ステップ分の処理
        _, gloss = sess.run([self.g_minimize, self.g_loss], feed_dict={self.image_inputs: image_inputs})
        _, dloss = sess.run([self.d_minimize, self.d_loss], feed_dict={self.image_inputs: image_inputs})
        return gloss, dloss

    def sample_images(self, row=8, col=8, inputs=None):
        # (12)サンプル出力の処理
        if inputs is None:
            inputs = self.random_inputs
        images = self.generator.build_network(inputs, training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))
