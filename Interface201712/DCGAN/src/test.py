# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import argparse
import tensorflow as tf
from dcgan import DCGAN
import matplotlib

matplotlib.use("Agg")

SIZE = 64


def test(filepath):
    """
    判定を行う。
    """
    with tf.Session() as sess:
        # (1)モデルの復元
        batch_size = 64
        dcgan = DCGAN(
            generator_layers=[1024, 512, 256, 128],
            discriminator_layer=[64, 128, 256, 512],
            batch_size=batch_size,
            image_inputs=tf.placeholder(tf.float32, [batch_size, SIZE, SIZE, 3]),
        )
        sess.run(tf.global_variables_initializer())
        g_saver = tf.train.Saver(dcgan.generator.variables)
        d_saver = tf.train.Saver(dcgan.discriminator.variables)
        g_saver.restore(sess=sess, save_path="../data/models/g_saver.ckpg")
        d_saver.restore(sess=sess, save_path="../data/models/d_saver.ckpg")

        # (2)画像の生成
        sample_z = tf.random_uniform([dcgan.batch_size, dcgan.z_dim], minval=-1.0, maxval=1.0)
        images = dcgan.sample_images(8, 8, inputs=sample_z)
        with open(filepath, 'wb') as f:
            f.write(sess.run(images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="image.jpg")

    args = parser.parse_args()

    test(args.output)
