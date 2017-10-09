# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import matplotlib
import numpy as np
import tensorflow as tf
from dcgan import DCGAN

matplotlib.use("Agg")
import glob
from util import preprocess_img

SIZE = 64


def load_images():
    """
    画像を読み込む

    :return:
    """
    # (1)画像のロード
    filepaths = glob.glob("../data/input/training/*.jpg") + glob.glob("../data/input/validation/*.jpg") + glob.glob(
        "../data/input/evaluation/*.jpg")

    images = [preprocess_img(filepath=filepath, size=SIZE) for filepath in filepaths];
    print("{0}枚の画像を取り込みました。".format(len(images)))
    return np.array(images, dtype=np.float32)


def train():
    """
    学習データを構築する。
    """
    # 画像をデータセットから読み込む
    imgs = load_images()
    with tf.Session() as sess:
        # (3)DCGANネットワークの生成
        batch_size = 64
        dcgan = DCGAN(
            generator_layers=[1024, 512, 256, 128],
            discriminator_layer=[64, 128, 256, 512],
            batch_size=batch_size,
            image_inputs=tf.placeholder(tf.float32, [batch_size, SIZE, SIZE, 3]),
        )
        sess.run(tf.global_variables_initializer())

        # (4)ファイル保存の準備
        g_saver = tf.train.Saver(dcgan.generator.variables)
        d_saver = tf.train.Saver(dcgan.discriminator.variables)

        maxstep = 10000
        N = len(imgs)

        # (5)サンプル出力の準備
        sample_z = tf.random_uniform([dcgan.batch_size, dcgan.z_dim], minval=-1.0, maxval=1.0)
        images = dcgan.sample_images(8, 8, inputs=sample_z)

        os.makedirs('../data/generated_images/', exist_ok=True)

        # (6)学習
        for step in range(maxstep):
            permutation = np.random.permutation(N)
            imgs_batch = imgs[permutation[0:batch_size]]
            g_loss, d_loss = dcgan.fit_step(sess=sess, image_inputs=imgs_batch)

            # 100 stepごとに学習結果を出力する。
            if step % 100 == 0:
                filename = os.path.join('../data/', "generated_images", '%05d.jpg' % step)
                with open(filename, 'wb') as f:
                    f.write(sess.run(images))
                print("Generator loss: {} , Discriminator loss: {}".format(g_loss, d_loss))

        # (7)学習済みモデルのファイル保存
        os.makedirs('../data/models/', exist_ok=True)
        g_saver.save(sess=sess, save_path="../data/models/g_saver.ckpg")
        d_saver.save(sess=sess, save_path="../data/models/d_saver.ckpg")


if __name__ == '__main__':
    train()
