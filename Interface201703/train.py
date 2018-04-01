# coding:utf-8
import tensorflow as tf
import numpy as np
import time
import os
from model import define_model, define_loss, training, evaluation
from util import input_image

BATCH_SIZE = 16
NUM_CHANNELS = 1
IMAGE_SIZE = 28
SEED = 82
NUM_LABELS = 2
EPOCHS = 10


def read_images(directory_path):
    """
    読み込み画像

    :param directory_path: 読み込むディレクトリ
    :return: ファイル一覧
    """
    face_images = []
    # フォルダの中の画像を読み込む
    for file in os.listdir(directory_path):
        if file.startswith("."):
            continue
        filepath = os.path.join(directory_path, file)
        image = input_image(filepath, IMAGE_SIZE)
        if image is None:
            continue

        face_images.append(image)
    return face_images


def input_faces():
    """
    顔画像を読み込みます。

    :return: 入力された画像のリストとラベルのペア
    """
    positive_datapath = "./data/faces/positive"
    negative_datapath = "./data/faces/negative"

    # ブッシュ画像を読み込む
    face_images = read_images(positive_datapath)
    # ラベルを1とする。
    labels = [1 for _ in range(len(face_images))]

    # ブッシュではない画像を読み込む。
    negative_face_images = read_images(negative_datapath)
    face_images += negative_face_images
    # ラベルを0とする。
    labels += [0 for _ in range(len(negative_face_images))]

    # 0-1へ画像をスケーリングする。
    face_images = np.array(face_images, dtype=np.float32) / 255.0

    return face_images, np.array(labels)


def main():
    faces, label = input_faces()
    n_class = len(list(set(label)))
    N_faces = len(faces)

    print (n_class), N_faces

    with tf.Graph().as_default():
        # 入力画像のプレースホルダー
        images = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])

        # ラベル 0: 1:
        labels = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

        # モデルの定義
        model = define_model(images=images)
        # 誤差関数
        loss = define_loss(model, labels=labels)

        # 誤差を保存、TensorBoardで可視化できる。
        tf.scalar_summary("loss", loss)
        # 学習の演算を定義する。
        train_op = training(loss, learning_rate=0.01)
        # 評価方法を定義、正解数を計算する。
        evaluation_op = evaluation(model, labels)
        # TensorBoardへの記載
        tf.image_summary('images', images, max_images=100)

        summary = tf.merge_all_summaries()
        saver = tf.train.Saver()

        # 変数初期化を行う。
        with tf.Session() as session:
            # 全ての変数を初期化する。
            tf.initialize_all_variables().run()
            # 計算グラフを定義
            summary_writer = tf.train.SummaryWriter("./log", session.graph)

            count = 0

            # 学習を行う。 何周学習を行うか
            for epoch in range(EPOCHS):
                perm = np.random.permutation(N_faces)

                loss_value = 0.0

                # バッチ学習を行う。
                for index, step in enumerate(range(0, N_faces - BATCH_SIZE, BATCH_SIZE)):
                    start_time = time.time()
                    batch_start = step
                    batch_end = step + BATCH_SIZE

                    feed_dict = {
                        images: faces[perm[batch_start:batch_end]],
                        labels: label[perm[batch_start:batch_end]]
                    }

                    # ①演算を行う。
                    _, loss_value, eval_value = session.run([train_op, loss, evaluation_op], feed_dict=feed_dict)

                    duration = time.time() - start_time
                    count += 1

                    print('Step %d: loss = %.2f (%.3f sec)' % (
                        index + epoch * N_faces // BATCH_SIZE, loss_value, duration))
                    summary_str = session.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, index + epoch * N_faces // BATCH_SIZE)
                    summary_writer.flush()

                # 学習が一周完了した段階で出力する。
                print (epoch, loss_value)

            # ②学習結果を保存する。
            saver.save(session, "model.ckpt")


if __name__ == '__main__':
    main()
