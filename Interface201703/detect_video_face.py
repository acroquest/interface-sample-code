# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
from model import define_model
import cv2

BATCH_SIZE = 1
NUM_CHANNELS = 1
IMAGE_SIZE = 28

cap = cv2.VideoCapture(0)
cascade_path = "INPUT YOUR FACE MODEL PATH"
cascade = cv2.CascadeClassifier(cascade_path)

with tf.Graph().as_default():
    # 入力画像のプレースホルダ
    images = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    # モデル定義
    model = define_model(images=images)
    # 出力定義
    softmax_op = tf.nn.softmax(model)
    # セーバーの初期化
    saver = tf.train.Saver()
    # 変数初期化を行う。
    with tf.Session() as session:
        # モデルの読み出し
        saver.restore(session, "./model.ckpt")

        # ユーザが割り込むまで、実行し続ける。
        while True:
            response, frame = cap.read()
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale_height = 512.0 / image_gray.shape[0]
            scale_width = 512.0 / image_gray.shape[1]
            resize_image_gray = cv2.resize(image_gray, (512, 512))

            minsize = (int(resize_image_gray.shape[0] * 0.1), int(resize_image_gray.shape[1] * 0.1))
            facerect = cascade.detectMultiScale(resize_image_gray, scaleFactor=1.1, minNeighbors=1, minSize=minsize)

            # 検出領域が1領域の場合に実行する。
            if len(facerect) == 1:
                for rect in facerect:
                    min_height = int(rect[0] / scale_height)
                    min_width = int(rect[1] / scale_width)
                    max_height = int(rect[2] / scale_height) + min_height
                    max_width = int(rect[3] / scale_width) + min_width

                    print(min_height, max_height, min_width, max_width)

                    face_img = image_gray[min_height:max_height, min_width:max_width]
                    resized_face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))

                    feed_dict = {
                        images: np.array([np.expand_dims(resized_face_img, 2)], dtype=np.float32)
                    }
                    # 結果
                    value = session.run(softmax_op, feed_dict=feed_dict)
                    print (value)
                    # 矩形の色指定を行う
                    if value[0][0] > 0.5:
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)
                    # 顔検出領域を描画する。
                    cv2.rectangle(frame, (min_width, min_height), (max_width, max_height), color, 10)
                    k = cv2.waitKey(1)

            cv2.imshow("face camera", frame)
