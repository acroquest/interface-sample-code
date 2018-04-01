# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import cv2
import os
import argparse

id = 0
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, choices=["lfw_pos", "lfw_neg", "evaluate"])
args = parser.parse_args()

# 対象人物の画像ではない。 python extract_face.py -t lfw_neg
if args.type == "lfw_neg":
    folder_list = ["Tony_Blair", "Hugo_Chavez"]
    SRC_DIR_PATH = "./data/lfw/"
    DST_DIR_PATH = "./data/faces/negative"
    suffix = "neg"
# 対象人物の画像 python extract_face.py -t lfw_pos
elif args.type == "lfw_pos":
    folder_list = ["George_W_Bush"]
    suffix = "pos"
    SRC_DIR_PATH = "./data/lfw/"
    DST_DIR_PATH = "./data/faces/positive"

# 顔画像検出器を初期化する。
cascade_path = "INPUT YOUR FACE MODEL PATH"
cascade = cv2.CascadeClassifier(cascade_path)
# ファイルセットを作る。
for folder in folder_list:
    src_file_list = os.listdir(SRC_DIR_PATH + folder)
    for file in src_file_list:
        # 不要なファイルに判定処理を行わない。
        if file.startswith(".") or file.endswith(".txt"):
            continue

        try:
            # 画像の読み込み
            image = cv2.imread(os.path.join(SRC_DIR_PATH + folder, file))
            # 画像を読み込めなかった場合に処理を飛ばす。
            if image is None:
                continue

            if (len(image.shape)) == 2:
                image_gray = image
                continue
            else:
                print ("グレー・スケールの変換を開始する。{}".format(os.path.join(SRC_DIR_PATH, file)))
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print (e, file)
            continue

        # 画像のスケール調整
        scale_height = 512.0 / image.shape[0]
        scale_width = 512.0 / image.shape[1]

        resize_image_gray = cv2.resize(image_gray, (512, 512))
        print ("グレー・スケールの変換を完了する。")

        minsize = (int(resize_image_gray.shape[0] * 0.1), int(resize_image_gray.shape[1] * 0.1))

        try:
            print ("顔画像検出を行う。")
            # ②顔画像を検出する。
            facerect = cascade.detectMultiScale(resize_image_gray, scaleFactor=1.1, minNeighbors=1, minSize=minsize)
            print ("顔画像検出を完了する。")
            if len(facerect) == 0 or len(facerect) > 1:
                print ("顔が検出できませんでした。")
                continue
            for rect in facerect:
                min_height = int(rect[0] / scale_height)
                min_width = int(rect[1] / scale_width)
                max_height = int(rect[2] / scale_height) + min_height
                max_width = int(rect[3] / scale_width) + min_width

                print(min_height, max_height, min_width, max_width)
                # 画像を保存する。
                cv2.imwrite(os.path.join(DST_DIR_PATH, "image_{}_{}.jpg".format(suffix, str(id))),
                            image[min_height:max_height, min_width:max_width])
                id += 1
        except Exception as e:
            print (e)
