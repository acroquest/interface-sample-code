# coding: utf-8
import glob
import shutil
from os import path

import cv2


class FaceDetector(object):

    def __init__(self):
        if not path.exists('haarcascade_frontalcatface_extended.xml'):
            raise Exception('Get xml file for detector.')

        # 顔検出器を初期化する。
        self.face_ext_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

    def detect(self, img):
        """
        与えられた画像から顔を検出する。検出結果はface.jpgに出力する。
        :param img: 画像配列
        :return: 検知されたらTrue
        """
        # グレースケールに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 顔を検知する
        faces = self.face_ext_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

        # 検出した顔部分を青い枠で囲む
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imwrite('./face.jpg', img)

        # 検出した顔があればTrueを返す
        face_count = len(faces)
        return face_count > 0

    def contains_face(self, image_path, box):
        """
        image_pathのファイルの画像の、boxで囲まれた位置に顔が含まれるか返す。
        :param image_path: 画像パス
        :param box: 検出する位置の矩形座標 (xmin, xmax, ymin, ymax)のタプル
        :return: 含まれればTrue
        """
        image = cv2.imread(image_path)

        xmin, xmax, ymin, ymax = box

        target_image = image[ymin:ymax, xmin:xmax]

        contains = self.detect(target_image)
        return contains

    def detect_multi_image(self, image_dir):
        """
        image_dirディレクトリに含まれるjpg画像全てで顔検出する。結果は "元ファイル名_result.jpg" として出力する。
        :param image_dir: 対処画像を含むディレクトリ
        """

        # ディレクトリ内の全jpg画像のパスを取得する
        image_list = glob.glob(image_dir + '/*.jpg')

        # 画像ごとに顔を検出し、結果画像を"元ファイル名_result.jpg"として出力する
        for image_path in image_list:
            print(image_path)

            image = cv2.imread(image_path)

            self.detect(image)

            head, ext = path.splitext(image_path)

            shutil.move('./face.jpg', head + '_result' + ext)


if __name__ == '__main__':
    detector = FaceDetector()
    detector.detect_multi_image('testimage/')
