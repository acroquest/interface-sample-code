# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import cv2
import numpy as np


def input_image(filepath, size):
    """
    画像の入力

    :param filepath: ファイルパス
    :return: 画像
    """
    img = cv2.imread(filepath)
    if len(img.shape) == 2:
        return None

    # グレー画像に変換
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 画像をリサイズする。
    resize_image = cv2.resize(gray_image, (size, size))
    # 28 x 28 → 28 x 28 x 1
    expand_image = np.expand_dims(resize_image, 2)

    return expand_image
