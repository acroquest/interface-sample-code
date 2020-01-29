# coding: utf-8

import glob
import shutil
import sys
from os import path


def detect_multi_image(detector, image_dir):
    image_list = glob.glob(image_dir + '/*.jpg')

    total_time = 0

    for index, image_path in enumerate(image_list, 1):
        if '_result' in image_path:
            continue
        print(image_path)

        detector.detect(image_path)

        head, ext = path.splitext(image_path)

        shutil.move('./result.jpg', head + '_result' + ext)


if __name__ == '__main__':

    detector_type = 'cpu'
    if len(sys.argv) == 1:
        detector_type = 'cpu'
    else:
        detector_type = sys.argv[1]

    if detector_type == 'cpu':
        from detector import ObjectDetector

        detector = ObjectDetector()
    else:
        from tpu_detector import TpuDetector

        detector = TpuDetector()

    detector.open()
    detect_multi_image(detector, 'test/')
    detector.close()
