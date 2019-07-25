# coding: utf-8

import glob
import shutil
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# モデルを格納しているディレクトリ名
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
# モデルディレクトリ内の、実際のモデルファイルのパス
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_FROZEN_GRAPH = 'train/model/frozen_inference_graph.pb'

# ラベルファイルのパス
PATH_TO_LABELS = '../models/research/object_detection/data/mscoco_label_map.pbtxt'


# PATH_TO_LABELS = 'train/pascal_label_map.pbtxt'


class ObjectDetector:
    """
    Object Detection APIを用いて、画像から物体を検知する。
    """

    def __init__(self):

        # self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

    def open(self):

        # モデルファイルを読み込む。
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def detect(self, image_path):

        # 引数パスの画像を読み込み、配列に変換する。
        image = Image.open(image_path)
        image_np = self._load_image_into_numpy_array(image)

        # 画像配列を物体検知にかける。
        result = self._detect_single_image(image_np)

        # 検知結果の名前、スコア、検知位置を元画像上に描画した画像を作る。
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            result['boxes'],
            result['classes'],
            result['scores'],
            self.category_index,
            instance_masks=result.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        # 作った画像をファイルに保存する。
        Image.fromarray(image_np).save('result.jpg')

        # 'classes'キーにある分類結果のインデックス値を、名前に置き換えたリストを作り、返却内容に加える。
        result['names'] = [self.category_index[class_idx]['name'] for class_idx in result['classes']]

        return result

    def list_names(self):

        names = [category['name'] for class_idx, category in sorted(self.category_index.items())]
        return names

    def _load_image_into_numpy_array(self, image):
        """
        引数画像を配列にして返す
        :param image: 画像
        :return: 画像の配列
        """
        (im_width, im_height) = image.size

        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def _detect_single_image(self, image):
        """
        実際の物体検知処理。引数画像から物体を検知し、物体ごとの名前、スコア、位置を返す。
        :param image: 画像
        :return: 検知結果の辞書
        """
        with self.detection_graph.as_default():
            with tf.Session() as sess:

                # 読み込まれたモデル構造から、出力用のテンソルを得る。
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)

                # 画像入力用のテンソルを得る。
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # 入出力のテンソルと検知対象の画像を渡して、実際の物体検知を実施する。
                result_array = sess.run(tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(image, 0)})

                # 物体検知の結果は、複数画像を渡されたときにも対応できるよう配列になっている。
                # 今回は画像1枚しか渡していないので、配列から第一要素だけ取り出し、
                # 返却用の辞書に詰める。
                num = int(result_array['num_detections'][0])
                result = {
                    'num': num,
                    'classes': result_array['detection_classes'][0].astype(np.uint8)[:num],
                    'boxes': result_array['detection_boxes'][0][:num],
                    'scores': result_array['detection_scores'][0][:num],
                }

        return result

    def detect_multi_image(self, image_dir):

        image_list = glob.glob(image_dir + '/*.jpg')

        for image_path in image_list:
            print(image_path)

            self.detect(image_path)

            head, ext = path.splitext(image_path)

            shutil.move('./result.jpg', head + '_result' + ext)


if __name__ == '__main__':
    detector = ObjectDetector()
    detector.open()
    detector.detect_multi_image('test/')
