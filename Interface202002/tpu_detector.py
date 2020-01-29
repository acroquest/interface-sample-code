from PIL import Image
from PIL import ImageDraw
from edgetpu.detection.engine import DetectionEngine

from lap_timer import measure_lap

# https://coral.withgoogle.com/models/ から、"MobileNet SSD v2 (COCO)"のEdge TPU modelとLabelsをダウンロードし、
# 配置先のパスを以下に記述する
MODEL_PATH = './mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
LABEL_PATH = './coco_labels.txt'


class TpuDetector:

    def __init__(self):
        pass

    def open(self):

        # Initialize engine.
        self.engine = DetectionEngine(MODEL_PATH)

        # ラベルファイルを読み込み、{ID:名前}の辞書に変換し、保持する。
        with open(LABEL_PATH, 'r') as f:
            lines = f.readlines()

        self.labels = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            id = int(pair[0])
            name = pair[1].strip()
            self.labels[id] = name

    def detect(self, image_path):

        # 引数パスの画像を読み込む。
        img = Image.open(image_path)

        # 画像を物体検知にかける。
        _ = measure_lap()
        detection_result = self.engine.DetectWithImage(
            img, threshold=0.9,
            keep_aspect_ratio=True,
            relative_coord=False,
            top_k=10)

        measure_lap(show=True)

        # 検知位置を元画像上に描画する。
        # また返り値の形式をdetector.pyと合うよう変換する。
        draw = ImageDraw.Draw(img)
        width, height = img.size

        result = {
            'classes': [],
            'names': [],
            'boxes': [],
            'scores': []
        }

        for detected_object in detection_result:
            name = self.labels[detected_object.label_id]

            # 画像に物体の検知枠を描画する。
            box = detected_object.bounding_box.flatten().tolist()
            draw.rectangle(box, outline='red')

            # 正規化座標に変換する。
            xmin, ymin, xmax, ymax = box
            box_n = (ymin / height, xmin / width, ymax / height, xmax / width)

            result['classes'].append(detected_object.label_id)
            result['boxes'].append(box_n)
            result['names'].append(name)
            result['scores'].append(detected_object.score)

        img.save('result.jpg')

        return result

    def close(self):
        pass


if __name__ == '__main__':
    detector = TpuDetector()
    detector.open()
    detector.detect('tmp.jpg')
    detector.close()
