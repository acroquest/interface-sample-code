# coding: utf-8

import argparse
import datetime
import os
import shutil
import time
from PIL import Image
import math
import collections

from tuned_mobilenet import TunedMobileNet
from tuned_mobilenet import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

# picameraのimport。失敗した場合はカメラなしで動くようにする。
try:
    import picamera

    without_camera = False
except ImportError:
    print('# This device does not have a camera. Install camera and picamera module.')
    print('# (Start up in test mode.)')
    without_camera = True

from word_tree import WordTree
from twitter_post import Twitter

# 画像判定時、上位何個までを判定対象とするか。
# 1つの画像に複数の物が写っていた場合、上位からこの数の分だけ対象の物なのか判定する。
PREDICT_NUM = 10

# カメラの解像度
CAMERA_RESOLUTION = (1024, 768)
# 部分画像で切り取って判定する際の縦横の長さ
CROP_LEN = int(CAMERA_RESOLUTION[1] * 0.5)
# 部分画像どうしが重なる幅
MARGIN = int(CROP_LEN * 0.5)
# 部分画像を切り取る際にずらしていく幅
STEP_LEN = CROP_LEN - MARGIN
# 画像判定のバッチ数
BATCH = 1


class DetectionCamera:
    """
    検知カメラ。指定した物体をカメラが写したときに画像を保存することを繰り返す。
    """

    def __init__(self):
        global test_mode
        """
        コンストラクタ。引数をパースする。また、カメラ、画像判定モデルを初期化する。
        """

        # (1)引数をパースする
        arg_parser = argparse.ArgumentParser(description='Detecting Camera')
        arg_parser.add_argument('target_name',
                                help='Target name to detecting.')
        arg_parser.add_argument('-i', '--interval', type=int, required=False, default=60,
                                help='Camera shooting interval[sec] (default:60)')
        arg_parser.add_argument('-p', '--period', type=int, required=False, default=0,
                                help='Shooting period[min]. Infinite if 0(default).')
        arg_parser.add_argument('-d', '--directory', required=False, default='./',
                                help='Output directory (default:current dir)')
        arg_parser.add_argument('-l', '--list', action='store_true', default=False,
                                help='List target names.')
        arg_parser.add_argument('-t', '--tweet', action='store_true', default=False,
                                help='Tweet detected.(default:Off)')
        arg_parser.add_argument('-e', '--test', action='store_true', default=False,
                                help='Test mode.')
        args = arg_parser.parse_args()

        # パースした引数の内容を変数で覚えておく。
        self._target_name = args.target_name
        self._output_directory = args.directory
        self._period_sec = args.period * 60
        self._interval_sec = args.interval
        self._list_option = args.list
        self._tweet_mode = args.tweet
        self._test_mode = args.test

        # 引数"-l"が指定されていたら、ImageNetの単語ツリーを全表示して終了。
        self._word_tree = WordTree()
        if self._list_option:
            self._word_tree.print_all_tree()
            exit(0)

        # カメラの無い環境であればテストモードを強制的にONにする。
        if without_camera:
            self._test_mode = True

        # (2)ターゲット名が、ImageNetの単語ツリー内に存在するか検索。
        # 存在すれば、その単語配下の全単語について単語IDを検索し、保持する。
        # 存在しなければプログラム終了。
        word_id = self._word_tree.find_id(self._target_name)
        if word_id is None:
            print('{0} is not ImageNet word. Retry input.'.format(self._target_name))
            print('(Search available words for -l options.)')
            exit(0)

        self._target_ids = self._word_tree.list_descendants(word_id)

        # (3)引数の内容を表示。
        print(('Camera started.\n'
               '  - Target name     : {0}\n'
               '  - Interval        : {1}\n'
               '  - Shooting period : {2}\n'
               '  - Output directory: {3}\n'
               '  - Tweet mode      : {4}\n'
               '  - Test mode       : {5}\n'
               ).format(self._target_name,
                        str(self._interval_sec) + '[sec]',
                        str(args.period) + '[min]' if self._period_sec != 0 else 'infinite',
                        self._output_directory,
                        self._tweet_mode,
                        self._test_mode)
              )

        # (4)画像判定モデルを初期化。
        print('Start model loading...')
        self._model = TunedMobileNet(weights=None)
        self._model.load_weights('./model.h5')
        print('Finish.')

        # (5)Twitter接続を初期化。
        if self._tweet_mode:
            self._twitter = Twitter()

    def run(self):
        """
        カメラ処理。撮影、画像判定、ファイル保存を繰り返す。
        """

        print('Start shooting.')

        # (1)処理開始時間を保持。このあと経過時間を計算するため。
        start_time = datetime.datetime.now()

        # (2)カメラを初期化。テストモードならスキップ。
        if not self._test_mode:
            camera = picamera.PiCamera()
            camera.resolution = CAMERA_RESOLUTION

        # 以下、撮影期間が終了するまで繰り返し
        counter = 1
        while True:
            now_time = datetime.datetime.now()
            # (3)カメラ画像を取得。
            if not self._test_mode:
                camera.capture('tmp.jpg')

            # (4)目的のものが写っているか判定。
            matched_target = self._match_target_image('tmp.jpg')
            print('[{0}] {1} - Detect {2}.'.format(counter, now_time.strftime('%Y/%m/%d %H:%M:%S'), matched_target))

            if matched_target is not None:
                # (5)写っていたら画像をTwitterに投稿。Tweet modeがONの場合のみ。
                if self._tweet_mode:
                    self._twitter.post(matched_target, 'tmp.jpg')

                # (6)画像を保存。Tweet modeがOFFの場合。
                # 上で一時的に保存したtmp.jpgを別名で保存する形で実施する。
                # ファイルパスは"{指定されたディレクトリ}/{年月日}_{時分秒}_{物体名}.jpg"とする。
                else:
                    now_str = now_time.strftime('%Y%m%d_%H%M%S')
                    file_name = '{0}_{1}.jpg'.format(now_str, matched_target)
                    file_path = os.path.join(self._output_directory, file_name)

                    shutil.copy('tmp.jpg', file_path)

            # (7)経過時間をチェック。オプションで指定した時間異常が過ぎていたら処理を停止する。
            elapsed_time = now_time - start_time
            elapsed_sec = elapsed_time.total_seconds()
            if self._period_sec != 0 and elapsed_sec < self._period_sec:
                print('Shooting period ({0} sec) ended.'.format(self._period_sec))
                break

            # (8)インターバル期間の分、待ち。
            wait_time = self._interval_sec - (elapsed_sec % self._interval_sec)
            time.sleep(wait_time)
            counter += 1

    def _match_target_image(self, img_path):
        """
        引数の画像に何が写っているか判定し、検知対象であればその物体名を返す。
        :param img_path: 確認したい画像
        :return: 物体名　検知対象でなければNone
        """

        # (1) 画像の内容を配列化
        img_list = []
        org_img = Image.open(img_path)
        img_list.append(org_img)

        # (2) タイル状に切り取り
        # 小さく写っている物体も検知できるよう、元画像を小さくタイル状に切り取った画像群を作る。
        # タイルは縦横がCROP_LENの正方形とする。
        # 境目に物体があるときも対応できるようMARGIN分の幅で重なりが出るように切り取る。
        org_width, org_height = org_img.size
        crop_left_list = self._split_step(org_width)
        crop_top_list = self._split_step(org_height)

        for left in crop_left_list:
            for top in crop_top_list:
                right = left + CROP_LEN
                bottom = top + CROP_LEN
                tile_img = org_img.crop((left, top, right, bottom))
                img_list.append(tile_img)

                # デバッグ用。切り取った画像を出力する。
                #file_name = 'crop_{0}_{1}.jpg'.format(left, top)
                #tile_img.save(file_name)

        # (3) 全画像を判定
        # 元画像＋タイル画像群の全てについて、画像判定モデルを用いて、目標の物体が写っているか判定する。
        preds_list = self._predict_image(img_list, top=PREDICT_NUM)

        # (4) 判定結果のチェック
        # 全画像分の判定結果について、写っている物体が検知したい対象かをチェックする。
        # 検知対象の物体であれば、その名前を記録する。
        pred_name_list = []

        for index, preds in enumerate(preds_list):
            for pred_id, pred_name, score in preds:
                if pred_id in self._target_ids:
                    pred_name_list.append(pred_name)

        # (5) ランキングTopの取得
        # 記録した名前でランキングを作り、1位のものの名前を返す。
        pred_name_ranking = collections.Counter(pred_name_list)

        top_name_count = pred_name_ranking.most_common(1)
        if len(top_name_count) == 0:
            return None
        else:
            ret, _ = top_name_count[0]
            return ret

    def _split_step(self, target):
        """
        引数targetの長さを、CROP_LENずつ、MARGIN分の幅で重なりが出るように区切ったときの
        始点の座標を返す
        :param target: 区切る対象の長さ
        :return: 座標(int)の配列
        """
        end_point = target - CROP_LEN
        step_num = int(math.floor(end_point / STEP_LEN))
        ret = [point * STEP_LEN for point in range(step_num + 1)]
        ret.append(end_point)
        return ret

    def _predict_image(self, img_list, top=10):
        """
        画像判定モデルを用いて、何が写っているか判定する。
        :param img_list: 判定画像のリスト
        :param top: 上位何位まで判定するか
        :return: 検知結果　(単語ID, 名称, 判定スコア)のタプルがtopで指定された数のリストになる。
        """
        # 判定画像を配列化する。
        img_array_list = []
        for img in img_list:
            resized = img.resize((224, 224))
            img_array = image.img_to_array(resized)
            img_array = preprocess_input(img_array)
            img_array_list.append(img_array)

        # バッチサイズ分ずつ取り出し、判定にかける。
        ret = []
        for index in range(0, len(img_array_list), BATCH):
            x = np.array(img_array_list[index:index + BATCH])
            preds = self._model.predict(x)
            decoded = decode_predictions(preds, top=top)
            ret.extend(decoded)

        return ret


if __name__ == "__main__":
    camera = DetectionCamera()
    camera.run()
