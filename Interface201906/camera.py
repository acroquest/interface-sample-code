# coding: utf-8

import argparse
import datetime
import os
import shutil
import time

# picameraのimport。失敗した場合はカメラなしで動くようにする。
try:
    import picamera

    without_camera = False
except ImportError:
    print('# This device does not have a camera. Install camera and picamera module.')
    print('# (Start up in test mode.)')
    without_camera = True

from detector import ObjectDetector
from twitter_post import Twitter

# 画像判定時の、スコアの閾値。単位は"%"
SCORE_THRESHOLD = 10

# カメラの解像度
CAMERA_RESOLUTION = (1024, 768)


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
                                help='Target name to detecting. Split by commas.')
        arg_parser.add_argument('-x', '--exclude', default=None,
                                help='Target excluded from detection. Split by commas.')
        arg_parser.add_argument('-i', '--interval', type=int, default=60,
                                help='Camera shooting interval[sec] (default:60)')
        arg_parser.add_argument('-p', '--period', type=int, default=0,
                                help='Shooting period[min]. Infinite if 0(default).')
        arg_parser.add_argument('-d', '--directory', default='./',
                                help='Output directory (default:current dir)')
        arg_parser.add_argument('-l', '--list', action='store_true', default=False,
                                help='List target names.')
        arg_parser.add_argument('-s', '--saveimage', action='store_true', default=False,
                                help='Save image mode.(default:Off)')
        arg_parser.add_argument('-t', '--tweet', action='store_true', default=False,
                                help='Tweet detected.(default:Off)')
        arg_parser.add_argument('-e', '--test', action='store_true', default=False,
                                help='Test mode.(default:Off)')
        args = arg_parser.parse_args()

        # パースした引数の内容を変数で覚えておく。
        self._target_names = args.target_name.split(',')
        self._exclude_names = args.exclude.split(',') if args.exclude is not None else []
        self._output_directory = args.directory
        self._period_sec = args.period * 60
        self._interval_sec = args.interval
        self._list_option = args.list
        self._save_image_mode = args.saveimage
        self._tweet_mode = args.tweet
        self._test_mode = args.test

        # 検知器を用意し、利用可能なターゲット名称をそこから取得する。
        self._detector = ObjectDetector()
        available_names = self._detector.list_names()

        # 引数"-l"が指定されていたら、利用可能なターゲット名称をリストアップして終了
        if self._list_option:
            print('Available names:')
            print(', '.join(available_names))
            exit(0)

        # (2)ターゲット名が、利用可能なターゲット名称に含まれるかチェック。
        # なければプログラム終了。
        for name in self._target_names + self._exclude_names:
            if name not in available_names:
                print('"{0}" is not available word. Retry input.'.format(name))
                print('(Search available words for -l options.)')
                exit(0)

        # カメラの無い環境であればテストモードを強制的にONにする。
        if without_camera:
            self._test_mode = True

        # (3)引数の内容を表示。
        print(('Camera started.\n'
               '  - Target name     : {0}\n'
               '  - Exclude name    : {1}\n'
               '  - Interval        : {2}\n'
               '  - Shooting period : {3}\n'
               '  - Output directory: {4}\n'
               '  - Save image mode : {5}\n'
               '  - Tweet mode      : {6}\n'
               '  - Test mode       : {7}\n'
               ).format(','.join(self._target_names),
                        ','.join(self._exclude_names),
                        str(self._interval_sec) + '[sec]',
                        str(args.period) + '[min]' if self._period_sec != 0 else 'infinite',
                        self._output_directory,
                        self._save_image_mode,
                        self._tweet_mode,
                        self._test_mode)
              )

        # (4)画像判定モデルを初期化。
        print('Start model loading...')
        self._detector.open()
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

            #camera.hflip = True
            #camera.vflip = True
            '''
            # カメラ設定用のオプション。必要に応じてコメント外にコピーすること。
            #https://picamera.readthedocs.io/en/release-0.2/api.html
            camera.sharpness = 0 # -100 to 100
            camera.contrast = 0 # -100 to 100
            camera.brightness = 50 # 0 to 100
            camera.saturation = 0
            camera.ISO = 0
            camera.video_stabilization = False
            camera.exposure_compensation = 0
            camera.exposure_mode = 'auto'
            camera.meter_mode = 'average'
            camera.awb_mode = 'auto'
            camera.image_effect = 'none'
            camera.color_effects = None
            camera.rotation = 0
            camera.hflip = False
            camera.vflip = False
            camera.crop = (0.0, 0.0, 1.0, 1.0)

            '''

        # 以下、撮影期間が終了するまで繰り返し
        counter = 1
        while True:
            now_time = datetime.datetime.now()
            # (3)カメラ画像を取得。
            if not self._test_mode:
                camera.capture('tmp.jpg')

            # (4)目的のものが写っているか判定。
            matched_name, matched_box = self._match_target_image('tmp.jpg', threshold=SCORE_THRESHOLD)
            duration = datetime.datetime.now() - now_time
            print('[{0}] {1} - Detect {2}. ({3:.1f}[sec])'.format(counter, now_time.strftime('%Y/%m/%d %H:%M:%S'),
                                                                  matched_name, duration.total_seconds()))

            if matched_name is not None:
                # (5)写っていたら画像をTwitterに投稿。Tweet modeがONの場合のみ。
                if self._tweet_mode:
                    self._twitter.post(matched_name, 'tmp.jpg')

                # (6)画像を保存。
                # 元画像は"{指定されたディレクトリ}/{年月日}_{時分秒}_{物体名}.jpg"とし、
                # 検知枠の付いた画像は末尾を"_result.jpg"として保存する。
                if self._save_image_mode:
                    now_str = now_time.strftime('%Y%m%d_%H%M%S')

                    original_file_name = '{0}_{1}.jpg'.format(now_str, matched_name)
                    original_file_path = os.path.join(self._output_directory, original_file_name)
                    shutil.copy('tmp.jpg', original_file_path)

                    result_file_name = '{0}_{1}_result.jpg'.format(now_str, matched_name)
                    result_file_path = os.path.join(self._output_directory, result_file_name)
                    shutil.copy('result.jpg', result_file_path)

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

    def _match_target_image(self, img_path, threshold=0):
        """
        引数の画像に何が写っているか判定し、検知対象であればその物体名を返す。
        :param img_path: 確認したい画像
        :return: 物体名　検知対象でなければNone
        """

        # (1) 画像を判定
        results = self._detector.detect(img_path)

        # (2) 判定結果のチェック
        # 全画像分の判定結果について、写っている物体がターゲット名と同じかチェックし、
        # スコアが最高のものだけを記録する。
        # またターゲット除外名ものがあれば、即座に中断する。
        max_score_name = None
        max_score_box = None
        max_score = 0

        for name, score, box in zip(results['names'], results['scores'], results['boxes']):

            # ターゲット除外名であれば即座に終了。
            if name in self._exclude_names:
                print('Exclude name "{0}" detected.'.format(name))
                return None, None

            # ターゲットであれば、今までの最高スコアと比較し、最高スコアを更新したら記録する。
            if name in self._target_names:

                if score > max_score:
                    max_score = score
                    max_score_name = name
                    max_score_box = box

        # 最高スコアの物を返す。
        return max_score_name, max_score_box


if __name__ == "__main__":
    camera = DetectionCamera()
    camera.run()
