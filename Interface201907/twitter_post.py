# coding:utf-8
import json

from requests_oauthlib import OAuth1Session

# Twitterにテキスト、およびメディアファイルをアップロードするためのエンドポイント
URL_TEXT = 'https://api.twitter.com/1.1/statuses/update.json'
URL_MEDIA = 'https://upload.twitter.com/1.1/media/upload.json'

# Twitter API認証用のアクセスキーとトークン。
# 自身で https://apps.twitter.com/ にて取得したものを以下の4変数に設定すること。
# ※記述したアクセスキーとトークンの内容を一般公開しないように注意！
CONSUMER_KEY = 'XXXXXXXX'
CONSUMER_SECRET = 'XXXXXXXX'
ACCESS_TOKEN = 'XXXXXXXX'
ACCESS_TOKEN_SECRET = 'XXXXXXXX'


class Twitter(object):
    def __init__(self):

        # Twitter APIへ認証し、接続する。
        self.twitter = OAuth1Session(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        print('Twitter connected.')

    def post(self, message, image_path):
        """
        引数のmessageを、image_pathの画像付きでTwitterに投稿する。
        :param message: メッセージ文字列
        :param image_path: 画像パス
        :return: 成功すればTrue
        """

        # 画像をTwitterにアップロードする。
        # アップロードに失敗した場合は終了。
        files = {'media': open(image_path, 'rb')}
        req_media = self.twitter.post(URL_MEDIA, files=files)

        if req_media.status_code != 200:
            print('Media Upload failed. Status Code:{0}, Detail:{1}'.format(req_media.status_code, req_media.text))
            return False

        # アップロードした画像のメディアIDを得る。
        media_id = json.loads(req_media.text)['media_id']

        # 引数のメッセージと、アップロードした画像のメディアIDを合わせてTwitterに投稿する。
        # 失敗した場合はメッセージを出力して終了。
        params = {'status': message, 'media_ids': [media_id]}
        req_text = self.twitter.post(URL_TEXT, params=params)

        if req_text.status_code != 200:
            print('Tweet failed. Status Code:{0}, Detail:{1}'.format(req_text.status_code, req_text.text))
            return False

        print('Tweet Succeeded.')
        return True


if __name__ == '__main__':
    twitter = Twitter()
    twitter.post('Cat!', 'cat1.jpg')
