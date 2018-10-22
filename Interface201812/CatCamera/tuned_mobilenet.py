# coding: utf-8

import os

import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical


def TunedMobileNet(weights='imagenet'):
    """
    ファインチューニング用のMobileNetモデルを生成して返す。
    :return: モデル
    """
    # (1) MobileNetモデルの取得　
    # Topの全結合層を外したものを得る。
    base_model = MobileNet(include_top=False,
                           weights=weights,
                           input_tensor=None,
                           input_shape=(224, 224, 3))

    # (2) ファインチューニング用の層を追加
    # 得たモデルに、平均プーリング層、全結合層、ドロップアウト層、全結合層を追加する。
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    top_model = Dense(2, activation='softmax')(x)

    # (3) コンパイル
    # 最適化関数にSGDを指定してモデルをコンパイルする。
    model = Model(inputs=base_model.input, outputs=top_model)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['categorical_crossentropy', 'accuracy'])

    # (4) サマリを出力
    model.summary()

    return model


def load_images(directory, start, end):
    """
    directory配下にある犬、猫の画像を、start～endの番号分だけ読み込んで返す。
    :param directory:画像ディレクトリ
    :param start:開始番号
    :param end:終了番号
    :return:画像リストとラベルリストのNumpy配列　ラベルは猫:0、犬:1
    """
    labels = []
    imgs = []

    # 猫と犬ごとに、start～endの番号の画像を読み込む。
    for animal in ['cat', 'dog']:
        for i in range(start, end):
            # 画像パスを得る。ファイル名はcat.{番号}.jpg、またはdog.{番号}.jpg である。
            filename = '{}.{}.jpg'.format(animal, i)
            path = os.path.join(directory, filename)

            # 学習/評価用に加工する。
            img = image.load_img(path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            imgs.append(x)

            # 猫なら0、犬なら1をラベルに追加する。
            label = 0 if animal == 'cat' else 1
            labels.append(label)

    # Numpy配列形式で返す。
    return np.array(imgs), np.array(labels)


def train(directory):
    """
    directory配下の画像でモデルを学習する。
    :param directory: 画像ディレクトリ
    :return: 学習結果のモデル
    """
    # (1) ファインチューニング用のモデル構造を生成
    model = TunedMobileNet()

    # (2) 学習用の画像とラベルを取得
    imgs, labels = load_images(directory, 0, 1400)
    categorical_labels = to_categorical(labels)

    # (3) 画像を再学習させる
    model.fit(imgs, categorical_labels, epochs=3, validation_split=0.2)

    # (4) 評価
    # 評価用の画像をモデルに与え、結果を出力する。
    valid_imgs, valid_labels = load_images(directory, 1400, 2000)
    valid_categorical_labels = to_categorical(valid_labels)
    result = model.evaluate(valid_imgs, valid_categorical_labels)
    print('Evaluate result:{0}'.format(result))

    # (5) モデルをファイルに保存
    model.save('./model.h5')

    return model


def test(model, directory):
    """
    モデルを使って判定の実験をする。
    :param model: モデル
    :param directory: 画像ディレクトリ
    """
    # (4) テスト用の画像を取得
    imgs, labels = load_images(directory, 3000, 3005)

    # (5) 判定
    results = model.predict(imgs)

    # (6) 判定結果をデコードし出力
    decoded_result = decode_predictions(results)
    for result, label in zip(decoded_result, labels):
        print('label:{0}, result:{1}'.format(label, result))


def decode_predictions(preds, threshold=0.8, top=10):
    """
    判定結果を解釈する。返却結果はKerasの同名のメソッドと同じ形式にそろえる。
    :param preds: 判定結果
    :param threshold: 正解とするスコアの閾値
    :return: (単語ID, 名称, 判定スコア)のタプルの1要素だけのリストの、preds分のリスト
    """
    results = []
    for cat_score, dog_score in preds:
        if cat_score >= threshold:
            results.append([('n02121808', 'domestic cat', cat_score)])
        elif dog_score >= threshold:
            results.append([('n02084071', 'dog', dog_score)])

    return results


if __name__ == '__main__':
    target_directory = './Dogs vs Cats/all/train/'

    # 学習を実施。
    # model = train(target_directory)

    # 学習結果のモデルを読み込み、テスト。
    model = TunedMobileNet(weights=None)
    model.load_weights('./model.h5')
    test(model, target_directory)
