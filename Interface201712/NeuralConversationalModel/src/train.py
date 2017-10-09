# coding:utf-8
import tensorflow as tf
import time
from seq2seq_model import Seq2SeqModel
import numpy as np
from util import parse_file, sentence_to_word_id, create_buckets, _buckets
import json


def train():
    """
    モデルの学習

    :return: None
    """
    # (1)学習データと辞書の取得
    questions, answers, word2id, id2word = parse_file("../data/conversation_data.txt")

    # 文章をidの配列に変換する
    ids_questions = sentence_to_word_id(questions, word2id=word2id)
    ids_answers = sentence_to_word_id(answers, word2id=word2id)
    vocab_size = len(word2id)

    # (2)前処理 バケット生成 内部的に形態素解析の幅が長すぎるデータを削除
    train_data = create_buckets(ids_questions, ids_answers)
    start_time = time.time()

    # (3)バケットごとの数の割合を計算
    train_bucket_sizes = [len(train_data[b]) for b in range(len(_buckets))]
    print(train_bucket_sizes, vocab_size)
    # データ数と等しい
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    # tensotflow での処理を開始する。
    with tf.Session() as sess:
        # (4)モデル作成
        model = Seq2SeqModel(
            vocab_size, vocab_size, _buckets,
            128, 3, 5.0, 16,
            0.5, 0.99, use_lstm=True)
        # 初期化
        sess.run(tf.global_variables_initializer())

        current_step = 0
        step_time, loss = 0.0, 0.0
        step_per_checkpoint = 100
        step_times = 10000

        for step_time_now in range(step_times):
            random_number_01 = np.random.random_sample()
            # bucket idを選択する。
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])
            # (5)バッチ処理を行うデータを選択
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_data, bucket_id)

            # (6)学習
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            # ステップ時間を計算する。
            step_time += (time.time() - start_time) / step_per_checkpoint
            current_step += 1

            # (7)一定間隔ごとにモデルの評価
            if current_step % step_per_checkpoint == 0:
                print("step:{} time:{}".format(current_step, step_time))
                # 誤差を計算し、描画する。
                for bucket_id in range(len(_buckets)):
                    if len(train_data[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        train_data, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    print("  eval: bucket %d loss %.2f" % (bucket_id, eval_loss))

        # (8)学習完了時点のモデルをファイル化
        model.saver.save(sess, "./tmp/model.ckpt")
        # 辞書データを保存する。
        json.dump(id2word, open("dictionary_i2w.json", "w"))
        json.dump(word2id, open("dictionary_w2i.json", "w"))


if __name__ == '__main__':
    train()
