# coding:utf-8
import MeCab
from slackbot.bot import default_reply
from slackbot.bot import Bot
import json
from seq2seq_model import Seq2SeqModel
import tensorflow as tf
import numpy as np
from util import parse_file, sentence_to_word_id, create_buckets, _buckets, parse_sentence, EOS, ignore_list

id2word = json.load(open("dictionary_i2w.json", "r"))
word2id = json.load(open("dictionary_w2i.json", "r"))
id2word = {int(key): value for key, value in id2word.items()}
vocab_size = len(word2id)

# (2)モデルの生成
sess = tf.Session()
model = Seq2SeqModel(
    vocab_size, vocab_size, _buckets,
    128, 3, 5.0, 1,
    0.5, 0.99,
    forward_only=True, use_lstm=True)
saver = tf.train.Saver()
model.saver.restore(sess, "./tmp/model.ckpt")

# (1)応答用の関数を示すデコレータの定義
@default_reply
def replay_message(message):
    """
    Slack Botの応答を定義する

    :param message: Slack Botのメッセージ
    :return: 応答文を返す。
    """
    # (3)質問文を単語ID列に変換
    questions = [parse_sentence(message.body["text"])]
    try:
        ids_questions = sentence_to_word_id(questions, word2id=word2id)
        token_ids = ids_questions[0]
        bucket_id = 0
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
        # (4)応答文(ID列)の推測
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
        # (5)単語ID列から応答文に変換
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        if EOS in outputs:
            outputs = outputs[:outputs.index(EOS)]
        reply_message = "".join([id2word[output] for output in outputs if not output in ignore_list])
        # (6)Slackへ応答
        message.reply(reply_message)
    except Exception as e:
        print(e)
        message.reply("解析できませんでした。")


def main():
    bot = Bot()
    bot.run()


if __name__ == "__main__":
    main()
