import tensorflow as tf
from seq2seq_model import Seq2SeqModel
import json
import MeCab
import os
import numpy as np
from util import parse_file, sentence_to_word_id, create_buckets, _buckets, EOS, ignore_list

tagger = MeCab.Tagger("mecabrc")

id2word = json.load(open("dictionary_i2w.json", "r"))
word2id = json.load(open("dictionary_w2i.json", "r"))

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# 学習データと辞書の取得
questions, answers, _, _ = parse_file("../data/conversation_data.txt")
# 文章をidの配列に変換する
print(questions)
ids_questions = sentence_to_word_id(questions, word2id=word2id)
print(ids_questions)

vocab_size = len(word2id) + 3
print(vocab_size)

ckpt = tf.train.get_checkpoint_state("./tmp")
print(ckpt)
print(tf.train.checkpoint_exists("./tmp/model.ckpt-5000"))

with tf.Session() as sess:
    print('init model')
    model = Seq2SeqModel(
        vocab_size, vocab_size, _buckets,
        128, 3, 5.0, 1,
        0.5, 0.99,
        forward_only=False, use_lstm=True)
    print('finish model')

    print('load model')
    model.saver.restore(sess, "./tmp/model.ckpt-5000")
    print('finish loading model')

    for index, token_ids in enumerate(ids_questions):
        bucket_id = 0
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        print(outputs)
        if EOS in outputs:
            # print(outputs)
            outputs = outputs[:outputs.index(EOS)]
            print(outputs)
        print("questions:", "".join(questions[index])
              , "answers:", "".join([id2word[str(output)] for output in outputs if not output in ignore_list]))
