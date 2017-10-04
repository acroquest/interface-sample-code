# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals
import MeCab

tagger = MeCab.Tagger('mecabrc')

PAD = 0
EOS = 1
ignore_list = [PAD, EOS]
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (80, 100)] #(4)バケット長の定義
# _buckets = [(10, 15), (20, 25), (40, 50)] #非力なマシン用の定義


def parse_sentence(sentence):
    """
    文章の形態素解析を行う。

    :param sentence: 文章
    :return: 形態素解析の結果
    """
    parsed = []
    for chunk in tagger.parse(sentence).splitlines()[:-1]:
        (surface, feature) = chunk.split('\t')
        parsed.append(surface)
    return parsed


def parse_file(filename):
    """
    会話ドキュメントを解析する。

    :param filename: ファイル名
    :return: 質問文、応答文、辞書（言葉→ID)、辞書（ID→言葉）
    """
    questions = []
    answers = []
    # (1)ファイルの読み込み
    with open(filename, "r") as f:
        lines = f.readlines()

        for line in lines:
            sentences = line.split("\t")
            if len(sentences) != 2:
                continue
            # (2)形態素解析
            question = parse_sentence(sentences[0])
            answer = parse_sentence(sentences[1])
            questions.append(question)
            answers.append(answer)
    word2id = {}
    id2word = {}
    id = 2
    # (3)番号割り付け
    sentences = questions + answers
    for sentence in sentences:
        for word in sentence:
            if word not in word2id:
                word2id[word] = id
                id2word[id] = word
                id += 1

    return questions, answers, word2id, id2word


def sentence_to_word_id(split_sentences, word2id):
    """
    文章をword idの配列に変換する。

    :param split_sentences: 形態素解析などで分割された文章
    :param word2id: wordからid変換する辞書
    :return: wordidの配列
    """
    id_sentences = []
    for sentence in split_sentences:
        ids = []
        for word in sentence:
            id = word2id[word]
            ids.append(id)
        ids.append(EOS)
        id_sentences.append(ids)
    return id_sentences


def create_buckets(question_ids_list, answer_ids_list):
    """
    Tensor FlowでRNNを実行するためのバケットを使う。

    :param question_ids_list: 質問文のidのリスト
    :param answer_ids_list: 回答文のidのリスト
    :return: TensorFlowで用いるバケット
    """
    # (5)バケットの生成
    data_set = [[] for _ in _buckets]
    for question_ids, answer_ids in zip(question_ids_list, answer_ids_list):
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(question_ids) < source_size and len(answer_ids) < target_size:
                data_set[bucket_id].append([question_ids, answer_ids])
                break
    return data_set
