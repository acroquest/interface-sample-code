# coding:utf-8
from bs4 import BeautifulSoup
import urllib.request
import copy


def scrape_aozora_document(url):
    """
    青空文庫のドキュメントを解析する。

    :param url: 青空文庫のurl
    :return: 文書のリスト
    """
    # (2)HTMLデータの取得
    html = urllib.request.urlopen(url)
    # (3)Beautiful Soupでの読み込み
    soup = BeautifulSoup(html)

    # 青空文庫の本文が含まれる箇所
    for node in soup.findAll(attrs={"class": "main_text"}): # (4)本文を取得
        # (5)ルビを除去
        for ruby_node in node.findAll("ruby"):
            # rt, rpのタグを除去する。
            ruby_node.rt.decompose()
            ruby_node.rp.decompose()
            ruby_node.rp.decompose()
        sentence = node.text
        is_start = False

        sentences_list = []
        sentences = []
        start = 0

        # (6)会話文の抽出
        for index in range(len(sentence)):
            if sentence[index] == u"「":
                is_start = True
                start = index
            elif sentence[index] == u"」" and is_start:
                end = index + 1
                if index + 1 == len(sentence):
                    sentences.append(sentence[start + 1:end - 1])
                    sentences_list.append(copy.deepcopy(sentences))
                    sentences = []
                    break
                if u"「" in sentence[index: index + 20]:
                    sentences.append(sentence[start + 1:end - 1])
                else:
                    sentences.append(sentence[start + 1:end - 1])
                    sentences_list.append(copy.deepcopy(sentences))
                    sentences = []
                is_start = False
    return sentences_list


conversation_list = []
# (1)URLの取得　青空文庫のurlリストを解析する。
with open("../data/urllist.txt", "r") as f:
    for line in f.readlines():
        url = line.replace("\n", "")
        conversation_list.extend(scrape_aozora_document(url))

# (7)テキストに保存
with open("../data/conversation_data.txt", "w") as f:
    for sentences in conversation_list:
        if len(sentences) > 1:
            pass
        # 対話が複数並んだ場合に、分割して出力する。
        for index in range(0, len(sentences), 2):
            if index + 1 == len(sentences):
                break

            f.write("{}\t{}\n".format(sentences[index], sentences[index + 1]))
