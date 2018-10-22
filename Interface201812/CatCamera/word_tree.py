# coding: utf-8

import sys
from collections import defaultdict

# 単語ファイル
WORD_ID_FILE = 'word_ids.txt'
# 全単語の祖先になる"entity"の単語ID
ROOT_ID = 'n00001740'


class WordTree:
    """
    ImageNetで扱う単語群の親子関係などを判定・出力する。
    word_ids.txtと一緒に配置すること。
    """

    def __init__(self):
        """
        コンストラクタ。単語ファイルを読み込み、IDと名前の対応や、単語の関係を保持する。
        """
        # 単語IDとその親のIDの辞書
        self._child_parent = {}
        # 単語IDとその子（複数）の対応
        self._parent_children = defaultdict(list)

        # 単語IDとその名前（複数）の辞書
        self._word_id_names = {}
        # 単語の名前とそのIDの辞書。名前は検索用に全て小文字化して保持する。
        self._word_name_id = {}

        # 単語ファイルを読み込み、上の各辞書に単語の対応関係を詰める。
        # 単語ファイルはCSV形式で、
        # 「単語ID, 親のID, 名前(複数)」
        # の内容である。
        with open(WORD_ID_FILE) as word_ids_file:
            for line in word_ids_file:
                # CSVの1行から、単語ID, 親のID, 名前(複数)を取り出す。
                params = line.strip().split(',')
                word_id = params[0]
                parent_id = params[1]
                names = params[2:]

                # 単語の親子関係を記録する。
                self._word_id_names[word_id] = names
                self._parent_children[parent_id].append(word_id)

                # 単語のIDと名前の関係を記録する。
                self._child_parent[word_id] = parent_id

                for name in names:
                    self._word_name_id[name.lower()] = word_id

    def find_id(self, name):
        """
        引数nameの単語名に対応するIDを返す。
        :param name: 単語名。大文字小文字は区別されない
        :return: nameに対応するID。無ければNone
        """
        # 名前からIDを引く。大文字小文字を区別しないよう初期化時に小文字で保持しているので、検索も小文字で行う。
        lower_name = name.lower()
        if lower_name in self._word_name_id:
            return self._word_name_id[lower_name]
        else:
            return None

    def print_all_tree(self):
        """
        全単語の親子な関係をツリー形式で標準出力する。
        :return:
        """
        tree_lines = word_tree.make_tree()
        for line in tree_lines:
            print(line)

    def make_tree(self, id=ROOT_ID, tree_lines='', bottom=True, depth=-1):
        """
        引数idの全配下の単語関係を出力する。
        :param id: 出力したい単語ID
        :param tree_lines: 出力時、各行の左側に出すツリーの線（内部処理用）
        :param bottom: この単語が兄弟関係にある単語の中の最後の出力か？（内部処理用）
        :param depth: 何階層まで出力するか
        :return:
        """
        ret = []

        '''
        `-親ID: 単語名
            +-子ID1: 単語名 
            |   `-孫ID1: 単語名
            `-子ID2: 単語名
                `-孫ID2: 単語名

        のように、引数のidが子ID1か2のどちがの位置か（兄弟関係の中で一番最後の出力か）で
        単語IDの左側に出す記号を切り替える。
        '''
        if bottom:
            leaf_prefix = '`-'
            branch_mark = '    '
        else:
            leaf_prefix = '+-'
            branch_mark = '|   '

        # ツリー構造の線と、ID:単語名(複数)を出力する。
        word_name_str = ','.join(self._word_id_names[id])
        ret.append('{0}{1}{2}: {3}'.format(tree_lines, leaf_prefix, id, word_name_str))

        # 出力すべき階層の最後に達したら終了する。
        if depth == 0:
            return ret

        # 自IDの子の単語について再帰呼び出しする。
        children = self._parent_children[id]
        for index, child in enumerate(children):
            bottom_child = (index == len(children) - 1)

            child_lines = self.make_tree(child, tree_lines + branch_mark, bottom_child, depth - 1)
            ret.extend(child_lines)

        return ret

    def list_descendants(self, id):
        """
        引数IDの配下にある全要素を返す
        :param id: 単語ID
        :return: idの配下にある単語。{ID:名前のリスト}の辞書形式。idも含む。
        """
        ret = {}

        # 引数idと名前(複数)を返却する辞書に追加する。
        word_name = self._word_id_names[id]
        ret[id] = word_name

        # 子要素に対して再帰呼び出し、その結果も返却する辞書に追加する。
        children = self._parent_children[id]
        for child in children:
            child_descendants = self.list_descendants(child)
            ret.update(child_descendants)

        return ret


if __name__ == "__main__":
    """
    第一引数に単語名を渡すと、ID、配下のツリー、配下の要素のリストを表示する。
    引数が無い場合は'domestic cat'を用いる。
    """
    word_tree = WordTree()

    argvs = sys.argv
    if len(argvs) == 1:
        word_name = 'domestic cat'
    else:
        word_name = argvs[1]

    word_id = word_tree.find_id(word_name)
    print('Name:{0}'.format(word_name))
    print('ID:{0}'.format(word_id))

    print('===================')
    print('Descendants tree:')
    lines = word_tree.make_tree(id=word_id, depth=3)
    for line in lines:
        print(line)

    print('===================')
    print('Descendants list:')
    descendants = word_tree.list_descendants(word_id)
    for desc_id, desc_names in descendants.items():
        print('  {0} : {1}'.format(desc_id, desc_names))
