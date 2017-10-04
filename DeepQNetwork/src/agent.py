from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf
from qfunction import clipped_error, QFunction
from replay_memory import ReplayMemory
from tqdm import tqdm


class History(object):
    """
    ゲームの履歴
    """

    def __init__(self, history_length, screen_height, screen_width):
        """
        履歴の初期化

        :param batch_size: バッチサイズ
        :param history_length: 履歴の長さ
        :param screen_height: 画面の高さ
        :param screen_width: 画面の幅
        :return:
        """
        self.history = np.zeros(
            [history_length, screen_height, screen_width], dtype=np.float32)

    def add(self, screen):
        """
        過去の行動を追加し、最も古いデータを削除する。

        :param screen: 画面
        """
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        """
        履歴を取得する。

        :return: 履歴
        """
        return np.transpose(self.history, (1, 2, 0))


class DQNAgent(object):
    def __init__(self, environment=None, display=False
                 ):
        self.weight_dir = 'weights'
        self.sess = None
        self.env = environment

        # 最大のステップ数
        self.max_step = 50000000
        # バッチサイズ
        self.batch_size = 32

        # εにより調整する値
        self.ep_end = 0.01
        self.ep_start = 1.
        self.ep_end_t = 1000000

        # 履歴の長さ
        self.history_length = 4
        # 学習開始のステップ数（これ以前は学習しない）
        self.learn_start = 50000.

        # 画面の高さの値
        self.screen_height = 84
        # 画面の幅の値
        self.screen_width = 84

        # 履歴の初期化
        self.history = History(self.history_length, self.screen_height, self.screen_width)

        # リプレイメモリへの保存数
        memory_size = 1000000
        # リプレイメモリの初期化
        self.memory = ReplayMemory("./model_dir", memory_size, self.batch_size, self.screen_height,
                                   self.screen_width,
                                   self.history_length)
        # 何回エージェントが行動したか、合計のステップ数
        self.step = 0


    def build_dqn(self, sess):
        """
        DQNをTensorFlowを使って構築する
        """

        self.sess = sess

        # (1) 予測ネットワークの作成
        with tf.variable_scope('prediction'):
            self.qfunction = QFunction(self.screen_height, self.screen_width, self.history_length, self.env.action_size)
            self.q_action = tf.argmax(self.qfunction.q, dimension=1)

        # (2) ターゲット用ネットワークの作成
        with tf.variable_scope('target'):
            self.target_qfunction = QFunction(self.screen_height, self.screen_width, self.history_length,
                                              self.env.action_size)

        # (3) パラメータコピー処理の作成
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.qfunction.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.target_qfunction.w[name].get_shape().as_list(),
                                                      name=name)
                self.t_w_assign_op[name] = self.target_qfunction.w[name].assign(self.t_w_input[name])

        # (4) 最適化関数の構成
        with tf.variable_scope('optimizer'):

            # 取った行動のQ値を計算する。
            self.action = tf.placeholder('int64', [None], name='action')
            action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.qfunction.q * action_one_hot, reduction_indices=1, name='q_acted')

            # 次ステップのQ値を取得する
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')

            # 2つQ値の差分を計算する。
            delta = self.target_q_t - q_acted

            self.global_step = tf.Variable(0, trainable=False)

            # 誤差を計算する。
            self.loss = tf.reduce_mean(clipped_error(delta), name='loss')
            # 学習律の調整
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            # 学習の係数を調整するパラメータ群
            learning_rate = 0.00025
            learning_rate_decay = 0.96
            learning_rate_decay_step = 50000
            self.learning_rate_op = tf.maximum(learning_rate,
                                               tf.train.exponential_decay(
                                                   learning_rate,
                                                   self.learning_rate_step,
                                                   learning_rate_decay_step,
                                                   learning_rate_decay,
                                                   staircase=True))
            # 最適化手法、RMSProp
            self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        tf.initialize_all_variables().run()
        # (5) モデル保存の準備
        self._saver = tf.train.Saver(list(self.qfunction.w.values()), max_to_keep=30)

        # (6) モデル復旧
        self.load_model()

        # (7)ターゲット用ネットワークにパラメータをコピー
        self.update_target_q_network()

    def update_target_q_network(self):
        """
        TargetのQ Networkを更新する関数を定義する。
        """
        for name in self.qfunction.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.qfunction.w[name].eval()})

    def load_model(self):
        """
        モデルを読み込む

        :return: 読み込みに成功すればTrue, 失敗すればFalse
        """
        print(" [*] モデルの読み込みを開始します...")

        # ファイルが生成されていれば、データを読み込む処理を実行する。
        checkpoint_dir = os.path.join('checkpoints', self.env.env_name + '/')
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(checkpoint_dir, ckpt_name)
            self._saver.restore(self.sess, fname)
            print(" [*] モデルの読み込みに成功しました。: %s" % fname)
            return True
        else:
            print(" [!] モデルの読み込みに失敗しました。: %s" % checkpoint_dir)
            return False

    def train(self, episodes):
        """
        エージェントの学習用

        :param episodes: ゲームの試行回数
        """
        start_step = 0
        max_avg_ep_reward = 0
        # テストの間隔
        test_step = 50000
        # (1) 履歴の初期化
        screen, reward, action, terminal = self.env.new_random_game()
        for _ in range(self.history_length):
            self.history.add(screen)

        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, total_loss, total_q = 0., 0., 0.
        ep_rewards, actions = [], []
        sum_num_game = 0

        # (2) 予測と学習のループの開始
        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):

            if self.step == self.learn_start:
                num_game, update_count, ep_reward = 0, 0, 0.
                total_reward, total_loss, total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # (3) 行動を予測
            action = self.predict(self.history.get())

            # (4) 予測した行動を環境に入力
            screen, reward, terminal = self.env.act(action, is_training=True)

            # (5) 環境からの画面と得点を履歴とリプレイメモリに追加
            self.history.add(screen)
            self.memory.add(screen, reward, action, terminal)

            # (6) 1ゲームが終了したら、ゲーム回数を増やし、総得点を記録
            if terminal:
                screen, reward, action, terminal = self.env.new_random_game()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
                sum_num_game += 1
            else:
                ep_reward += reward
            actions.append(action)
            total_reward += reward

            # (7) 全ゲーム数が指定された試行回数になったらループを終了
            if sum_num_game > episodes:
                print("規定のゲーム試行数を超えたので学習を終了します。")
                break

            # (8) 学習開始ステップになるまでは、学習をせずにループの先頭に戻る
            if not (self.step >= self.learn_start):
                continue

            # (9) 学習
            loss, q = self.observe(screen, reward, action, terminal)
            total_loss += loss
            total_q += q

            # これ以降の経過出力とモデル保存は一定間隔でしか実施しないようにする
            if not ( self.step % test_step == test_step - 1):
                continue

            # (10) 途中経過の出力
            avg_reward = total_reward / test_step
            avg_loss = total_loss / self.update_count
            avg_q = total_q / self.update_count

            try:
                max_ep_reward, min_ep_reward, avg_ep_reward = np.max(ep_rewards), np.min(ep_rewards), np.mean(
                    ep_rewards)
            except:
                max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0
            print(
                '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
                % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

            # 出力した経過を次回に向けてリセットする
            num_game, total_reward, total_loss, total_q, self.update_count, ep_reward = 0, 0., 0., 0., 0, 0
            ep_rewards, actions = [], []

            # (11) モデルの中間保存
            if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                self.save_model(self.step + 1)
                max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)


    def predict(self, s_t, test_ep=None):
        """
        行動価値関数を予測する

        :param s_t: 現状の状態
        :param test_ep: 予測の場合に使うepsilon（ランダムに振舞う確率）
        :return: 起こす行動
        """
        # ランダムでの行動を調整する。学習が進むごとにランダムに振る舞う確率を下げる。
        ep = test_ep or (self.ep_end +
                         max(0., (self.ep_start - self.ep_end)
                             * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

        if random.random() < ep:
            # ランダムで行動する。
            action = random.randrange(self.env.action_size)
        else:
            # DQNを利用して行動を予測する。
            action = self.q_action.eval({self.qfunction.s_t: [s_t]})[0]

        self.step += 1

        return action

    def observe(self, screen, reward, action, terminal):
        """
        観測したときの行動

        :param screen: 状態
        :param reward: 報酬
        :param action: 行動
        :param terminal: ゲームが終了したかどうか
        :return: 誤差、Q値の平均
        """
        # 報酬の最大、最小を定義する。
        reward = max(-1, min(1.0, reward))
        train_frequency = 4
        target_q_update_step = 10000

        loss, mean_q = 0.0, 0.0

        # 学習を行う。
        if self.step % train_frequency == 0:
            loss, mean_q = self.q_learning_mini_batch()

        # DQNと教師側のTarget Q Network側のパラメータを同期する。
        if self.step % target_q_update_step == target_q_update_step - 1:
            self.update_target_q_network()

        return loss, mean_q


    def q_learning_mini_batch(self):
        """
        Q学習の学習を行う。

        :return: 誤差、qの値の平均
        """

        # リプレイメモリからランダム時点の状態と、行動、報酬、次ステップ状態、終了状態を得る。
        s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

        # ターゲット用ネットワークを使い、次ステップの状態のQ値を計算し、その中から最大のものを取得する。
        q_t_plus_1 = self.target_qfunction.q.eval({self.target_qfunction.s_t: s_t_plus_1})
        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)

        # 割引率
        discount = 0.99
        # 次ステップの最大Q値から教師データを計算する。
        # 「割引率×次ステップの最大Q値 ＋報酬」である。
        target_q_t = (1. - terminal) * discount * max_q_t_plus_1 + reward

        # DQNの学習を行う。
        _, q_t, loss = self.sess.run([self.optim, self.qfunction.q, self.loss], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.qfunction.s_t: s_t,
            self.learning_rate_step: self.step,
        })
        self.update_count += 1

        return loss, q_t.mean()

    def save_model(self, step=None):
        """
        モデルを保存する

        :param step: ファイル名に利用するステップ
        """
        print(" [*] モデルのチェックポイントを保存します。...")
        checkpoint_dir = os.path.join('checkpoints', self.env.env_name + '/model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self._saver.save(self.sess, checkpoint_dir, global_step=step)


    def play(self, n_episode):
        """
        ゲームをプレイする。

        :param n_episode: ゲームを実行する回数
        """
        test_ep = self.ep_end
        n_step = 100000

        for idx in range(n_episode):
            current_reward, best_reward, best_idx = 0, 0, 0
            # (1) ゲームの開始命令
            screen, reward, action, terminal = self.env.new_random_game()

            for _ in range(self.history_length):
                self.history.add(screen)

            # (2) ゲームのループを開始
            for _ in tqdm(range(n_step), ncols=70):
                # (3) ゲーム内の処理を開始
                action = self.predict(self.history.get(), test_ep)
                # ゲームに対して行動する
                screen, reward, terminal = self.env.act(action, is_training=False)
                # ゲーム画面をプレイの履歴に加える。
                self.history.add(screen)
                current_reward += reward

                # ゲーム終了時には以降の処理を継続しない。
                if terminal:
                    break

            # 最も良いゲームのスコアを記録する。
            if current_reward > best_reward:
                best_reward = current_reward
                best_idx = idx
        # (4) 結果の表示
        print("=" * 30)
        print(" [%d] 最大スコア : %d" % (best_idx, best_reward))
        print("=" * 30)
