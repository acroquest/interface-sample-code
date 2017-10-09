"""
Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py
"""

import numpy as np
import random


class ReplayMemory(object):
    def __init__(self, model_dir, memory_size, batch_size, screen_height, screen_width, history_length):
        self.model_dir = model_dir

        self.memory_size = memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.integer)
        self.screens = np.empty((self.memory_size, screen_height, screen_width), dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = history_length
        self.dims = (screen_height, screen_width)
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size,) + self.dims + (self.history_length,), dtype=np.float16)
        self.poststates = np.empty((self.batch_size,) + self.dims + (self.history_length,), dtype=np.float16)

    def add(self, screen, reward, action, terminal):
        """
        Experiance Replay用のデータを追加する。

        :param screen: 観測している画面
        :param reward: 報酬
        :param action: 行動
        :param terminal: 完了するか否か
        """
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        """
        状態を取得する

        :param index: 取得対象になるインデックス
        :return: 状態
        """
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return np.transpose(self.screens[(index - (self.history_length - 1)):(index + 1), ...], (1, 2, 0))
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return np.transpose(self.screens[indexes, ...], (1, 2, 0))

    def sample(self):
        """
        保持しているメモリからランダムに取得する。

        :return: 前の状態、行動、報酬、後の状態、完了結果、それぞれのリスト
        """
        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals