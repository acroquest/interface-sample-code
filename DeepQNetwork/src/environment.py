import random

import gym
import numpy as np

try:
    from scipy.misc import imresize
except:
    import cv2

    imresize = cv2.resize


def rgb2gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


class GymEnvironment(object):
    def __init__(self, env_name="Breakout-v0", display=False):
        screen_width = 84
        screen_height = 84
        self.env_name = env_name
        self.env_type = "game"
        # ゲームの環境を構築する。
        self.env = gym.make(env_name)

        self.random_start = 30

        self.display = display
        self.dims = (screen_width, screen_height)

        self._screen = None
        self.reward = 0
        self.terminal = True
        self.step_info = None
        self.action_size = self.env.action_space.n

    def new_random_game(self):
        """
        新しくゲームを開始する。
        但し、random_startの間ランダムで行動する。

        :return:
        """
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._screen, self.reward, self.terminal, self.step_info = self.env.step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_game(self, from_random_game=False):
        """
        新しくゲームを開始する

        :param from_random_game:
        :return:
        """
        if self.lives == 0:
            self._screen = self.env.reset()
        self._screen, self.reward, self.terminal, self.step_info = self.env.step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    @property
    def screen(self):
        """
        観測情報、グレーとリサイズを行う。

        :return: 画面のグレー化した画像
        """
        return imresize(rgb2gray(self._screen) / 255., self.dims)

    @property
    def lives(self):
        if self.step_info is None:
            return 0
        else:
            return self.step_info['ale.lives']

    def render(self):
        """
        プレイ画面の描画
        """
        if self.display:
            self.env.render()

    def act(self, action, is_training=True):
        """
        環境に対し、行動をする。

        :param action: アクション
        :param is_training: 学習用かそうでないか。
        :return:
        """
        start_lives = self.lives

        self._screen, self.reward, self.terminal, self.step_info = self.env.step(action)
        cumulated_reward = self.reward

        if is_training and start_lives > self.lives:
            cumulated_reward -= 1
            self.terminal = True

        self.reward = cumulated_reward
        self.render()

        return self.screen, self.reward, self.terminal
