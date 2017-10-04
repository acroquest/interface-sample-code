"""
Open AI Gym動作用
"""
from __future__ import absolute_import
from __future__ import unicode_literals

import gym
env = gym.make('Breakout-v0')
env.reset()
# 1000回ランダムに行動する。終了地点でゲームを完了する。
for _ in range(1000):
    env.render()
    # ランダムで行動を行う。画面情報、報酬、ゲーム完了、ステップごとの情報をDQNから取得する。
    _screen, reward, terminal, step_info = env.step(env.action_space.sample())

    # 終了
    if terminal is True:
        break