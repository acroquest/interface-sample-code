import agent
import tensorflow as tf
import argparse
from environment import GymEnvironment

env_agent = GymEnvironment(display=True)
agent = agent.DQNAgent(environment=env_agent, display=True)

with tf.Session() as sess:
    agent.build_dqn(sess)
    sess.run(tf.global_variables_initializer())
    agent.load_model()
    agent.play(10)