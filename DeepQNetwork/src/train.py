import agent
from environment import GymEnvironment
import tensorflow as tf

env_agent = GymEnvironment()
agent = agent.DQNAgent(environment=env_agent)

with tf.Session() as sess:
    agent.build_dqn(sess)
    sess.run(tf.global_variables_initializer())

    agent.train(episodes=50000)
