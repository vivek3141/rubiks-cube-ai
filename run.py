import gym
import gym_Rubiks_Cube
from baselines import deepq

PATH = "./model.pkl"

env = gym.make("RubiksCube-v0")

# Set m to be the number of moves the scramble should be
m = 2

env.setScramble(m, m)
act = deepq.load(PATH)
total_reward = []

for i in range(100):
    obs, done = env.reset(), False
    episode_rew = 0

    while not done:
        # uncomment this if you want it to render the cube
        # env.render()

        obs, rew, done, _ = env.step(act(obs[None], update_eps=0)[0])
        episode_rew += rew
    total_reward.append(episode_rew)
    # uncomment this if you want it to render the last state
    # env.render()

    print("Episode reward: {}".format(episode_rew))
    print("scramble, action_history: {}".format(env.getlog()))
    print("-----------------------")

print("total: {}, Solved: {}, Unsolved: {}".format(len(total_reward), total_reward.count(1), total_reward.count(0)))
