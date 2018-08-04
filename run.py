import gym
import gym_Rubiks_Cube
from baselines import deepq


class Run:
    def __init__(self, m=2, path="./model.pkl", num_episodes=100):
        self.path = path
        self.m = m
        self.num_episodes = num_episodes

    def main(self):
        env = gym.make("RubiksCube-v0")
        env.setScramble(self.m, self.m)
        act = deepq.load(self.path)
        total_reward = []

        for i in range(self.num_episodes):
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

        print("total: {}, Solved: {}, Unsolved: {}".format(len(total_reward), total_reward.count(1),
                                                           total_reward.count(0)))


if __name__ == "__main__":
    r = Run()
    r.main()
