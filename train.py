import gym
import gym_Rubiks_Cube

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger


def callback(lcl, glb):
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 0.95
    if is_solved:
        return True


SEED = 0
CHECK_FREQ = 10000
CHECK_PATH = "./model_checkpoint"
TIMESTEPS = 1000000
logger.configure()
set_global_seeds(SEED)
env = gym.make("RubiksCube-v0")

model = deepq.models.mlp([128, 128])

act = deepq.learn(
    env,
    q_func=model,
    lr=1e-4,
    max_timesteps=TIMESTEPS,
    buffer_size=50000,
    exploration_fraction=0.3,
    exploration_final_eps=0.01,
    print_freq=100,
    train_freq=4,
    learning_starts=10000,
    target_network_update_freq=1000,
    gamma=0.99,
    prioritized_replay=True,
    prioritized_replay_alpha=0.6,
    checkpoint_freq=CHECK_FREQ,
    checkpoint_path=CHECK_PATH,
)

print("Saving...")
act.save("model.pkl")
