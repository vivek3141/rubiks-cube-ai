import argparse
from train import Train
from run import Run

parser = argparse.ArgumentParser(description='Run the program')
parser.add_argument('mode', metavar='mode', type=str,
                    help="Specify 'train' or 'run' to run or train the model")
args = parser.parse_args()

if args.mode.upper() == "TRAIN":
    # Feel free to change these
    t = Train(
        lr=0.0001,
        seed=0,
        check_freq=10000,
        check_path="./model_checkpoint",
        timesteps=1000000
    )
    t.main()

if args.mode.upper() == "RUN":
    # Feel free to change these
    r = Run(
        m=2,
        path="./model.pkl",
        num_episodes=100
    )
    r.main()

else:
    print("Please enter 'train' or 'mode'")
