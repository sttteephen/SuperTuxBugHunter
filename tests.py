import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from utils import make_env

def test_model(model_str):
    eval_env = SubprocVecEnv(
        [lambda: Monitor(make_env(id, quality='ld')()) for id in range(1)], start_method='spawn'
        )
    
    #eval_env = DummyVecEnv([make_env(id, quality='ld')])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    model = PPO.load(model_str)
    mean_rew, mean_std = evaluate_policy(model, eval_env, n_eval_episodes=3, render=True)
    print("Mean reward:", mean_rew, "Mean std:", mean_std)


if __name__ == "__main__":
    test_model(sys.argv[1])
