import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import time
import numpy as np
from utils import make_env
from matplotlib import pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

def test_model():
    model_str = './stk_checkpoints/ppo_stk_2040000_steps'
    eval_env = SubprocVecEnv(
        [lambda: Monitor(make_env(id, quality='ld')()) for id in range(1)], start_method='spawn'
        )
    eval_env = VecFrameStack(eval_env, n_stack=4)
    model = PPO.load(model_str)
    mean_rew, mean_std = evaluate_policy(model, eval_env, n_eval_episodes=3, render=True)
    print("Mean reward:", mean_rew, "Mean std:", mean_std)

def train_model():

    env = SubprocVecEnv(
        [lambda: Monitor(make_env(id, quality='ld')()) for id in range(15)], start_method='spawn'
        )
    env = VecFrameStack(env, n_stack=4)
    
    #env = make_env(0)()

    checkpoint_callback = CheckpointCallback(save_freq=8000, save_path='./stk_checkpoints/', name_prefix='ppo_stk')
    
    model = PPO('CnnPolicy', env, n_steps=256, verbose=1, tensorboard_log="./stk_tensorboard/")
    model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    model.save('ppo_stk' + str(int(time.time())))

    if False:
        env.reset()

        action = [0, 1, 0, 0, 0, 0] # brake
        for _ in range(100):
            print(_)
            action = action #env.action_space.sample()
            image, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                env.reset()

            plt.imshow(np.array(image).astype(np.uint8), cmap='gray')
            plt.pause(0.1)
        
        plt.close()

    env.close()
    print("src/env.py test successful")

if __name__ == "__main__":
    train_model()
    #test_model()
    #test_ppo()
    #test_vae_model()
