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

def train_model():

    env_count = 6
    env = SubprocVecEnv(
        [lambda: Monitor(make_env(id, quality='ld')()) for id in range(env_count)], start_method='spawn'
        )
    
    #env = DummyVecEnv([make_env(id, quality='ld')])
    
    env = VecFrameStack(env, n_stack=4)

    save_freq = int(100000 / env_count)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./stk_checkpoints/' + str(int(time.time())) + '/', name_prefix='ppo_stk')
    
    model = PPO('CnnPolicy', env, n_steps=256, verbose=1, tensorboard_log="./stk_tensorboard/")

    #model = PPO.load('./stk_checkpoints/ppo_stk_500000_steps.zip')
    #model.set_env(env)

    model.learn(total_timesteps=5000000, callback=checkpoint_callback)
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
    print("src/env.py train successful")

if __name__ == "__main__":
    train_model()