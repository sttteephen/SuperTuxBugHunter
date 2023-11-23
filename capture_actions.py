import time
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2

from utils import make_env

# capture the actions taken by the model for a number of episodes
def capture_actions():
    max_ep_count = 10
    actions_list = [[] for _ in range(max_ep_count)]

    env_count = 1
    env = SubprocVecEnv(
        [lambda: Monitor(make_env(id, quality='ld')()) for id in range(env_count)], start_method='spawn'
        )
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load('baseline.zip', env)

    observation = env.reset()
    ep_count = 0

    while ep_count < max_ep_count:

        action = model.predict(observation)[0]
        actions_list[ep_count].append(action)
        
        observation, reward, done, info = env.step(action)

        #image = cv2.cvtColor(observation[0, :, :, 0], cv2.COLOR_BGR2RGB) 
        #cv2.imshow('', image)
        #cv2.waitKey(1)

        if done:
            env.reset()
            ep_count += 1
            print(ep_count)

    env.close()
    return actions_list


# execute the same sequence of actions for a number of episodes, recording the fps at each frame
def execute_actions(actions):
        max_ep_count = 100

        env_count = 1
        env = SubprocVecEnv(
            [lambda: Monitor(make_env(id, quality='ld')()) for id in range(env_count)], start_method='spawn'
            )
        env = VecFrameStack(env, n_stack=4)

        model = PPO.load('baseline.zip', env)

        observation = env.reset()
        ep_count = 0

        while ep_count < max_ep_count:

            action_index = 0
            while action_index < len(actions):
                action = actions[action_index]
                action_index += 1
                
                observation, reward, done, info = env.step(action)

                #image = cv2.cvtColor(observation[0, :, :, 0], cv2.COLOR_BGR2RGB) 
                #cv2.imshow('', image)
                #cv2.waitKey(1)

                if done:
                    env.reset()
                    ep_count += 1
                    print(ep_count)

        env.close()

# graph the render times for each frame of each recorded episode
def graph_render_times():
    # Load the CSV file
    file_path = 'same_action_fps7d869b53-bc18-4f21-93ed-268ab1730c61.csv'
    fps_data = pd.read_csv(file_path)
    print(len(fps_data))
    # Drop any 'Unnamed' columns if they exist
    fps_data = fps_data.drop(columns=[col for col in fps_data.columns if 'Unnamed' in col], errors='ignore')

    # Transpose the data so that each row becomes a column
    transposed_data = fps_data.T

    # Filtering out columns where any value is below 10
    filtered_data = transposed_data.loc[:, transposed_data.apply(lambda x: x.min() >= 10)]

    # Resetting the index of the filtered data to ensure numeric x-axis
    filtered_data_reset = filtered_data.reset_index(drop=True)

    # Plotting the data
    plt.figure(figsize=(15, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(filtered_data_reset.columns)))

    for i, column in enumerate(filtered_data_reset.columns):
        plt.plot(filtered_data_reset[column], color=colors[i])

    plt.title('Render Time for 100 Episodes of Same Actions')
    plt.xlabel('Time Step')
    plt.ylabel('Render Time')
    plt.ylim(9, 17)  # Setting the y-axis limit
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    if False:
        file = open('actions_list.pkl', 'wb')
        pickle.dump(capture_actions(), file)
        file.close()

    if False:
        file = open('actions_list.pkl', 'rb')
        actions_list = pickle.load(file)
        execute_actions(actions_list[0])
        file.close()

    if True:
        graph_render_times()