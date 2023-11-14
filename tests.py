import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import time

def test_env():

    import numpy as np
    from utils import make_env
    from matplotlib import pyplot as plt

    env = make_env(0, quality='ld')()

    start = time.time()
    model = PPO('CnnPolicy', env, verbose=1, n_steps=64, tensorboard_log="./stk_tensorboard/")
    model.learn(total_timesteps=10)
    model.save('ppo_stk' + str(int(time.time())))
    end = time.time()

    print("Training time: ", end - start)

    if False:
        env.reset()

        action = [0, 1, 0, 0, 0, 0] # brake
        for _ in range(100):
            print(_)
            action = env.action_space.sample()
            image, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                env.reset()

            plt.imshow(np.array(image).astype(np.uint8), cmap='gray')
            plt.pause(0.1)
        
        plt.close()

    env.close()
    print("src/env.py test successful")


def test_model():

    import torch
    from torchinfo import summary
    from src.model import Net

    ACT_DIM = (2, 2, 3, 2, 2, 2)
    DEVICE, BATCH_SIZE, ZDIM, NUM_FRAMES = torch.device('cuda'), 8, 256, 5

    model = Net(ZDIM, ACT_DIM, BATCH_SIZE)
    model.to(DEVICE)
    rand_input = torch.rand(
        NUM_FRAMES, BATCH_SIZE, ZDIM, device=DEVICE, dtype=torch.float32
    )

    # summary(model, input_data=rand_input, verbose=1) # remove MultiCategorical while using summary
    policy, value = model(rand_input)
    print(policy.sample(), value, sep='\n')
    print("src/model.py test successful")


def test_ppo():

    from torch import optim
    from torch.utils.tensorboard import SummaryWriter
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from src.ppo import PPO
    from src.model import Net
    from src.utils import Logger, make_env
    from src.vae.model import ConvVAE, Encoder, Decoder

    DEVICE, BUFFER_SIZE, NUM_FRAMES, NUM_ENVS, LR, ZDIM = (
        'cuda',
        8,
        5,
        1,
        1e-3,
        256,
    )
    env = SubprocVecEnv(
        [make_env(id) for id in range(NUM_ENVS)], start_method='spawn'
    )
    obs_shape, act_shape = env.observation_space.shape, env.action_space.nvec

    vae = ConvVAE(obs_shape, Encoder, Decoder, ZDIM)
    vae.to(DEVICE)
    lstm = Net(ZDIM + 4, act_shape, NUM_ENVS)
    lstm.reset(BUFFER_SIZE, NUM_ENVS)
    lstm.to(DEVICE)

    buf_args = {
        'buf_size': BUFFER_SIZE,
        'num_envs': NUM_ENVS,
        'zdim': ZDIM + 4,
        'act_dim': act_shape,
        'num_frames': NUM_FRAMES,
        'gamma': PPO.GAMMA,
        'lam': PPO.LAMBDA,
    }
    optimizer = optim.Adam(lstm.parameters(), lr=LR)
    writer = SummaryWriter('/tmp/tensorboard')
    logger = Logger(writer)

    ppo = PPO(env, vae, lstm, optimizer, logger, DEVICE, **buf_args)
    ppo.rollout()
    ppo.train()
    env.close()
    print("src/ppo.py test successful")


def test_vae_model():

    import torch
    from torchinfo import summary
    from src.vae.model import ConvVAE, Encoder, Decoder

    OBS_DIM = (600, 400, 1)
    DEVICE, BATCH_SIZE = 'cuda', 8

    rand_input = torch.randint(
        0,
        255,
        (BATCH_SIZE, OBS_DIM[-1], *OBS_DIM[:-1]),
        device=DEVICE,
        dtype=torch.float32,
    )
    vae = ConvVAE(OBS_DIM, Encoder, Decoder, 128)
    vae.to(DEVICE)

    # summary(vae, input_data=rand_input, verbose=1)
    recons_image_sto, _, _ = vae(rand_input)
    recons_image_det = vae.reconstruct(rand_input)
    # print(recons_image_sto.shape, recons_image_det.shape)
    print("src/vae/model.py test successful")


if __name__ == "__main__":
    test_env()
    #test_model()
    #test_ppo()
    #test_vae_model()
