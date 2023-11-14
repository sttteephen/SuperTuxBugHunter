import pystk
import cv2
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from torchvision import transforms as T
import numpy as np
from sympy import Point3D, Line3D
import torch
from matplotlib import pyplot as plt

class STKAgent:
    """
    SuperTuxKart agent for handling actions and getting state information from the environment.
    The `STKEnv` class passes on the actions to this class for it to handle and gets the current
    state(image) and various other info.

    :param graphicConfig: `pystk.GraphicsConfig` object specifying various graphic configs
    :param raceConfig: `pystk.RaceConfig` object specifying the track and other configs
    """

    def __init__(self, graphicConfig: pystk.GraphicsConfig, raceConfig: pystk.RaceConfig, id=1):

        pystk.init(graphicConfig)
        self.id = id
        self.node_idx = 0
        self.started = False
        self.observation_shape = (graphicConfig.screen_height, graphicConfig.screen_width, 3)
        self.graphicConfig = graphicConfig
        self.race = pystk.Race(raceConfig)
        self.reverse = raceConfig.reverse
        self.track = pystk.Track()
        self.state = pystk.WorldState()
        self.currentAction = pystk.Action()
        self.image = np.zeros(self.observation_shape, dtype=np.uint8) # np.float32
        self.AI = True
        for player in raceConfig.players:
            if player.controller == pystk.PlayerConfig.Controller.PLAYER_CONTROL:
                self.AI = False

    def _check_nitro(self) -> bool:
        kartLoc = np.array(self.playerKart.location)
        nitro = [pystk.Item.Type.NITRO_SMALL, pystk.Item.Type.NITRO_BIG]

        for item in self.state.items:
            if item.type in nitro:
                itemLoc = np.array(item.location)
                squared_dist = np.sum((kartLoc - itemLoc) ** 2, axis=0)
                dist = np.sqrt(squared_dist)
                if dist <= 1:
                    return True
        return False

    def _compute_lines(self, nodes):
        return [Line3D(*node) for node in nodes]

    def _update_node_idx(self):
        dist_down_track = (
            0
            if self.reverse and self.playerKart.overall_distance <= 0
            else self.playerKart.distance_down_track
        )
        path_dist = self.path_distance[self.node_idx]
        while not (path_dist[0] <= dist_down_track <= path_dist[1]):
            if dist_down_track < path_dist[0]:
                self.node_idx -= 1
            elif dist_down_track > path_dist[1]:
                self.node_idx += 1
            path_dist = self.path_distance[self.node_idx]

    def _get_powerup(self):
        return self.playerKart.powerup.type

    def _get_attachment(self):
        return self.playerKart.attachment.type

    def _get_finish_time(self) -> int:
        return int(self.playerKart.finish_time)

    def _get_overall_distance(self) -> int:
        return max(0, self.playerKart.overall_distance)

    def _get_kart_dist_from_center(self):
        # compute the dist b/w the kart and the center of the track
        # should have called self._update_node_idx() before calling this to avoid errors
        location = self.playerKart.location
        path_node = self.path_nodes[self.node_idx]
        return path_node.distance(Point3D(location)).evalf()

    def _get_is_inside_track(self):
        # should i call this inside step?
        # divide path_width by 2 because it's the width of the current path node
        # and the dist of kart is from the center line
        self._update_node_idx()
        curr_path_width = self.path_width[self.node_idx][0]
        kart_dist = self._get_kart_dist_from_center()
        return kart_dist <= curr_path_width / 2

    def _get_velocity(self):
        # returns the magnitude of velocity
        return np.sqrt(np.sum(np.array(self.playerKart.velocity) ** 2))

    def _update_action(self, action: list):
        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        # action_space = [2, 2, 3, 2, 2, 2, 2]
        self.currentAction.acceleration = action[0]
        self.currentAction.brake = bool(
            max(0, action[1] - action[0])
        )  # only True when acc is not 1
        self.currentAction.steer = action[2] - 1
        self.currentAction.fire = bool(action[3])
        self.currentAction.drift = bool(action[4])
        self.currentAction.nitro = bool(action[5])
        # self.currentAction.rescue = bool(action[6])
        self.currentAction.rescue = False

    def get_env_info(self) -> dict:
        info = {}
        info['id'] = self.id
        info['laps'] = self.race.config.laps
        info['track'] = self.race.config.track
        info['reverse'] = self.race.config.reverse
        info['num_kart'] = self.race.config.num_kart
        info['step_size'] = self.race.config.step_size
        info['difficulty'] = self.race.config.difficulty
        return info

    def get_info(self) -> dict:
        info = {}
        info["done"] = self.done()
        info["nitro"] = self._check_nitro()
        info["powerup"] = self._get_powerup()
        info["velocity"] = self._get_velocity()
        info["attachment"] = self._get_attachment()
        info["finish_time"] = self._get_finish_time()
        info["is_inside_track"] = self._get_is_inside_track()
        info["overall_distance"] = self._get_overall_distance()
        return info

    def done(self) -> bool:
        """
        `playerKart.finish_time` > 0 when the kart finishes the race.
        Initially the finish time is < 0.k
        """
        return self.playerKart.finish_time > 0

    def reset(self):
        #print('resetting')
        # changed this
        if self.started:
            #return self.image, self.get_info()
            self.race.restart()
        else:
            self.race.start()

        self.backward = 0
        self.no_movement = 0
        self.out_of_track_count = 0

        self.started = True
        self._update_action([1, 0, 1, 0, 0, 0, 0])
        for _ in range(10):
            self.race.step(self.currentAction)
            self.state.update()
            self.track.update()
            self.image = np.array(self.race.render_data[0].image, dtype=np.uint8) #np.float32

        self.playerKart = self.state.players[0].kart
        self.path_width = np.array(self.track.path_width)
        self.path_distance = np.array(
            sorted(self.track.path_distance[::-1], key=lambda x: x[0])
            if self.reverse
            else self.track.path_distance
        )
        self.path_nodes = np.array(self._compute_lines(self.track.path_nodes))
        self._update_node_idx()
        return self.image, self.get_info()

    def step(self, action=None):
        if self.AI:
            self.race.step()
        else:
            self._update_action(action)
            self.race.step(self.currentAction)

        info = self.get_info()
        self.state.update()
        self.track.update()
        self.image = np.array(self.race.render_data[0].image, dtype=np.uint8) # np.float32
        done = self.done()

        return self.image, 0, done, done, info

    def close(self):
        self.race.stop()
        del self.race
        pystk.clean()


class STKEnv(gym.Env):
    """
    A simple gym compatible STK environment for controlling the kart and interacting with the
    environment. The goal is to place 1st among other karts.

    Observation:
        Image of shape `self.observation_shape`.

    Actions:
        -----------------------------------------------------------------
        |         ACTIONS               |       POSSIBLE VALUES         |
        -----------------------------------------------------------------
        |       Acceleration            |           (0, 1)              |
        |       Brake                   |           (0, 1)              |
        |       Steer                   |         (-1, 0, 1)            |
        |       Fire                    |           (0, 1)              |
        |       Drift                   |           (0, 1)              |
        |       Nitro                   |           (0, 1)              |
        |       Rescue                  |           (0, 1)              |
        -----------------------------------------------------------------

    References:
        1. https://blog.paperspace.com/creating-custom-environments-openai-gym
    """

    def __init__(self, env: STKAgent):
        super(STKEnv, self).__init__()
        self.env = env
        self.observation_shape = self.env.observation_shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8
            #low=np.zeros(self.env.observation_shape),
            #high=np.full(self.env.observation_shape, 255, dtype=np.float32),
        )

        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        self.action_space = MultiDiscrete([2, 2, 3, 2, 2, 2])

    def step(self, action):
        if action is not None:
            assert self.action_space.contains(action), f'Invalid Action {action}'
        return self.env.step(action)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def render(self, mode: str = 'human'):
        return self.env.image

    def get_info(self):
        return self.env.get_info()

    def get_env_info(self):
        return self.env.get_env_info()

    def close(self):
        self.env.close()


class STKReward(gym.Wrapper):

    FINISH = 1
    VELOCITY = 0.4
    COLLECT_POWERUP = 0.2
    USE_POWERUP = 0.2
    DRIFT = 0.2
    NITRO = 0.2
    EARLY_END = -1
    NO_MOVEMENT = -0.2
    OUT_OF_TRACK = -0.4
    BACKWARDS = -0.7

    def __init__(self, env: STKEnv):
        # TODO: handle rewards for attachments
        # TODO: rewards for using powerup - only if it hits other karts
        # TODO: change value of USE_POWERUP when accounted for hitting other karts
        super(STKReward, self).__init__(env)
        self.observation_shape = self.env.observation_shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8
            #low=np.zeros(self.env.observation_shape),
            #high=np.full(self.env.observation_shape, 255, dtype=np.float32),
        )
        self.reward = 0
        self.backward = 0  # number of times the kart goes backwards
        self.prevInfo = None
        self.no_movement = 0 # number of times the kart doesn't move
        self.out_of_track_count = 0 
        self.backward_threshold = 50 #number of times the kart can go backwards
        self.no_movement_threshold = 5 # number of times the kart can stay still
        self.out_of_track_threshold = 50 # number of times the kart can go out of track

    def _get_reward(self, action, info):

        reward = -0.02
        if self.prevInfo is None:
            self.prevInfo = info

        #  0             1      2      3     4      5      6
        # {acceleration, brake, steer, fire, drift, nitro, rescue}
        # [2,            2,     3,     2,    2,     2,     2]   # action_space
        if action is not None:
            if action[5] and info["nitro"]:
                reward += STKReward.NITRO
            if action[4] and info["velocity"] > 10:
                reward += STKReward.DRIFT
            if action[3] and info["powerup"].value:
                reward += STKReward.USE_POWERUP

        if info["done"]:
            reward += STKReward.FINISH

        # agent might purposely slow down and speed up again to get rewards, so add the or term
        if info["velocity"] > (self.prevInfo["velocity"] + 1) or info["velocity"] > 29:
            reward += STKReward.VELOCITY

        if not info["is_inside_track"]:
            reward += STKReward.OUT_OF_TRACK
            self.out_of_track_count += 1
            if self.out_of_track_count > self.out_of_track_threshold:
                info["early_end"] = True
                info["early_end_reason"] = "Outside track"

        delta_dist = info["overall_distance"] - self.prevInfo["overall_distance"]
        if delta_dist < 0:
            #print('delta_dist')
            reward += STKReward.BACKWARDS
            self.backward += 1
        elif delta_dist == 0:
            self.no_movement += 1
        elif delta_dist > 5:
            reward += (delta_dist) / 10
        else:
            reward += max(0, delta_dist)

        if self.no_movement >= self.no_movement_threshold:
            reward += STKReward.NO_MOVEMENT
            self.no_movement = 0

        if info["powerup"].value and not self.prevInfo["powerup"].value:
            reward += STKReward.COLLECT_POWERUP

        if self.backward >= self.backward_threshold:
            info["early_end"] = True
            info["early_end_reason"] = "Going backwards"
            self.backward = 0

        if info.get("early_end", False):
            reward += STKReward.EARLY_END

        self.prevInfo = info
        return np.clip(reward, -5, 5)

    def step(self, action):
        state, reward, done, done, info = self.env.step(action)
        #print('1', info)
        if len(info) > 1:
            reward = self._get_reward(action, info)
            #print('2', info)
            if info.get("early_end", False):
                done = True
                #print(f'env_id: {self.env.env.id} - {info.get("early_end_reason", "None")}')
                info = {}
        return state, reward, done, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.observation_shape = self.env.observation_shape[:2]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.observation_shape[0], self.observation_shape[1], 1),
            dtype=np.uint8
            #low=np.zeros(self.env.observation_shape),
            #high=np.full(self.env.observation_shape, 255, dtype=np.float32),
        )
        self.transform = T.Grayscale()

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        return torch.from_numpy(observation)

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, axis=2)
        #plt.imshow(np.array(obs).astype(np.uint8), cmap='gray')
        #plt.pause(0.1)
        return obs
        #return self.transform(self.permute_orientation(obs)).squeeze(dim=0)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.observation_shape = self.env.observation_shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8
            #low=np.zeros(self.env.observation_shape),
            #high=np.full(self.env.observation_shape, 255, dtype=np.float32),
        )

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward / self._skip, done, done, info