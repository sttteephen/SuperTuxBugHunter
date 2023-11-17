import pystk
import cv2
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
from sympy import Point3D, Line3D

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
        self.image = np.zeros(self.observation_shape, dtype=np.uint8)
        self.AI = True
        for player in raceConfig.players:
            if player.controller == pystk.PlayerConfig.Controller.PLAYER_CONTROL:
                self.AI = False

    def _compute_lines(self, nodes):
        return [Line3D(*node) for node in nodes]

    def _update_node_idx(self):
        dist_down_track = self.playerKart.distance_down_track   # distance travelled on current lap
        path_dist = self.path_distance[self.node_idx]   # distance of current node
        
        while not (path_dist[0] <= dist_down_track <= path_dist[1]):
            if dist_down_track < path_dist[0]:
                self.node_idx -= 1
            elif dist_down_track > path_dist[1]:
                self.node_idx += 1
            path_dist = self.path_distance[self.node_idx]


    def _get_overall_distance(self) -> int:
        return max(0, self.playerKart.overall_distance)

    def _get_kart_dist_from_center(self):
        # compute the dist b/w the kart and the center of the track
        # should have called self._update_node_idx() before calling this to avoid errors
        location = self.playerKart.location
        path_node = self.path_nodes[self.node_idx]
        return path_node.distance(Point3D(location)).evalf()

    def _get_game_time(self):
        return self.state.time

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
        self._update_node_idx()
        info["overall_distance"] = self._get_overall_distance()
        info["game_time"] = self._get_game_time()
        return info

    def reset(self):
        if self.started:
            self.race.restart()

            self._update_action([1, 0, 1, 0, 0, 0, 0])
            for _ in range(10):
                self.race.step(self.currentAction)
                self.state.update()
                self.track.update()
                self.image = np.array(self.race.render_data[0].image, dtype=np.uint8)

        else:
            self.race.start()
            
            self._update_action([1, 0, 1, 0, 0, 0, 0])
            for _ in range(10):
                self.race.step(self.currentAction)
                self.state.update()
                self.track.update()
                self.image = np.array(self.race.render_data[0].image, dtype=np.uint8)

            self.started = True
            self.playerKart = self.state.players[0].kart
            self.path_width = np.array(self.track.path_width) # an array containing the width of each path segment
            self.path_distance = np.array(self.track.path_distance) # an array containing the distance of each path segment
            # an array containing lines (3D start point, 3D end point) of each path segment
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
        self.image = np.array(self.race.render_data[0].image, dtype=np.uint8)
        
        #image = cv2.cvtColor(self.race.render_data[0].image, cv2.COLOR_BGR2RGB) 
        #cv2.imshow('', image)
        #cv2.waitKey(1)

        terminated = info["game_time"] > 20
        truncated = terminated

        return self.image, 0, terminated, truncated, info

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

    FINISH = 1000
    VELOCITY = 0.4

    def __init__(self, env: STKEnv):
        super(STKReward, self).__init__(env)
        self.observation_shape = self.env.observation_shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8
        )
        self.reward = 0
        self.prevInfo = None

    def _get_reward(self, action, info):

        reward = 0
        if self.prevInfo is None:
            self.prevInfo = info

        # negative reward for going out of track
        if self.env.env._get_kart_dist_from_center() > 15:
            reward = -1
        else:

            # reward for moving in the right direction
            delta_dist = info["overall_distance"] - self.prevInfo["overall_distance"]
            if delta_dist <= 0:
                reward = 0
            else:
                # maximum reward of 10 for moving in the right direction,
                # otherwise reward is proportional to the distance moved
                reward = min(10, delta_dist)

        self.prevInfo = info
        return reward

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if len(info) > 1:
            reward = self._get_reward(action, info)
            
            if done:
                terminated = True
                truncated = True

        return state, reward, terminated, truncated, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.timestep = 0
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.observation_shape[0], self.observation_shape[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, axis=2)
        return obs


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
        )

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward / self._skip, terminated, truncated, info