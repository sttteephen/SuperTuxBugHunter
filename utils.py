import pystk
import numpy as np

from random import choice


class STK:

    TRACKS = [
        'abyss',
        'black_forest',
        'candela_city',
        'cocoa_temple',
        'cornfield_crossing',
        'fortmagma',
        'gran_paradiso_island',
        'hacienda',
        'lighthouse',
        'minigolf',
        'olivermath',
        'ravenbridge_mansion',
        'sandtrack',
        'scotland',
        'snowmountain',
        'snowtuxpeak',
        'stk_enterprise',
        'volcano_island',
        'xr591',
        'zengarden',
    ]

    KARTS = [
        'adiumy',
        'amanda',
        'beastie',
        'emule',
        'gavroche',
        'gnu',
        'hexley',
        'kiki',
        'konqi',
        'nolok',
        'pidgin',
        'puffy',
        'sara_the_racer',
        'sara_the_wizard',
        'suzanne',
        'tux',
        'wilber',
        'xue',
    ]

    GRAPHICS = {
        "hd": pystk.GraphicsConfig.hd,
        "sd": pystk.GraphicsConfig.sd,
        "ld": pystk.GraphicsConfig.ld,
        "none": pystk.GraphicsConfig.none,
    }

    WIDTH = 200 # 600
    HEIGHT = 150 # 400

    #@staticmethod
    def get_graphic_config(quality='ld'):
        config = STK.GRAPHICS[quality]()
        config.screen_width = STK.WIDTH
        config.screen_height = STK.HEIGHT
        config.animated_characters = False
        config.bloom = False
        config.dof = False
        config.dynamic_lights = False
        config.glow = False
        config.motionblur - False
        config.particles_effects = 0
        config.high_definition_textures = 0
        config.texture_compression = True
        config.display_adapter = 0

        return config

    #@staticmethod
    def get_race_config(
        track=None,
        kart=None,
        numKarts=5,
        laps=1,
        reverse=False,
        difficulty=1,
        vae=False,
    ):

        config = pystk.RaceConfig()
        config.difficulty = 1
        config.num_kart = 1 # numKarts
        config.reverse = False # np.random.choice([True, False])
        config.step_size = 0.045
        config.track = 'scotland' # track
        config.laps = 1
        config.players[0].team = 0
        config.players[0].kart = 'tux' # kart
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL


        return config


def make_env(id: int, quality='ld', race_config_args={}):
    """
    Utility function to create an env.

    :param env_id: (str) the environment ID
    :return: (Callable)
    """

    import gym

    def _init() -> gym.Env:
        from env import (
            STKAgent,
            STKEnv,
            STKReward,
            SkipFrame,
            GrayScaleObservation,
        )

        env = STKAgent(
            STK.get_graphic_config(quality),
            STK.get_race_config(**race_config_args),
            id,
        )
        env = STKEnv(env)
        env = STKReward(env)
        env = SkipFrame(env, 2)
        env = GrayScaleObservation(env)
        return env

    return _init
    