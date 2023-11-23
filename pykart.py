import pystk
import cv2
import numpy as np
import time

class FPSLogger:

    def __init__(self):
        self.fps_file = "fps.txt"
        self.buffer_length = 1000
        self.fps_buffer = np.zeros(self.buffer_length)
        self.buffer_index = 0

    def add_fps(self, fps):
        self.fps_buffer[self.buffer_index] = fps
        self.buffer_index += 1

        if self.buffer_index == self.buffer_length:
            with open(self.fps_file, "ab") as f:
                np.savetxt(f, self.fps_buffer)
            self.buffer_index = 0

logger = FPSLogger()

config = pystk.GraphicsConfig.sd()
config.screen_width = 800
config.screen_height = 600
pystk.init(config)

config = pystk.RaceConfig()
config.num_kart = 2 # Total number of karts
config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL
config.laps = 1000

config.track = 'scotland'

race = pystk.Race(config)

race.start()
for n in range(10000):
    start = time.time()
    race_ended = race.step()
    end = time.time()
    logger.add_fps(end - start)

    #print((race.render_data[0].image.shape))
    image = cv2.cvtColor(race.render_data[0].image, cv2.COLOR_BGR2RGB) 
    cv2.imshow('', image)
    cv2.waitKey(1)

race.stop()
del race
