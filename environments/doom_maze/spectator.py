#!/usr/bin/env python
from __future__ import print_function
from ...utils import generate_video
import argparse
from time import sleep
from vizdoom import *
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('map_name', default="medium")
parser.add_argument('-fs', '--fixed_start', default=True)
flags = parser.parse_args()

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    wads_directory = current_directory + "/wads/" + ("fixed_start" if flags.fixed_start else "reset_anywhere")
    wad_directory_path = wads_directory + "/" + flags.map_name + ".wad"

    game = DoomGame()
    game.load_config("./configurations/default.cfg")
    game.set_doom_scenario_path(wad_directory_path)
    game.add_game_args("+freelook 1")
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_doom_map("MAP00")
    game.set_mode(Mode.SPECTATOR)
    game.init()
    episodes = 10

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        images = []

        game.new_episode()
        while not game.is_episode_finished():
            state: GameState = game.get_state()
            images.append(np.moveaxis(state.screen_buffer.copy(), 0, -1))
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()
        generate_video(images, current_directory, "episode_" + str(i))

        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("************************")
        sleep(2.0)

    game.close()
