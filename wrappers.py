from abc import abstractmethod
from typing import *

import gym
from toybox import Input
from toybox.envs.atari.base import ToyboxBaseEnv


# thin layer for feature-based toybox environment
# ToyboxBaseEnv is already a gym wrapper as a subclass of gym.atari.AtariEnv
class FeatureVecWrapper(gym.ObservationWrapper):
    def __init__(self, tbenv: ToyboxBaseEnv):
        super().__init__(tbenv)
        self.env = tbenv
        # note: self.env is a toybox.env and will also have its own self.env.toybox
        self.toybox = tbenv.toybox

    # abstract method for gym.ObservationWrapper
    # this can be a good place to return a custom state feature vector
    @abstractmethod
    def observation(self, observation):
        return [1]

    def step(self, action: Union[int, Input]):
        if type(action) == int:
            return self.step_ale(action)
        elif type(action) == Input:
            return self.step_toybox_actions(action)
        else:
            return self.step_ale(int(action))

    def step_ale(self, action: int):
        # this is a little clunky because self.env.step returns the RGB state
        _, reward, done, info = self.env.step(action)
        # and we could skip right to the feature vec
        # step_toybox_actions avoids this extra work
        state_vec = self.observation(1)
        return state_vec, reward, done, info

    def step_toybox_actions(self, action_input: Input):
        obs_state_vec = None
        reward = None
        done = False
        info = {}

        assert type(action_input) == Input
        self.env.toybox.apply_action(action_input)

        if self.toybox.game_over():
            print("GAME OVER")
            info["cached_state"] = self.toybox.to_state_json()

        obs_state_vec = self.observation(1)

        # Compute the reward from the current score and reset the current score.
        # this gives the raw reward
        # and would require an additional gym.RewardWrapper to use other reward schemes
        # e.g. clipped rewards are common for Atari
        score = self.toybox.get_score()
        reward = max(score - self.env.score, 0)
        self.env.score = score

        # Check whether the episode is done
        done = self.toybox.game_over()

        # Send back diagnostic information
        info["lives"] = self.toybox.get_lives()
        # info['frame'] = frame
        info["score"] = 0 if done else self.env.score

        return obs_state_vec, reward, done, info

