import unittest

import gym
from toybox.interventions.space_invaders import SpaceInvadersIntervention

from .space_invaders_feature_vec_wrapper import *


class TestSpaceInvadersFeatureVecWrapper(unittest.TestCase):
    def setUp(self):
        self.env = {
            "start": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "fewer_enemies": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "not_under_laser": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "under_laser": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "shield_right": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "shield_left": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "shield_special": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
            "fewer_shields": gym.make("SpaceInvadersToyboxNoFrameskip-v4"),
        }
        self.wrapper = {
            "start": SpaceInvadersFeatureVecWrapper(self.env["start"]),
            "fewer_enemies": SpaceInvadersFeatureVecWrapper(self.env["fewer_enemies"]),
            "not_under_laser": SpaceInvadersFeatureVecWrapper(self.env["not_under_laser"]),
            "under_laser": SpaceInvadersFeatureVecWrapper(self.env["under_laser"]),
            "shield_right": SpaceInvadersFeatureVecWrapper(self.env["shield_right"]),
            "shield_left": SpaceInvadersFeatureVecWrapper(self.env["shield_left"]),
            "shield_special": SpaceInvadersFeatureVecWrapper(
                self.env["shield_special"]
            ),
            "fewer_shields": SpaceInvadersFeatureVecWrapper(
                self.env["fewer_shields"]
            ),
        }

        with SpaceInvadersIntervention(
            self.env["fewer_enemies"].toybox
        ) as intervention:
            enemies = [e for e in intervention.game.enemies]
            for i, e in enumerate(enemies):
                if i % 5 == 0 or i % 7 == 0 or i >= 30:
                    e.alive = False

        example_enemy_laser = {
            'x': 150, 'y': 113, 'w': 2, 
            'h': 11, 't': 0, 'movement': 'Down', 'speed': 3, 
            'color': {'r': 144, 'b': 144, 'g': 144, 'a': 255}
        }
        with SpaceInvadersIntervention(self.env["not_under_laser"].toybox) as intervention:
            not_under_laser = example_enemy_laser
            not_under_laser["x"] = 150
            intervention.game.enemy_lasers = [not_under_laser]

        with SpaceInvadersIntervention(self.env["under_laser"].toybox) as intervention:
            under_laser = example_enemy_laser
            under_laser["x"] = intervention.game.ship.x
            intervention.game.enemy_lasers = [under_laser]

        with SpaceInvadersIntervention(self.env["shield_right"].toybox) as intervention:
            for shield in intervention.game.shields:
                shield.x += 25

        with SpaceInvadersIntervention(self.env["shield_left"].toybox) as intervention:
            for shield in intervention.game.shields:
                shield.x -= 25

        with SpaceInvadersIntervention(
            self.env["shield_special"].toybox
        ) as intervention:
            i = 0
            for shield in intervention.game.shields:
                if i == 0:
                    shield.x -= 40
                if i == 1:
                    shield.x -= 80
                i = i + 1
        pass

        with SpaceInvadersIntervention(self.env["fewer_shields"].toybox) as intervention:
            intervention.game.shields=[]

    def test_ship_x(self):
        state = self.env["start"].toybox.state_to_json()
        expected = state["ship"]["x"]
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ship_x]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_laser_mid_air(self):
        expected = 0
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ship_laser_mid_air]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_num_enemies(self):
        expected = 36
        actual = self.wrapper["start"].observation(None)[INDEX_OF_num_enemies]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_num_fewer_enemies(self):
        expected = 20
        actual = self.wrapper["fewer_enemies"].observation(None)[INDEX_OF_num_enemies]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_lowest_enemy_height(self):
        expected = 62
        actual = self.wrapper["start"].observation(None)[INDEX_OF_lowest_enemy_height]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_lowest_enemy_height_fewer_enemies(self):
        expected = 80
        actual = self.wrapper["fewer_enemies"].observation(None)[
            INDEX_OF_lowest_enemy_height
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_in_danger_vacuous(self):
        expected = False
        actual = self.wrapper["start"].observation(None)[INDEX_OF_in_danger]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_in_danger_false(self):
        expected = False
        actual = self.wrapper["not_under_laser"].observation(None)[INDEX_OF_in_danger]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_in_danger_true(self):
        expected = True
        actual = self.wrapper["under_laser"].observation(None)[INDEX_OF_in_danger]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ufo_on_screen_false(self):
        expected = False
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ufo_on_screen]
        self.assertAlmostEqual(expected, actual)

        # TODO: add test where ufo IS on screen
        pass

    def test_ufo_sign_distance(self):
        expected = -70
        actual = self.wrapper["start"].observation(None)[INDEX_OF_ufo_sign_distance]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under(self):
        expected = True
        actual = self.wrapper["start"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envshield_right(self):
        expected = False
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_completely_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envshield_right(self):
        expected = False
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envshield_right(self):
        expected = 25
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envshield_right(self):
        expected = 41
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envshield_right(self):
        expected = 0
        actual = self.wrapper["shield_right"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envshield_left(self):
        expected = False
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_completely_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envshield_left(self):
        expected = True
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envshield_left(self):
        expected = 0
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envshield_left(self):
        expected = -9
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envshield_left(self):
        expected = 8
        actual = self.wrapper["shield_left"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envshield_special(self):
        expected = True
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_completely_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envshield_special(self):
        expected = True
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_partially_under_shield
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envshield_special(self):
        expected = 0
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envshield_special(self):
        expected = 0
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envshield_special(self):
        expected = 17
        actual = self.wrapper["shield_special"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete
        ]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_completely_under_envfewershields(self):
        expected = False
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_completely_under_shield]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_ship_partially_under_envfewershields(self):
        expected = False
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_partially_under_shield]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_partial_envfewershields(self):
        expected = 320
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_partial]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_shield_complete_envfewershields(self):
        expected = 320
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_sign_distance_closest_shield_complete]
        self.assertAlmostEqual(expected, actual)
        pass

    def test_sign_distance_closest_UN_shield_complete_envfewershields(self):
        expected = 0
        actual = self.wrapper["fewer_shields"].observation(None)[
            INDEX_OF_sign_distance_closest_UN_shield_complete]
        self.assertAlmostEqual(expected, actual)
        pass


if __name__ == "__main__":
    unittest.main()
