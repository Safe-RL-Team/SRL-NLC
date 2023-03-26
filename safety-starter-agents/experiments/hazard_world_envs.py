from safety_gym.envs.engine import Engine

import gym
from gym import Env

from copy import deepcopy
import json
import random
from pathlib import Path

DEFAULT = {
    "robot_base": "xmls/point.xml",
    "task": "goal",
    "observe_goal_lidar": True,
    "observe_box_lidar": True,
    "lidar_max_dist": 3,
    "lidar_num_bins": 16,
    "placements_extents": [-2, -2, 2, 2],
    "goal_size": 0.3,
    "goal_keepout": 0.305,
    # puddles
    "hazards_size": 0.2,
    "hazards_keepout": 0.18,
    "observe_hazards": True,
    "hazards_num": 4,
    # vases
    "observe_vases": True,
    "vases_num": 4,
    # gremlins
    "gremlins_travel": 0.35,
    "gremlins_keepout": 0.4,
    "observe_gremlins": True,
    "gremlins_num": 4,
    # button
    "buttons_num": 4,
    "buttons_size": 0.1,
    "buttons_keepout": 0.2,
    "observe_buttons": True,
    # pillars
    "observe_pillars": True,
    "pillars_num": 4,
}

pillar_env_config = deepcopy(DEFAULT)
pillar_env_config["constrain_pillars"] = True

hazard_env_config = deepcopy(DEFAULT)
hazard_env_config["constrain_hazards"] = True

vases_env_config = deepcopy(DEFAULT)
vases_env_config["constrain_vases"] = True

gremlins_env_config = deepcopy(DEFAULT)
gremlins_env_config["constrain_gremlins"] = True

buttons_env_config = deepcopy(DEFAULT)
buttons_env_config["constrain_buttons"] = True


class HW3D(Env):
    def __init__(self) -> None:
        super().__init__()

        self.envs = [
            (Engine(pillar_env_config), "pillars"),
            (Engine(hazard_env_config), "hazards"),
            (Engine(vases_env_config), "vases"),
            (Engine(gremlins_env_config), "gremlins"),
            (Engine(buttons_env_config), "buttons"),
        ]
        # store configs for sequential environment
        self.curr_env = None
        self.observation_space = self.envs[0][0].observation_space
        self.action_space = self.envs[0][0].action_space
        self.cost_lim = 0
        self.env_type = "budgetary"

    def reset(self):
        self.curr_env, self.constrained_object = random.choice(self.envs)
        return self.curr_env.reset()

    def get_constraint(self, constraint_dict):
        lst = constraint_dict[self.env_type][
            self.constrained_object + str(self.cost_lim)
        ]
        return random.choice(lst)

    def render(self, **kwargs):
        return self.curr_env.render(**kwargs)

    def __str__(self):
        return self.envs[0][0].__str__()

    def step(self, action):
        return self.curr_env.step(action)


class HW3DVariableCost(HW3D):
    def __init__(self):
        super(HW3DVariableCost, self).__init__()
        self.reset_cost()

    def reset_cost(self):
        self.cost_lim = random.randint(0, 4) * 5

    def reset(self):
        self.reset_cost()
        return super().reset()


class HW3DRelational(HW3D):
    def __init__(self) -> None:
        super().__init__()
        self.env_type = "relational"

    def reset_min_distance(self):
        self.curr_env.min_distance = random.randint(0, 5)

    def get_constraint(self, constraint_dict):
        lst = constraint_dict[self.env_type][
            self.constrained_object + str(self.curr_env.min_distance)
        ]
        return random.choice(lst)

    def reset(self):
        ret = super().reset()
        self.reset_min_distance()
        return ret


class HW3DS(Env):
    def __init__(self):
        super().__init__()
        self.configs = [
            (pillar_env_config, "pillars"),
            (hazard_env_config, "hazards"),
            (vases_env_config, "vases"),
            (gremlins_env_config, "gremlins"),
            (buttons_env_config, "buttons"),
        ]
        self.cost_lim = 0
        self.env_type = "sequential"
        self.is_second_phase = False
        self._randomize_env()

        self.observation_space = self.curr_env.observation_space
        self.action_space = self.curr_env.action_space

    def _randomize_env(self):
        self.first_config, self.first_obj = random.choice(self.configs)
        self.curr_env = Engine(self.first_config)
        self.second_config, self.second_obj = random.choice(self.configs)

    def reset(self):
        self._randomize_env()
        return self.curr_env.reset()

    def get_constraint(self, dict):
        lst = dict[self.env_type][self.first_obj + self.second_obj]
        return random.choice(lst)

    def step(self, action):
        obs, reward, done, infos = self.curr_env.step(action)

        # transition between environments:
        if not self.is_second_phase and infos["cost"] > 0:
            self.is_second_phase = True
            self.curr_env.parse(self.second_config)

        return obs, reward, done, infos

    def render(self, **kwargs):
        return self.curr_env.render(**kwargs)

    def __str__(self):
        return self.curr_env.__str__()


class HW3DWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        this_folder = Path(__file__).parent
        templates_path = this_folder.joinpath("templates.json")
        json_path = Path(templates_path)
        with json_path.open(mode="r") as json_file:
            self.missions = json.load(json_file)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        constraint = self.env.get_constraint(self.missions)
        episode_info = {"constraint": constraint}
        return obs, episode_info