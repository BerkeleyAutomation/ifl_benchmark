"""
A simple PointBot environment that can be useful for debugging.
The robot's goal is to reach the goal location specified in map.txt. 
State representation is (x, y). Action representation is (dx, dy).
"""

import os
import pickle

import os.path as osp
import numpy as np
from gym import Env
from gym import utils
from gym.spaces import Box

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import cv2
import math

"""
Constants associated with the PointBot env.
"""

MAX_FORCE = 1
HORIZON = 250
NOISE_SCALE = 0.25


class Map:

    def __init__(self, filename="env/assets/map.txt"):
        with open(filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content.reverse()
        self.num_rows = len(content)
        self.num_cols = len(content[0])
        self.map = np.zeros((self.num_rows, self.num_cols))

        self.start = None
        self.goal = None


        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if content[i][j] == " ":
                    continue
                elif content[i][j] == "#":
                    self.map[i,j] = 1
                elif content[i][j] == "S":
                    self.map[i,j] = 2
                    self.start = self.cell_to_world((i, j))
                elif content[i][j] == "G":
                    self.map[i,j] = 3
                    self.goal = self.cell_to_world((i, j))


    def world_to_cell(self, pos):
        return int(pos[1] // 5), int(pos[0] // 5)

    def cell_to_world(self, coord):
        return coord[1] * 5 + 2.5, coord[0] * 5 + 2.5

    def obs(self, pos):
        return self.map[self.world_to_cell(pos)] == 1

    def free(self, pos):
        return self.map[self.world_to_cell(pos)] != 1

    def is_start(self, pos):
        return self.map[self.world_to_cell(pos)] == 2

    def is_goal(self, pos):
        return self.map[self.world_to_cell(pos)] == 3



def process_action(a):
    return np.clip(a, -MAX_FORCE, MAX_FORCE)



class Navigation(Env, utils.EzPickle):
    def __init__(self, idx=0):
        utils.EzPickle.__init__(self)
        self.hist = self.cost = self.done = self.time = self.state = None
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.horizon = HORIZON
        self.action_space = Box(-np.ones(2) * MAX_FORCE,
                                np.ones(2) * MAX_FORCE)
        self.observation_space = Box(-np.ones(2) * float('inf'),
                                     np.ones(2) * float('inf'))
        self.max_episode_steps = HORIZON
        self.map = Map()
        self.get_offline_data = get_offline_data

    def step(self, a, noiseless=False):
        a = process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a, noiseless=noiseless)
        cur_cost = self.step_reward(next_state, a)
        self.cost.append(cur_cost)
        self.state = next_state
        self.time += 1
        self.hist.append(self.state)
        self.constraint = self.obstacle(self.state)
        self.done = self.at_goal(next_state) or self.obstacle(next_state) or self.time >= self.horizon

        return self.state, cur_cost, self.done, {
            "constraint": self.obstacle(next_state),
            "reward": cur_cost,
            "state": old_state,
            "next_state": next_state,
            "action": a,
            'success': cur_cost > 0
        }

    def obstacle(self, s):
        return self.map.obs(s)

    def reset(self, hard=False):
        self.state = self.map.start + np.random.randn(2)
        self.time = 0
        self.cost = []
        self.done = False
        self.hist = [self.state]
        self.constraint = self.obstacle(self.state)
        return self.state

    def _next_state(self, s, a, noiseless=False):
        if self.obstacle(s):
            return s
        noise = 0. if noiseless else NOISE_SCALE
        return self.A.dot(s) + self.B.dot(a) + noise * np.clip(np.random.randn(
            len(s)), -1., 1.)


    def at_goal(self, s):
        return self.map.is_goal(s)

    def step_reward(self, s, a):
        return int(self.map.is_goal(s))

    def values(self):
        return np.cumsum(np.array(self.cost)[::-1])[::-1]

    def sample(self):
        """
        samples a random action from the action space.
        """
        return np.random.random(2) * 2 * MAX_FORCE - MAX_FORCE

    def plot_trajectories(self, states=None):
        if states == None:
            states = self.hist
        states = np.array(states)
        plt.scatter(states[:, 0], states[:, 1])
        plt.savefig("trajectories.png")

    def human_action(self, s, mode='safe'):
        return teacher_action(s, self.map)


def get_offline_data(num_transitions, task_demos=False, save_rollouts=False):
    env = Navigation()
    transitions = []
    rollouts = []
    if task_demos:
        for i in range(math.ceil(num_transitions / HORIZON)):
            rollouts.append([])
            state = env.reset()
            for j in range(HORIZON):
                action = env.human_action(state)
                next_state, rew, done, _ = env.step(action, noiseless=True)
                transitions.append(
                    (state, action, rew, next_state, not done)) # last elem is a mask (1 - done)
                rollouts[-1].append(
                    (state, action, rew, next_state, not done))
                state = next_state
                if rew:
                    print('yes')
                if done:
                    break
    else:
        for i in range(num_transitions // 10):
            rollouts.append([])
            state = np.array(
                [np.random.uniform(0, 100),
                np.random.uniform(0, 100)])
            for j in range(10):
                action = np.clip(np.random.randn(2), -1, 1)
                next_state = env._next_state(state, action)
                constraint = env.obstacle(next_state)
                reward = env.step_reward(state, action)
                transitions.append(
                    (state, action, constraint, next_state, not constraint))
                rollouts[-1].append(
                    (state, action, constraint, next_state, not constraint))
                state = next_state
                if constraint:
                    break

    if save_rollouts:
        return rollouts
    else:
        return transitions

def teacher_action(state, map):
    cell = map.world_to_cell(state)
    if cell[1] == 1 and cell[0] > 1:
        goal = map.cell_to_world([1, 1])
    elif cell[0] == 1 and cell[1] < 18:
        goal = map.cell_to_world([1, 18])
    else:
        goal = map.goal
    disp = np.subtract(goal, state)
    disp[disp > MAX_FORCE] = MAX_FORCE
    disp[disp < -MAX_FORCE] = -MAX_FORCE
    return disp

