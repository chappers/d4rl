#!/usr/bin/env python
# -*- coding: utf-8 -*-

from d4rl.gym_minigrid.minigrid import *
from d4rl.gym_minigrid.register import register
from d4rl.pointmaze.gridcraft import grid_env
from d4rl.pointmaze.gridcraft import grid_spec
import numpy as np
import random


MAZE = \
"###################\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"###################\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"#OOOOOOOO#OOOOOOOO#\\"+\
"###################\\"




class FourRoomsBaseEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, **kwargs):
        self._agent_default_pos = agent_pos
        if goal_pos is None:
            goal_pos = (12, 12)
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=19, max_steps=100, **kwargs)

    def get_target(self):
        return self._goal_default_pos

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        # print(self._goal_default_pos)
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info



class FourRoomsEnv(FourRoomsBaseEnv):
    def __init__(self, agent_pos=None, goal_pos=None, reward_type="sparse", **kwargs):
        self.seed_n = random.choice(range(99999999))
        self.curr_reward = 0
        self.seed(self.seed_n)
        FourRoomsBaseEnv.__init__(self, agent_pos=agent_pos, goal_pos=goal_pos, **kwargs)
        self.reward_type = reward_type
        self.seed(self.seed_n)
        self.reset()

    def reset(self):
        self.curr_reward = 0
        return FourRoomsBaseEnv.reset(self)
        

    def _gen_grid(self, width, height):
        self.seed(self.seed_n)
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        # print(self._goal_default_pos)
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                # done = True
                self._agent_default_pos = self.agent_pos
                self._agent_dir = self.agent_dir
                
                env = grid_env.GridEnv(grid_spec.spec_from_string(MAZE))  # note that the maze will be blocked
                reset_locations = list(zip(*np.where(env.gs.spec == grid_spec.EMPTY)))
                # self.step_count = 0
                random_loc = random.choice(reset_locations)
                # reset goal...
                self._goal_default_pos = random_loc

                self.seed(self.seed_n)
                self._gen_grid(self.width, self.height)
                self.agent_dir = self._agent_dir

            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        reward = self._reward()
        return obs, reward, done, {}


    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        # return 1 - 0.9 * (self.step_count / self.max_steps)
        if self.reward_type == 'cumulative_dense':
            base_reward = np.exp(-np.linalg.norm(self.agent_pos - self.get_target()))
            reward = self.curr_reward + base_reward
            if base_reward > 0.99:
                self.curr_reward = reward
        elif self.reward_type == 'cumulative_sparse':
            base_reward = 1.0 if np.linalg.norm(self.agent_pos - self.get_target()) <= 0.5 else 0.0
            reward = self.curr_reward + base_reward
            if base_reward == 1:
                self.curr_reward = reward
        elif self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(self.agent_pos - self.get_target()) <= 0.5 else 0.0
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(self.agent_pos - self.get_target()))
        return reward
        


register(
    id='MiniGrid-FourRooms-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv'
)
