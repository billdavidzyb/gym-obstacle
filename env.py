#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import randint
from gym import Env
from gym.envs.classic_control import rendering
from object import GeomContainer, Segment, Wall, rotate
from robot import Robot, _NUM_DISTANCE_SENSOR
from math import *

UNIT_SQUARE = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]) / 2

def polyline_to_segmentlist(polyline):
    return [Segment(polyline[i - 1], polyline[i]) for i in range(len(polyline))]

class ObstacleEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.screen_width = 600
        self.screen_height = 400
        self.state = np.zeros(_NUM_DISTANCE_SENSOR, dtype=np.float32)
        self.viewer = None
        self.robot = Robot()
        self.obstacles = []
        self.edges = []
        ## add static obstacles
        for i in range(15):
            obs = GeomContainer(rendering.make_polygon(UNIT_SQUARE * 50), 
                lambda pos, angle: polyline_to_segmentlist(rotate(UNIT_SQUARE, angle) * 50 + pos))
            obs.set_color(0, 1, 0)
            self.obstacles.append(obs)
        ## add walls to surround the environment
        self.edges.append(Wall([0, 0], [self.screen_width, 0], (0, 1, 0)))
        self.edges.append(Wall([self.screen_width, 0], [self.screen_width, self.screen_height], 
                                (0, 1, 0)))
        self.edges.append(Wall([self.screen_width, self.screen_height], [0, self.screen_height], 
                                (0, 1, 0)))
        self.edges.append(Wall([0, self.screen_height], [0, 0], (0, 1, 0)))
        
        self.visible_object = []
        self.register_visible_object(self.robot)
        for obs in self.obstacles:
            self.register_visible_object(obs)
    
    # 
    def _checkCollision(self):
        num = 0
        for obs in self.obstacles:
            if num == len(self.obstacles):
                return False
            axes = self.robot.getRectAxis()
            axes.extend(obs.getRectAxis())
            dist = np.array([self.robot.pos[0] - obs.pos[0],
                        self.robot.pos[1] - obs.pos[1]])
            if obs.getProjectRadius(axes[0]) + self.robot.width / 2 < \
                    fabs(dist.dot(axes[0])) or \
                obs.getProjectRadius(axes[1]) + self.robot.length / 2 < \
                    fabs(dist.dot(axes[1])) or \
                self.robot.getProjectRadius(axes[2]) + obs.width / 2 < \
                    fabs(dist.dot(axes[2])) or \
                self.robot.getProjectRadius(axes[3]) + obs.length / 2 < \
                    fabs(dist.dot(axes[3])):
                num += 1
                continue
            else:
                return True

    def _step(self, action):
        if action == 0:
            self.robot.move(2)
        elif action == 1:
            self.robot.rotate(60, deg=True)
        else:
            self.robot.rotate(-60, deg=True)
        self.update_state()
        if self.robot.pos[0] >= self.screen_width or self.robot.pos[1] >= self.screen_height:
            print 'I hate the wall'
            done = True
            reward = -50
        elif self._checkCollision():
            print 'collision detected'
            done = True
            reward = -50
        else:
            reward = 1
            done = False
        return self.state, reward, done, {}

    def _reset(self):
        self.robot.set_pos(100, 100)
        self.robot.set_angle(0)
        for obs in self.obstacles:
            obs.set_pos(randint(0, self.screen_width), randint(0, self.screen_height))
            obs.set_angle(randint(0, 180))
        for edge in self.edges:
            edge.set_pos(randint(0, self.screen_width), randint(0, self.screen_height))
            edge.set_angle(0)
        self.update_state()
        return self.state

    def update_state(self):
        self.robot.update_sensors(self.visible_object)
        self.state[:] = self.robot.get_sensor_values()

    def register_visible_object(self, geom_container):
        self.visible_object.extend(geom_container.get_geom_list())

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            for geom in self.visible_object:
                self.viewer.add_geom(geom)
        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))