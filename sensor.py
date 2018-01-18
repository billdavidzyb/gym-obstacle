#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from gym.envs.classic_control import rendering
from object import GeomContainer, Segment, rotate

_DISTANCE_SENSOR_MAX_DISTANCE = 200

class Sensor(GeomContainer):
    def __init__(self, geom, **kwargs):
        GeomContainer.__init__(self, geom, **kwargs)
        self.value = None

    def get_nearest_point(self, pos_list):
        sorted_pos_list = sorted(pos_list, key=lambda pos: \
                            np.linalg.norm(pos - self.abs_pos, ord=2))
        return sorted_pos_list[0]

    def detect(self, obstacles):
        raise NotImplementedError()

    def _set_sensor_value(self, value):
        self.value = value

class DistanceSensor(Sensor):
    def __init__(self, geom, **kwargs):
        Sensor.__init__(self, geom, **kwargs)
        self.ray_geom = rendering.Line()
        self.ray_geom.set_color(1, 0.5, 0.5)
        self.effect_geom = GeomContainer(rendering.make_circle(radius=5, filled=False))
        self.effect_geom.set_color(1, 0.5, 0.5)
        self.intersection_pos = [0, 0]
        self.max_distance = _DISTANCE_SENSOR_MAX_DISTANCE
        self._ray_segment = Segment()
        self._update_ray_segment()

    def render(self):
        Sensor.render(self)
        self.ray_geom.start = self.abs_pos
        self.ray_geom.end = self.intersection_pos
        self.ray_geom.render()
        self.effect_geom.set_pos(*self.intersection_pos)
        self.effect_geom.render()

    def get_geom_list(self):
        return Sensor.get_geom_list(self) + [self.ray_geom]

    def _update_ray_segment(self):
        self._ray_segment.update_start_end(self.abs_pos, self.abs_pos + \
                rotate([self.max_distance, 0], self.abs_angle))
    
    def detect(self, obstacles):
        self._update_ray_segment()
        intersections = []
        for obs in obstacles:
            intersections += obs.get_intersections([self._ray_segment])
        if len(intersections) > 0:
            intersection_pos = self.get_nearest_point(intersections)
            distance = np.linalg.norm(self.intersection_pos - self.abs_pos, ord=2)
        else:
            intersection_pos = self._ray_segment.end
            distance = self.max_distance
        self.intersection_pos = intersection_pos
        self._set_sensor_value(distance)