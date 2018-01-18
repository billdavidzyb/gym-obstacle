#!/usr/bin/python
# -*- coding: utf-8 -*-

from gym.envs.classic_control import rendering
from object import GeomContainer, rotate
from sensor import DistanceSensor

_NUM_DISTANCE_SENSOR = 4

class Robot(GeomContainer):
    def __init__(self, **kwargs):
        self.length = 20
        self.width = 30
        geom = rendering.make_polygon([(-self.width / 2, self.length / 2), 
                                        (self.width / 2, self.length / 2), 
                                        (self.width / 2, -self.length / 2), 
                                        (-self.width / 2, -self.length / 2)])
        collider_func = None
        GeomContainer.__init__(self, geom, collider_func=collider_func)
        self.set_color(0, 0, 1)
        
        self.sensors = []
        for i in range(_NUM_DISTANCE_SENSOR):
            dist_sensor = DistanceSensor(rendering.make_circle(5))
            dist_sensor.set_color(1, 0, 0)
            if i % 2 == 0:
                dist_sensor.set_pos(*(rotate([15, 0], 360 / _NUM_DISTANCE_SENSOR * i, deg=True)))
            else:
                dist_sensor.set_pos(*(rotate([10, 0], 360 / _NUM_DISTANCE_SENSOR * i, deg=True)))
            dist_sensor.set_angle(360 / _NUM_DISTANCE_SENSOR * i, True)
            dist_sensor.add_attr(self.trans)
            self.sensors.append(dist_sensor)

    def render(self):
        GeomContainer.render(self)
        for sensor in self.sensors:
            sensor.render()

    def get_geom_list(self):
        return GeomContainer.get_geom_list(self) + self.sensors

    def update(self):
        GeomContainer.update(self)
        for sensor in self.sensors:
            sensor.update()

    def update_sensors(self, visible_objects):
        for sensor in self.sensors:
            sensor.detect(visible_objects)
    
    def get_sensor_values(self):
        return [sensor.value for sensor in self.sensors]