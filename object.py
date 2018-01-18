#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from gym.envs.classic_control import rendering
from math import *

class GeomContainer(rendering.Geom):
    def __init__(self, geom, collider_func=None, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.vertice = self._getVertice()
        self.collider_func = collider_func
        self.pos = np.asarray([pos_x, pos_y], dtype=np.float32)
        assert self.pos.shape == (2,), 'Invalid pos-array shape'
        self.angle = angle
        self.abs_pos = np.copy(self.pos)
        self.abs_angle = self.angle
        self.trans = rendering.Transform()
        self.segments_cache = None
        self.add_attr(self.trans)
     
    def _getVertice(self):
        try:
            return self.geom.v 
        except:
            raise NonePolygonErr

    def getProjectRadius(self, ax):
        if self.vertice is not None:
            self.width = max(fabs(self.vertice[0][0] - self.vertice[1][0]),
                        fabs(self.vertice[1][0] - self.vertice[2][0]))
            self.length = max(fabs(self.vertice[1][1] - self.vertice[2][1]),
                        fabs(self.vertice[0][1] - self.vertice[1][1]))
            # print self.width, self.length
            assert ax.shape == (2,), 'Invalid axis shape'
            axis = self.getRectAxis()
            return self.width / 2 * fabs(ax.dot(axis[0])) + \
                    self.length / 2 * fabs(ax.dot(axis[1]))
        else:
            return None

    def getRectAxis(self):
        axis = []
        axis.append(np.array([cos(np.deg2rad(self.angle)), 
                            sin(np.deg2rad(self.angle))]))
        axis.append(np.array([-sin(np.deg2rad(self.angle)), 
                            cos(np.deg2rad(self.angle))]))
        # print 'axis ' + str(axis)
        return axis

    def render(self):
        self.geom._color = self._color
        self.geom.attrs = self.attrs
        self.geom.render()
    
    def set_pos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
        self.update()

    def _move_by_xy(self, diff_x, diff_y):
        self.set_pos(self.pos[0] + diff_x, self.pos[1] + diff_y)

    def move(self, v):
        self._move_by_xy(v * np.cos(self.angle), v * np.sin(self.angle))
    
    def set_angle(self, angle, deg=False):
        self.angle = angle if not deg else np.deg2rad(angle)
        self.update()

    def rotate(self, diff_angle, deg=False):
        self.set_angle(self.angle + diff_angle if not deg else np.deg2rad(diff_angle))
    
    def update(self):
        self.trans.set_translation(*self.pos)
        self.trans.set_rotation(self.angle)
        self.abs_pos[:] = 0
        self.abs_angle = 0
        prev_angle = 0
        for attr in reversed(self.attrs):
            if isinstance(attr, rendering.Transform):
                self.abs_pos += rotate(attr.translation, prev_angle)
                self.abs_angle += attr.rotation
                prev_angle = attr.rotation
        self.segments_cache = None

    def get_segments(self):
        if self.segments_cache is None:
            self.segments_cache = self.collider_func(self.abs_pos, self.abs_angle)
        return self.segments_cache

    def get_intersections(self, segment_list):
        if self.collider_func is None:
            return []
        intersections = []
        for collider_segment in self.get_segments():
            for segment in segment_list:
                intersection = collider_segment.get_intersection(segment)
                if intersection is not None:
                    intersections.append(intersection)
        return intersections

    def get_geom_list(self):
        return [self]

class Segment():
    def __init__(self, start=(0, 0), end=(0, 0)):
        self.start = np.asarray(start, dtype=np.float32)
        self.end = np.asarray(end, dtype=np.float32)

    def diff_x(self):
        return self.end[0] - self.start[0]

    def diff_y(self):
        return self.end[1] - self.start[1]

    def update_start_end(self, start, end):
        self.start[:] = start
        self.end[:] = end
        
    def get_intersection(self, segment):
        def check_intersection_ls(line, segment):
            l = line.end - line.start
            p1 = segment.start - line.start
            p2 = segment.end - line.start
            return (p1[0]*l[1] - p1[1]*l[0] > 0) ^ (p2[0]*l[1] - p2[1]*l[0] > 0) # TODO: sign==0
        def check_intersection_ss(seg1, seg2):
            return check_intersection_ls(line=seg1, segment=seg2) and check_intersection_ls(line=seg2, segment=seg1)
        s1, s2 = self, segment
        if check_intersection_ss(s1, s2):
            r = (s2.diff_y() * (s2.start[0] - s1.start[0]) - s2.diff_x() * \
                (s2.start[1] - s1.start[1])) / (s1.diff_x() * s2.diff_y() - s1.diff_y() * s2.diff_x())
            return (1 -r) * s1.start + r * s1.end
        else:
            return None

class Wall(GeomContainer):
    def __init__(self, start, end, color, **kwargs):
        vertice = [start, end, end + [1, 1], start + [1, 1]]
        GeomContainer.__init__(self, rendering.make_polyline(vertice), 
                collider_func=self.collider_func, **kwargs)
        self.set_color(*color)
        self.wall_segment = Segment(start, end)
        
    def set_pos(self, pos_x, pos_y):
        pass

    def set_angle(self, angle, deg=False):
        pass

    def collider_func(self, *args):
        return [self.wall_segment]

def rotate(pos_array, angle, deg=False):
    pos_array = np.asarray(pos_array)
    if deg:
        angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.asarray([[c, -s], [s, c]])
    return np.dot(rotation_matrix, pos_array.T).T