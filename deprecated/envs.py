#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
import threading
from enum import Enum, unique
import random
from gym import Env
from gym.envs.classic_control import rendering
import numpy as np 

@unique # 碰撞状态
class State(Enum):
    SAFE = 0
    DANGER = 1
    CRASH = -1

class Pose: # 坐标系构建根据右手定则，逆时针为正方向
    def __init__(self, x = 0, y = 0, r = 0):    
        self.x = x  # x坐标
        self.y = y  # y坐标
        self.r = r  # theta朝向

class Point:    # 二维向量
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
    def dot(self, b):   # 点乘
        return (self.x * b.x + self.y * b.y)

class Velocity: # 二维向量
    def __init__(self, lin, rot):   # 线速度linx，角速度rotz
        self.linx = linx
        self.rotz = rotz

class Obstacle(object):
    def __init__(self, name, init_pose, init_vel, footsize, 
                    shape = 'rect'):
        self._name = name
        self._init_pose = init_pose
        self._vel = init_vel
        self._shape = shape
        self._footprint = self._getFootprint(footsize, shape)
    
    def _getFootprint(self, footsize, shape):
        assert shape == 'rect' or shape == 'circle', 'Invalid shape.'
        if shape == 'rect':
            assert type(footsize) == list
            base = Point()
            self._width = fabs(footsize[0].y - footsize[1].y)
            self._length = fabs(footsize[0].x - footsize[1].x)
            endpoint = Point(self._width / 2, self._length / 2)
            self._radius = calcDist(base, endpoint)
            self._vertex = footsize

        elif shape == 'circle':
            assert type(footsize) == float or type(footsize) == int
            self._radius = footsize
            # self._width = self._length = 2 * footsize

    def getRTVertex(self):
        assert self._shape == 'rect', 'No vertex for circles.'
        rot = self._pose.r
        mat = np.array([[cos(rot), -sin(rot), self._pose.x],
                        [sin(rot), cos(rot), self._pose.y],
                        [0, 0, 1]])
        A = []
        pos = np.array()
        for pt in self._vertex:
            vec = np.array([pt.x, pt.y])
            pos = mat.dot(vec)
            A.append(Point(pos[0],pos[1]))
        return A

    def getRectAxis(self):
        axis = list(2)
        assert self._shape == 'rect', 'No axis for circles.'
        axis[0] = Point(cos(self._pose.r), 
                            sin(self._pose.r))
        axis[1] = Point(-sin(self._pose.r), 
                            cos(self._pose.r))
        return axis
    
    def getProjectRadius(self, ax):
        axis = self.getRectAxis()
        return self._width / 2 * fabs(ax.dot(axis[0])) + \
                self._length / 2 * fabs(ax.dot(axis[1]))
    
    # 需要实时更新
    def step(self, vel):
        self._pose.r += vel.rotz * self._interval
        self._pose.x += vel.linx * self._interval * cos(self._pose.r)
        self._pose.y += vel.linx * self._interval * sin(self._pose.r)
    
    def getShape(self):
        return self._shape

    def getRadius(self):
        return self._radius

    def setInitPose(self, init_pose):
        self._pose = init_pose
        print 'Reset Agent: ' + self._name + 'pose to coordinate x: ' + \
                self._pose.x + ' y: ' + self._pose.y + ' theta: ' + \
                self._pose.r

    def getPose(self):
        return self._pose
'''
    def getWidth(self):
        return self._width

    def getLength(self):
        return self._length
'''

class Agent(Obstacle):    # 长度单位为m，角度单位为rad，时间单位为s
    def __init__(self, name, init_pose, init_vel, max_vel, footsize, 
                    shape = 'rect'):
        Obstacle.__init__(self, name, init_pose, init_vel, footsize, shape)
        self._sim_pose = init_pose
        self._max_vel = max_vel
        self._goal = Pose()
        self._interval = 0.1    # 步长
        self._arrived = True
        self._goal_tol = [0.1, 0.1]

    # 需要实时更新
    def step(self, vel):
        self._pose.r += vel.rotz * self._interval
        self._pose.x += vel.linx * self._interval * cos(self._pose.r)
        self._pose.y += vel.linx * self._interval * sin(self._pose.r)
        if calcDist(self._pose, self._goal) <= self._goal_tol[0] and \
            fabs(self._pose.r - self._goal.r) <= self._goal_tol[1]:
            self._arrived = True

    def stepSim(self, sim_steps): # sim_steps 仿真步长
        self._sim_pose.r = self._pose.r + vel.rotz * self._interval * \
                            sim_steps
        self._sim_pose.x = self._pose.x + vel.linx * self._interval * \
                            sim_steps * cos(self._pose.r)
        self._sim_pose.y = self._pose.y + vel.linx * self._interval * \
                            sim_steps * sin(self._pose.r)

    def setGoal(self, goal):
        self._arrived = False
        self._goal = goal

    def setRandomGoal(self, width, length):
        self._arrived = False
        self._goal = Pose(random.uniform(0, width),
                    random.uniform(0, length),
                    random.uniform(0, 2 * pi))

    def getPose(self, sim = 0):
        if sim == 0:
            return self._pose
        else:
            self.stepSim(sim)
            return self._sim_pose
    
    def getRectAxis(self, sim = 0):
        axis = list(2)
        if self._shape == 'circle':
            print 'No axis for circles.'
        elif self._shape == 'rect':
            if sim == 0:
                axis[0] = Point(cos(self._pose.r), 
                                    sin(self._pose.r))
                axis[1] = Point(-sin(self._pose.r), 
                                    cos(self._pose.r))
            else:
                self.stepSim(sim)
                axis[0] = Point(cos(self._sim_pose.r), 
                                    sin(self._sim_pose.r))
                axis[1] = Point(-sin(self._sim_pose.r), 
                                    cos(self._sim_pose.r))
        return axis

    def getProjectRadius(self, ax, sim = 0):
        axis = self.getRectAxis(sim)
        return self._width / 2 * fabs(ax.dot(axis[0])) + \
                self._length / 2 * fabs(ax.dot(axis[1]))

def calcDist(a, b):
    return hypot(a.x - b.x, a.y - b.y)

# return value: 
def checkCollision(a, b):
    dist = 0
    v, u, base = Point()
    if a.getShape() == b.getShape():    # 两个个体形状一致
        if a.getShape() == 'circle':  # 均为圆形
            dist = calcDist(a.getPose(), b.getPose())
            if dist <= (a.getRadius() + b.getRadius()):
                return State.CRASH
            else:   # 判断是否危险
                dist = calcDist(a.getPose(3), b.getPose(3))
                if dist <= (a.getRadius() + b.getRadius()):
                    return State.DANGER
                else:
                    return State.SAFE
        elif a.getShape() == 'rect':    # 均为矩形
            axes = []
            axes.extend(a.getRectAxis())
            axes.extend(b.getRectAxis())
            diff = Point(a.getPose().x - b.getPose().x,
                            a.getPose().y - b.getPose().y)
            for ax in axes:
                if a.getProjectRadius(ax) + b.getProjectRadius(ax) \
                    <= fabs(diff.dot(ax)):
                    return State.CRASH
            
            axes = []
            axes.extend(a.getRectAxis(3))
            axes.extend(b.getRectAxis(3))
            diff = Point(a.getPose(3).x - b.getPose(3).x,
                            a.getPose(3).y - b.getPose(3).y)
            for ax in axes:
                if a.getProjectRadius(ax, 3) + b.getProjectRadius(ax, 3) \
                    <= fabs(diff.dot(ax)):
                    return State.DANGER
            return State.SAFE

    else:   # 两个个体形状不一致
        if a.getShape() == 'circle' and b.getShape() == 'rect':
            A = b.getRTVertex()  # 必须经过转换
            r = a.getRadius()
            p = Point()
            p.x = a.getPose().x
            p.y = a.getPose().y
            if checkPointInRect(p, A) or \
                calcDistPoint2Line(p, A[0], A[1]) < r or \
                calcDistPoint2Line(p, A[1], A[2]) < r or \
                calcDistPoint2Line(p, A[2], A[3]) < r or \
                calcDistPoint2Line(p, A[3], A[0]) < r:
                return State.CRASH

            # 点到四个边的距离是否小于半径 && 圆心是否在矩形内
            # 适用于矩形不旋转的情况
            '''
            v = Point(fabs(a.getPose().x - b.getPose().x), 
                        fabs(a.getPose().y - b.getPose().y))
            u = Point(fabs(v.x - b.getLength() / 2.0),
                        fabs(v.y - b.getWidth() / 2.0))
            if (calcDist(base, u) <= a.getRadius()):
                return State.CRASH
            else:
                v = Point(fabs(a.getPose(3).x - b.getPose(3).x), 
                        fabs(a.getPose(3).y - b.getPose(3).y))
                u = Point(fabs(v.x - b.getLength() / 2.0),
                        fabs(v.y - b.getWidth() / 2.0))    
                if (calcDist(base, u) <= a.getRadius()):
                    return State.DANGER
                else:
                    return State.SAFE    
            '''
        elif b.getShape() == 'circle' and a.getShape() == 'rect':
            return checkCollision(b, a)

'''
@param a为点坐标
@param b，c分别为线段两端点
@output 点到线段距离
''' 
def calcDistPoint2Line(a, b, c):
    slope = (b.y - c.y) / (b.x - c.x)
    constant = (b.x * c.y - b.y * c.x) / (b.x - c.x)
    return abs(slope * a.x - a.y + constant) / \
            sqrt(slope ** 2 + 1)

'''
@param p为点坐标
@param A为四个端点，顺时针排列
@output 点是否在矩形内
'''
def checkPointInRect(p, A):
    width = calcDist(A[0], A[1])
    length = calcDist(A[1], A[2])
    return (calcDistPoint2Line(p, A[0], A[1]) <= length) and \
            (calcDistPoint2Line(p, A[1], A[2]) <= width) and \
            (calcDistPoint2Line(p, A[2], A[3]) <= length) and \
            (calcDistPoint2Line(p, A[3], A[0]) <= width)

def checkCircleRect(p_circle, radius_circle, p_rect, size_rect, rot_rect):
    rect_x = p_rect.x - size_rect.x / 2
    rect_y = p_rect.y - size_rect.y / 2
    # Rotate circle's center point back
    p_unrotated_circle = Point()
    p_unrotated_circle.x = cos(rot_rect) * (p_circle.x - p_rect.x) - \
                            sin(rot_rect) * (p_circle.y - p_rect.y) + \
                            p_rect.x
    p_unrotated_circle.y = sin(rot_rect) * (p_circle.x - p_rect.x) - \
                            cos(rot_rect) * (p_circle.y - p_rect.y) + \
                            p_rect.y
    
    # Closest point in the rectangle to the center of circle rotated backwards(unrotated)
    closest = Point()
    # Find the unrotated closest x point from center of unrotated circle
    if p_unrotated_circle.x < rect_x:
		closest.x = rect_x
    elif p_unrotated_circle.x > rect_x + size_rect.x:
		closest.x = rect_x + size_rect.x
    else:
		closest.x = p_unrotated_circle.x

    # Find the unrotated closest y point from center of unrotated circle
    if p_unrotated_circle.y < rect_y:
		closest.y = rect_y
    elif p_unrotated_circle.y > rect_y + size_rect.y:
		closest.y = rect_y + size_rect.y
    else:
		closest.y = p_unrotated_circle.y

    # Determine collision
    collision = False
    distance = calcDist(closest, p_unrotated_circle)
    if distance < radius_circle:
		collision = True
    else:
		collision = False
    return collision

if __name__ == '__main__':
    footsize = [Point(0, 0.2), Point(0.2, 0.2), Point(0.2, 0), Point(0, 0)]
    robot1 = Agent('1', [0, 0, 0], [0.5, 0.0], [1.0, 0.5], footsize)
    robot2 = Agent('2', [10, 10, 0], [-0.5, 0.0], [1.0, 0.5], footsize)