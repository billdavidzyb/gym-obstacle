import numpy as np
from numpy.random import randint
from gym import Env
from gym.envs.classic_control import rendering
from sympy.geometry import Point, Line, Ray, Circle, Polygon, intersection

class GeomContainer(rendering.Geom):
    def __init__(self, geom, collider_func=None, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.collider_func = collider_func
        self.collider = None
        self.pos = np.asarray([pos_x, pos_y], dtype=np.float32)
        assert self.pos.shape == (2,), 'Invalid pos-array shape'
        self.angle = angle
        self.trans = rendering.Transform()
        #
        self.add_attr(self.trans)
        self.update_collider()
    def render1(self):
        self.geom._color = self._color
        self.geom.attrs = self.attrs
        self.trans.set_translation(*self.pos)
        self.trans.set_rotation(self.angle)
        self.geom.render1()
    #
    def set_pos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
        self.update_collider()
    def _move_by_xy(self, diff_x, diff_y):
        self.set_pos(self.pos[0] + diff_x, self.pos[1] + diff_y)
    def move(self, v):
        self._move_by_xy(v * np.cos(self.angle), v * np.sin(self.angle))
    #
    def set_angle(self, angle, deg=False):
        self.angle = angle if not deg else np.deg2rad(angle)
        self.update_collider()
    def rotate(self, diff_angle, deg=False):
        self.set_angle(self.angle + diff_angle if not deg else np.deg2rad(diff_angle))
    #
    def update_collider(self):
        if self.collider_func is not None:
            self.collider = self.collider_func(self.pos, self.angle)
    def get_intersections(self, collider):
        if self.collider_func is not None:
            return intersection(self.collider_func(), collider)
        else:
            return []

def choose_nearest_point(points, reference_point):
    if len(points) == 0:
        raise RuntimeError()
    points = sorted(points, key=lambda point: point.distance(reference_point))
    return point[0]

class Sensor(GeomContainer):
    def __init__(self, geom, **kwargs):
        GeomContainer.__init__(self, geom, **kwargs)
    def detect(self, objects):
        raise NotImplementedError()

class DistanceSensor(Sensor):
    def __init__(self, geom, **kwargs):
        Sensor.__init__(self, geom, **kwargs)
        self.ray_geom = rendering.Line()
        self.intersection_pos = self.pos
        self.distance = 0
        self.max_distance = 1000
    def render1(self):
        self.ray_geom.start = self.pos
        self.ray_geom.end = self.intersection_pos
        self.ray_geom.render1()
        Sensor.render1(self)
    def detect(self, visible_objects):
        ray = Ray(self.pos, angle=self.angle)
        candidates = []
        for obj in visible_objects:
            candidates.extend(obj.get_intersections(ray))
        if len(candidates) > 0:
            point = choose_nearest_point(candidates, ray.source)
            self.intersection_pos = [point.x.evalf(), point.y.evalf()]
            self.distance = np.linalg.norm(self.pos - self.intersection_pos)
        else:
            self.intersection_pos = self.pos
            self.distance = self.max_distance
        return self.distance

class Robot(GeomContainer):
    def __init__(self, **kwargs):
        geom = rendering.make_circle(30)
        collider_func = lambda pos, angle: Circle(Point(*pos), 30)
        GeomContainer.__init__(self, geom, collider_func=collider_func)
        #
        self.set_color(0, 0, 1)
        #
        self.sensors = []
        for i in range(3):
            dist_sensor = DistanceSensor(rendering.make_circle(5))
            dist_sensor.set_color(1, 0, 0)
            dist_sensor.set_pos(*(rotate([0, 30], 120 * i, deg=True)))
            dist_sensor.add_attr(self.trans)
            self.sensors.append(dist_sensor)
    def render1(self):
        GeomContainer.render1(self)
        for sensor in self.sensors:
            sensor.render1()

UNIT_SQUARE = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) / 2

def rotate(pos_array, angle, deg=False):
    pos_array = np.array(pos_array)
    if deg:
        angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.dot(rotation_matrix, pos_array.T).T

class ObstacleEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.screen_width = 600
        self.screen_height = 400
        self.state = np.zeros(8, dtype=np.float32)
        self.viewer = None
        self.robot = Robot()
        self.obstacles = []
        self.visible_object = []
        for i in range(3):
            obs = GeomContainer(rendering.make_polygon(UNIT_SQUARE * 50), lambda pos, angle: Polygon(*rotate(UNIT_SQUARE * 50 + pos, angle)))
            obs.set_color(0, 1, 0)
            self.obstacles.append(obs)
        #
        self.register_visible_object(self.robot, *self.robot.sensors)
        self.register_visible_object(*self.obstacles)
    def _step(self, action):
        if action == 0:
            self.robot.move(3)
        elif action == 1:
            self.robot.rotate(30, deg=True)
        else:
            self.robot.rotate(-30, deg=True)
        self.update_state()
        #
        reward = 1
        done = (self.robot.pos[0] > 200 and self.robot.pos[1] > 200)
        return self.state, reward, done, {}
    def _reset(self):
        self.robot.set_pos(100, 100)
        self.robot.set_angle(0)
        for obs in self.obstacles:
            obs.set_pos(randint(0, self.screen_width), randint(0, self.screen_height))
            obs.set_angle(0)
        self.update_state()
        return self.state
    def update_state(self):
        self.state[0:2] = self.robot.pos
        self.state[2:4] = self.obstacles[0].pos
        self.state[4:6] = self.obstacles[1].pos
        self.state[6:8] = self.obstacles[2].pos
    def register_visible_object(self, *obj):
        self.visible_object.extend(obj)
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            #
            for geom in self.visible_object:
                self.viewer.add_geom(geom)
        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))


def main():
    from gym.envs.registration import register
    register(
        id='Obstacle-v0',
        entry_point=ObstacleEnv,
        max_episode_steps=200,
        reward_threshold=100.0,
        )
    import gym
    env = gym.make('Obstacle-v0')
    for episode in range(5):
        step_count = 0
        state = env.reset()
        while True:
            env.render()
            if step_count != 30:
                action = 0
            else:
                action = 1
            state, reward, done, info = env.step(action)
            step_count += 1
            if done:
                print('finished episode {}, reward={}'.format(episode, reward))
                break

if __name__ == '__main__':
    main()

