import numpy as np
from numpy.random import randint
from gym import Env
from gym.envs.classic_control import rendering

class GeomContainer(rendering.Geom):
    def __init__(self, geom, collider_func=None, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.collider_func = collider_func
        self.pos = np.asarray([pos_x, pos_y], dtype=np.float32)
        assert self.pos.shape == (2,), 'Invalid pos-array shape'
        self.angle = angle
        self.abs_pos = np.copy(self.pos)
        self.abs_angle = self.angle
        self.trans = rendering.Transform()
        self.segments_cache = None
        #
        self.add_attr(self.trans)
    def render(self):
        self.geom._color = self._color
        self.geom.attrs = self.attrs
        self.geom.render()
    #
    def set_pos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
        self.update()
    def _move_by_xy(self, diff_x, diff_y):
        self.set_pos(self.pos[0] + diff_x, self.pos[1] + diff_y)
    def move(self, v):
        self._move_by_xy(v * np.cos(self.angle), v * np.sin(self.angle))
    #
    def set_angle(self, angle, deg=False):
        self.angle = angle if not deg else np.deg2rad(angle)
        self.update()
    def rotate(self, diff_angle, deg=False):
        self.set_angle(self.angle + diff_angle if not deg else np.deg2rad(diff_angle))
    #
    def update(self):
        self.trans.set_translation(*self.pos)
        self.trans.set_rotation(self.angle)
        #
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

def get_nearest_point(pos_list, ref_pos):
    sorted_pos_list = sorted(pos_list, key=lambda pos: np.linalg.norm(pos - ref_pos, ord=2))
    return sorted_pos_list[0]

class Segment():
    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=np.float32)
        self.end = np.asarray(end, dtype=np.float32)
    def diff_x(self):
        return self.end[0] - self.start[0]
    def diff_y(self):
        return self.end[1] - self.start[1]
    def get_intersection(self, segment):
        def check_intersection_ls(line, segment):
            l = line.end - line.start
            p1 = segment.start - line.start
            p2 = segment.end - line.start
            return np.sign(np.cross(l, p1)) != np.sign(np.cross(l, p2)) # TODO: sign==0
        def check_intersection_ss(seg1, seg2):
            return check_intersection_ls(line=seg1, segment=seg2) and check_intersection_ls(line=seg2, segment=seg1)
        s1, s2 = self, segment
        if check_intersection_ss(s1, s2):
            r = (s2.diff_y() * (s2.start[0] - s1.start[0]) - s2.diff_x() * (s2.start[1] - s1.start[1])) / (s1.diff_x() * s2.diff_y() - s1.diff_y() * s2.diff_x())
            return (1 -r) * s1.start + r * s1.end
        else:
            return None

class Wall(GeomContainer):
    def __init__(self, start, end, color, **kwargs):
        GeomContainer.__init__(self, rendering.Line(start, end), collider_func=self.collider_func, **kwargs)
        self.set_color(*color)
        self.wall_segment = Segment(start, end)
    def set_pos(self, pos_x, pos_y):
        pass
    def set_angle(self, angle, deg=False):
        pass
    def collider_func(self, *args):
        return [self.wall_segment]

class Sensor(GeomContainer):
    def __init__(self, geom, **kwargs):
        GeomContainer.__init__(self, geom, **kwargs)
    def detect(self, obstacles):
        raise NotImplementedError()

class DistanceSensor(Sensor):
    def __init__(self, geom, **kwargs):
        Sensor.__init__(self, geom, **kwargs)
        self.ray_geom = rendering.Line()
        self.ray_geom.set_color(1, 0.5, 0.5)
        self.effect_geom = GeomContainer(rendering.make_circle(radius=5, filled=False))
        self.effect_geom.set_color(1, 0.5, 0.5)
        self.intersection_pos = [0, 0]
        self.distance = 0
        self.max_distance = 200
    def render(self):
        Sensor.render(self)
#        print(self.abs_pos)
        self.ray_geom.start = self.abs_pos
        self.ray_geom.end = self.intersection_pos
        self.ray_geom.render()
        self.effect_geom.set_pos(*self.intersection_pos)
        self.effect_geom.render()
    def get_geom_list(self):
        return Sensor.get_geom_list(self) + [self.ray_geom]
    def detect(self, obstacles):
        seg = Segment(self.abs_pos, self.abs_pos + rotate([self.max_distance, 0], self.abs_angle))
        intersections = []
        for obs in obstacles:
            intersections += obs.get_intersections([seg])
        if len(intersections) > 0:
            self.intersection_pos = get_nearest_point(intersections, self.abs_pos)
            self.distance = np.linalg.norm(self.intersection_pos - self.abs_pos, ord=2)
        else:
            self.intersection_pos = seg.end
            self.distance = self.max_distance

class Robot(GeomContainer):
    def __init__(self, **kwargs):
        geom = rendering.make_circle(30)
        collider_func = None
        GeomContainer.__init__(self, geom, collider_func=collider_func)
        #
        self.set_color(0, 0, 1)
        #
        self.sensors = []
        for i in range(20):
            dist_sensor = DistanceSensor(rendering.make_circle(5))
            dist_sensor.set_color(1, 0, 0)
            dist_sensor.set_pos(*(rotate([30, 0], 360 / 20 * i, deg=True)))
            dist_sensor.set_angle(360 / 20 * i, True)
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

UNIT_SQUARE = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) / 2

def rotate(pos_array, angle, deg=False):
    pos_array = np.array(pos_array)
    if deg:
        angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.dot(rotation_matrix, pos_array.T).T

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
        self.state = np.zeros(8, dtype=np.float32)
        self.viewer = None
        self.robot = Robot()
        self.obstacles = []
        for i in range(3):
            obs = GeomContainer(rendering.make_polygon(UNIT_SQUARE * 50), lambda pos, angle: polyline_to_segmentlist(rotate(UNIT_SQUARE, angle) * 50 + pos))
            obs.set_color(0, 1, 0)
            self.obstacles.append(obs)
        self.obstacles.append(Wall([0, 0], [self.screen_width, 0], (0, 1, 0)))
        self.obstacles.append(Wall([self.screen_width, 0], [self.screen_width, self.screen_height], (0, 1, 0)))
        self.obstacles.append(Wall([self.screen_width, self.screen_height], [0, self.screen_height], (0, 1, 0)))
        self.obstacles.append(Wall([0, self.screen_height], [0, 0], (0, 1, 0)))
        #
        self.visible_object = []
        self.register_visible_object(self.robot)
        for obs in self.obstacles:
            self.register_visible_object(obs)
    def _step(self, action):
        if action == 0:
            self.robot.move(2)
        elif action == 1:
            self.robot.rotate(60, deg=True)
        else:
            self.robot.rotate(-60, deg=True)
        self.robot.update_sensors(self.visible_object)
        self.update_state()
        #
        reward = 1
        done = (self.robot.pos[0] > 400 and self.robot.pos[1] > 400)
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
            if step_count != 60:
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

