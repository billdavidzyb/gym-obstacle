import numpy as np
from numpy.random import randint
from gym import Env
from gym.envs.classic_control import rendering

class GeomContainer(rendering.Geom):
    def __init__(self, geom, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.pos = np.asarray([pos_x, pos_y], dtype=np.float32)
        assert self.pos.shape == (2,), 'Invalid pos-array shape'
        self.angle = angle
        self.trans = rendering.Transform()
        self.add_attr(self.trans)
    def render1(self):
        self.geom._color = self._color
        self.geom.attrs = self.attrs
        self.trans.set_translation(*self.pos)
        self.trans.set_rotation(self.angle)
        self.geom.render1()
    #
    def set_pos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
    def _move_by_xy(self, diff_x, diff_y):
        self.pos[:] += diff_x, diff_y
    def move(self, v):
        self._move_by_xy(v * np.cos(self.angle), v * np.sin(self.angle))
    #
    def set_angle(self, angle, deg=False):
        self.angle = angle if not deg else np.deg2rad(angle)
    def rotate(self, diff_angle, deg=False):
        self.angle += diff_angle if not deg else np.deg2rad(diff_angle)

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
        self.robot = GeomContainer(rendering.make_circle(30))
        self.robot.set_color(0, 0, 1)
        self.obstacles = []
        for i in range(3):
            obs = GeomContainer(rendering.make_circle(30))
            obs.set_color(0, 1, 0)
            self.obstacles.append(obs)
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
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            #
            self.viewer.add_geom(self.robot)
            for geom in self.obstacles:
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

