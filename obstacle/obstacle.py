import numpy as np
from gym import Env
from gym.envs.classic_control import rendering

class GeomContainer(rendering.Geom):
    def __init__(self, geom, pos_x=0, pos_y=0, angle=0):
        rendering.Geom.__init__(self)
        self.geom = geom
        self.pos = np.asarray([pos_x, pos_y])
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
    def set_color(self, r, g, b):
        self.geom.set_color(r, g, b)
    def set_pos(self, pos_x, pos_y):
        self.pos[:] = pos_x, pos_y
    def move(self, diff_x, diff_y):
        self.pos[:] += diff_x, diff_y
    def set_angle(self, angle):
        self.angle = angle
    def rotate(self, diff_angle):
        self.angle += diff_angle

class ObstacleEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.state = np.zeros(2, dtype=np.float32)
        self.viewer = None
        self.robot = GeomContainer(rendering.make_circle(10))
        self.robot.set_color(.5, .5, .5)
    def _step(self, action):
        self.robot.move(1, 2)
        self.update_state()
        #
        reward = 1
        done = (self.robot.pos[0] > 200 and self.robot.pos[1] > 200)
        return self.state, reward, done, {}
    def _reset(self):
        self.robot.set_pos(100, 100)
        self.update_state()
        return self.state
    def update_state(self):
        self.state[0:2] = self.robot.pos
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            screen_width = 600
            screen_height = 400
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #
            self.viewer.add_geom(self.robot)
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
        env.reset()
        while True:
            env.render()
            observation, reward, done, info = env.step(None)
            if done:
                print('finished episode {}, reward={}'.format(episode, reward))
                break

if __name__ == '__main__':
    main()

