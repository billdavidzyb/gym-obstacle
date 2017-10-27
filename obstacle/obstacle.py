import numpy as np
from gym import Env
from gym.envs.classic_control import rendering

class Obstacle(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.state = None
        self.viewer = None
    def _step(self, action):
        self.state[1] += 1
        reward = 1
        done = (self.state[1] >= 100)
        return self.state, reward, done, {}
    def _reset(self):
        self.state = np.array([0., 0.])
        return self.state
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
            robot_size = 10
            robot_fig = rendering.FilledPolygon([(-robot_size,-robot_size),(-robot_size,robot_size),(robot_size,robot_size),(robot_size,-robot_size)])
            robot_fig.set_color(.5, .5, .5)
            self.robot_trans = rendering.Transform()
            robot_fig.add_attr(self.robot_trans)
            self.viewer.add_geom(robot_fig)
        self.robot_trans.set_translation(*self.state)
        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))


def main():
    from gym.envs.registration import register
    register(
        id='Obstacle-v0',
        entry_point=Obstacle,
        max_episode_steps=200,
        reward_threshold=100.0,
        )
    import gym
    env = gym.make('Obstacle-v0')
    for episode in range(10):
        env.reset()
        while True:
            env.render()
            observation, reward, done, info = env.step(None)
            if done:
                print('finished episode ' + str(episode))
                break

if __name__ == '__main__':
    main()

