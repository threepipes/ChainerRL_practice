import chainer
import chainerrl
from gym import spaces
import numpy as np
import math


def angle(a, b):
    if abs(a[0]-b[0]) < 1e-8 and abs(a[1]-b[1]) < 1e-8:
        return 0
    return math.acos((a[0]*b[0] + a[1]*b[1]) / (dist(a)*dist(b))) \
           * np.sign(cross(a, b))

def dist(a):
    return math.sqrt(a[0]**2 + a[1]**2)

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def same(a, b):
    return abs(a[0]-b[0]) < 1e-8 and abs(a[1]-b[1]) < 1e-8

class Cource:
    FORCE_DIST = 4
    ACTIONS = 6
    OBS_SIZE = 2
    def __init__(self):
        self.car = Car(0, 5, math.pi/2)
        self.turn = 0
        self.action_space = spaces.Discrete(self.ACTIONS)

    def reset(self):
        car_dir = np.random.rand() * math.pi*2 - math.pi
        self.car = Car(0, 5, car_dir)
        self.turn = 0
        return self.car.observe()

    def render(self):
        print('turn: %3d; car=%s' % (self.turn, str(self.car)))

    def step(self, action):
        pre_vec = self.car.get_vec()
        self.car.update(action)
        dist = self.car._dist()
        reward = 0
        if dist <= self.FORCE_DIST:
            self.car.force_move(self.FORCE_DIST - dist)
        done = self.turn >= 2000
        reward += angle(self.car.get_vec(), pre_vec)
        if reward < 0:
            reward *= 2
        info = str(self.car)
        self.turn += 1
        return self.car.observe(), reward, done, info

    def _calc_angle_diff(self, a, b):
        if same(a, b):
            return 0
        return math.acos((a[0]*b[0] + a[1]*b[1]) / (self._dist(a)*self._dist(b)))

    def _dist(self, a):
        return math.sqrt(a[0]**2 + a[1]**2)

    def get_action_space(self):
        return self.action_space.sample


class Car:
    OP_ACC = 1
    OP_LT = 4
    OP_RT = 2
    HND_GRD = 0.3
    SPEED = 2
    def __init__(self, _x, _y, _dir):
        self.x = _x
        self.y = _y
        self.dir = _dir

    def observe(self):
        return np.array([self._dist(), self._angle()], dtype=np.float32)

    def _dist(self):
        return math.sqrt(self.x**2 + self.y**2)

    def _angle(self):
        return angle((-self.x, -self.y), (math.cos(self.dir), math.sin(self.dir)))

    def get_vec(self):
        return (self.x, self.y)

    def update(self, action):
        self._op_handle(action)
        if action & self.OP_ACC:
            self._op_accel()

    def _op_accel(self):
        self.x += math.cos(self.dir)*self.SPEED
        self.y += math.sin(self.dir)*self.SPEED

    def _op_handle(self, op):
        if op & self.OP_LT:
            self.dir += self.HND_GRD
            if self.dir >= math.pi:
                self.dir -= math.pi*2
        elif op & self.OP_RT:
            self.dir -= self.HND_GRD
            if self.dir < -math.pi:
                self.dir += math.pi*2

    def force_move(self, force):
        dist = self._dist()
        self.x += self.x/dist * force
        self.y += self.y/dist * force

    def __str__(self):
        return '(%f, %f: %f)' % (self.x, self.y, self.dir)


def make_agent(env, obs_size, n_actions):
    n_hidden_channels = 50
    n_hidden_layers = 2
    q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions, n_hidden_channels, n_hidden_layers
    )

    optimizer = chainer.optimizers.Adam(1e-2)
    optimizer.setup(q_func)

    gamma = 0.95

    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.get_action_space()
    )

    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(10 ** 6)

    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_frequency=1,
        target_update_frequency=100
    )
    return agent


def train(env, agent):
    n_episodes = 50
    max_episode_len = 200
    log = []
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0
        t = 0
        while not done and t < max_episode_len:
            action = agent.act_and_train(obs, reward)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1
            log.append(
                "acc=%d handle=%d car=%s rew=%f obs=%s" %
                (action&1, action&6, info, reward, str(obs))
            )
        if i % 10 == 0:
            print(
                'episode: ', i,
                'R:', R,
                'obs:', obs,
                'statistics:', agent.get_statistics()
            )
            env.render()
        agent.stop_episode_and_train(obs, reward, done)

    print('Finished')
    return log


def play(env, agent):
    best_episode = []
    max_R = 0
    for i in range(10):
        obs = env.reset()
        done = False
        R = 0
        t = 0
        pos_tmp = []
        while not done and t < 200:
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            pos_tmp.append(env.car.get_vec())
        print('test episode:', i, 'R:', R)
        agent.stop_episode()

        if R > max_R:
            max_R = R
            best_episode = pos_tmp

    # play best episode
    import Canvas
    Canvas.draw(best_episode)
    print('Finish demo')


if __name__ == '__main__':
    env = Cource()
    env.render()

    obs_size = env.OBS_SIZE
    n_actions = env.ACTIONS
    agent = make_agent(env, obs_size, n_actions)

    save_path = 'agent/circle'
    # agent.load(save_path)

    # training
    train(env, agent)
    agent.save(save_path)

    play(env, agent)
