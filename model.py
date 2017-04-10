# -*- encoding: utf-8 -*-
import math
from gym import spaces
import numpy as np

"""
2次元フィールド上で車を動かすシミュレーションのモデル
"""

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
    """
    Carが動くフィールドを表す(2次元空間)
    Carは(0, 0)の周りを時計回りに回ると報酬がもらえる
    """
    FORCE_DIST = 4  # 中心に近いと計算が壊れるので，距離4以内に反発力が存在
    ACTIONS = 6     # 取れる行動の種類数
    OBS_SIZE = 2    # エージェントの観察値の種類数
    def __init__(self):
        self.turn = 0
        self.action_space = spaces.Discrete(self.ACTIONS)

    def reset(self):
        """
        環境の初期化をする
        """
        car_dir = np.random.rand() * math.pi*2 - math.pi
        self.car = Car(0, 5, car_dir)
        self.turn = 0
        return self.car.observe()

    def render(self):
        """
        ステップごとの描画関数
        """
        print('turn: %3d; car=%s' % (self.turn, str(self.car)))

    def step(self, action):
        """
        agentが選んだ行動が与えられるので
        環境を変更し，観察値や報酬を返す

        今回は，actionに応じて車を動かし，
        時計回りに進んだ角度に応じて報酬を決定している

        :param int action: どの行動を選んだか
        :return:
            observe: numpy array: 環境の観察値を返す
            reward : float      : 報酬
            done   : boolean    : 終了したか否か
            info   : str (自由?): (デバッグ用などの)情報
        """
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
        """
        :return: Descrete: とれる行動の種類数を返す
        """
        return self.action_space.sample


class Car:
    """
    車を表す
    2次元平面上を走ることができ，アクセル，方向転換ができる
    簡単のため，慣性はなく，スピードが0で方向転換可能
    """
    OP_ACC = 1    # アクセル操作
    OP_RT = 2     # 右ハンドル操作
    OP_LT = 4     # 左ハンドル操作
    HND_GRD = 0.3 # 方向転換の度合い
    SPEED = 2     # アクセルで進む量
    def __init__(self, _x, _y, _dir):
        """
        現在位置と向き(絶対角度)を持つ
        """
        self.x = _x
        self.y = _y
        self.dir = _dir

    def observe(self):
        """
        (0, 0)までの距離と相対角度を返す
        """
        return np.array([self._dist(), self._angle()], dtype=np.float32)

    def _dist(self):
        return math.sqrt(self.x**2 + self.y**2)

    def _angle(self):
        return angle((-self.x, -self.y), (math.cos(self.dir), math.sin(self.dir)))

    def get_vec(self):
        return (self.x, self.y)

    def update(self, action):
        """
        actionに応じて車を移動させる
        下位1ビットがアクセル
        2ビット目と3ビット目でハンドル操作
        """
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
        """
        (0, 0)からの反発力を受ける
        """
        dist = self._dist()
        self.x += self.x/dist * force
        self.y += self.y/dist * force

    def __str__(self):
        return '(%f, %f: %f)' % (self.x, self.y, self.dir)
