import chainer
import chainerrl
from model import Cource

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
