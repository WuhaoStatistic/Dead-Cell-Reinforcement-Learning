import gym
import torch
from torch.backends import cudnn

import game_environment

import models

import trainer

import buffer

DEVICE = 'cuda'
cudnn.benchmark = True


def get_model(env: gym.Env, n_frames: int):
    # print(env.observation_space.shape)
    c, *shape = env.observation_space.shape
    m = models.SimpleExtractor(shape, n_frames * c)
    m = models.DuelingMLP(m, env.action_space.n, noisy=True)
    return m.to(DEVICE)


# when you want to start training get in to the boss room and walk to trigger boss fight
# press esc into equipment surface and then come here to start training
def train(dqn):
    dqn.save_explorations(1)  # This will run a env.reset function, so make sure you are n
    dqn.load_explorations()
    # raise ValueError
    print('dqn warm up')
    dqn.learn()  # warmup
    print('dqn warm up finished')

    saved_rew = float('-inf')
    saved_train_rew = float('-inf')
    for i in range(1, 301):
        print('episode', i)
        rew, loss, lr = dqn.run_episode()
        if rew > saved_train_rew:
            print('new best train model found')
            saved_train_rew = rew
            dqn.save_models('best_train')
        if i % 10 == 0:
            eval_rew = dqn.evaluate()
            if eval_rew > saved_rew:
                print('new best eval model found')
                saved_rew = eval_rew
                dqn.save_models('best')
        dqn.save_models('latest')

        dqn.log({'reward': rew, 'loss': loss}, i)
        print(f'episode {i} finished, total step {dqn.steps}, learned {dqn.learn_steps}, epsilon {dqn.eps}',
              f'total rewards {round(rew, 3)}, loss {round(loss, 3)}, current lr {round(lr, 8)}', sep='\n')
        print()


def main():
    n_frames = 4  # This parameter is used in multi step buffer, n_frames pictures will packed together
                  # in this task rgb or gray is the same, cuz we care only about the move of boss and play.
                  # so gray is used here
    shape = (224, 224)  # the resolution of image

    env = game_environment.DCEnv(shape, w1=0.8, w2=0.78, w3=1e-4)
    m = get_model(env, n_frames)
    replay_buffer = buffer.MultistepBuffer(100000, n=10, gamma=0.98)
    # prioritized={
    #     'alpha': 0.6,
    #     'beta': 0.4,
    #     'beta_anneal': 0.6 / 300
    # })

    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.98, eps=0.,
                          eps_func=(lambda val, step: 0.),
                          target_steps=8000,
                          learn_freq=1,
                          model=m,
                          lr=9e-4,
                          criterion=torch.nn.SmoothL1Loss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          DrQ=True,
                          reset=0,  # no reset
                          no_save=False)

    print('--------------start--------------')
    train(dqn)


if __name__ == '__main__':
    DEVICE = 'cuda'
    cudnn.benchmark = True
    print('created by Wuhao')
    main()
    # torch.FloatTensor()
