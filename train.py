import gym
import torch
from torch.backends import cudnn

import game_environment

import models

import trainer

import buffer
import random

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
def get_exp(dqn):
    dqn.save_explorations(1)
    print('save finish')


def train(dqn):
    # This will run a env.reset function, so make sure you are in the correct place
    dqn.load_explorations(number=1)
    # raise ValueError
    dqn.learn()  # warmup
    saved_rew = float('-inf')
    saved_train_rew = float('-inf')
    for i in range(1, 3):
        print('------------------------------episode', i,'------------------------------')
        rew, loss, lr = dqn.run_episode()
        if rew > saved_train_rew:
            print('------------new best train model found----------------')
            saved_train_rew = rew
            dqn.save_models('best_train')
        if i % 2 == 0:
            print('----------------------start eval-----------------------')
            eval_rew = dqn.evaluate()
            if eval_rew > saved_rew:
                print('new best eval model found')
                saved_rew = eval_rew
                dqn.save_models('best')
            print('--------------------finish eval------------------------')
        dqn.save_models('latest')

        dqn.log({'reward': rew, 'loss': loss}, i)
        print(f'episode {i} finished, total step {dqn.steps}, learned {dqn.learn_steps}, epsilon {dqn.eps}',
              f'total rewards {round(rew, 3)}, loss {round(loss, 3)}, current lr {round(lr, 8)}', sep='\n')


def eps_func1(eps, step):
    """
    linearly reduce eps
    """
    return 0.05 + 0.2 - max(step, 8000) / 40000


def main():

    # seperate save exploration and train 1 for save exploration 0 for training
    save = 0

    n_frames = 4  # This parameter is used in multi step buffer, n_frames pictures will packed together
    # in this task rgb or gray is the same, cuz we care only about the move of boss and play.
    # so gray is used here
    shape = (256, 256)  # the resolution of image

    # w1 hurt reward weight
    # w2 hit reward weight
    # w3 do nothing reward  weight
    env = game_environment.DCEnv(shape, w1=0.05, w2=0.005, w3=1e-5)
    m = get_model(env, n_frames)
    # prefix can be 'best' 'best_train' 'latest'
    # 'best' is best in evaluation
    # 'best_train' is best in training
    # 'latest' is the latest model saved every episode
    replay_buffer = buffer.MultistepBuffer(100000, n=10, gamma=0.98)
    # prioritized={
    #     'alpha': 0.6,
    #     'beta': 0.4,
    #     'beta_anneal': 0.6 / 300
    # })

    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.98, eps=0.4,
                          eps_func=eps_func1,
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
                          no_save=False,
                          prefix=None,
                          new_optimizer=False)

    print('--------------start--------------')

    if save ==1:
        get_exp(dqn)
    else:
        train(dqn)


if __name__ == '__main__':
    DEVICE = 'cuda'
    cudnn.benchmark = True
    print('created by Wuhao')
    main()
    # torch.FloatTensor()

# 修改load_exploration 的逻辑 动作帧下标应该永远小于等于状态帧下标

# 检查动作是否会超时 trainer 224行 看一下gap的逻辑


# for 循环内 concate 的问题
# buffer.add之前 concatenate把多帧变成array 出现在load_exploration那里
# 将所有concatenate 换成赋值操作  https://blog.csdn.net/qq_36627158/article/details/123356336
#
