# import gym
# import torch
# from torch.backends import cudnn
#
# import game_environment
#
# import models
# from collections import deque
# import trainer
# import numpy as np
# import buffer
# import random
# import os
#
#
# def get_model(env: gym.Env, n_frames: int):
#     # print(env.observation_space.shape)
#     c, *shape = env.observation_space.shape
#     m = models.SimpleExtractor(shape, n_frames * c)
#     m = models.DuelingMLP(m, env.action_space.n, noisy=True)
#     return m.to('cuda')
#
#
# n_frames = 4
# shape = (224, 224)
#
# replay_buffer = buffer.MultistepBuffer(100000, n=10, gamma=0.98)
# env = game_environment.DCEnv(shape, w1=0.8, w2=0.78, w3=1e-4)
# m = get_model(env, n_frames)
# dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
#                       n_frames=n_frames, gamma=0.98, eps=0.,
#                       eps_func=(lambda val, step: 0.),
#                       target_steps=8000,
#                       learn_freq=1,
#                       model=m,
#                       lr=9e-4,
#                       criterion=torch.nn.SmoothL1Loss(),
#                       batch_size=32,
#                       device='cuda',
#                       is_double=True,
#                       DrQ=True,
#                       reset=0,  # no reset
#                       no_save=False)
# # dqn.load_explorations()
# save_loc = './explorations/0.npz'
# arrs = np.load(save_loc)
# # print(arrs['o'].shape) (209,1,256,256)
# obs_lst = arrs['o']
# action_lst = arrs['a']
# rew_lst = arrs['r']
# done_lst = arrs['d']
# print(obs_lst[0].shape)
# replay_buffer = buffer.MultistepBuffer(100000, n=10, gamma=0.98)
# stacked_obs = deque(  # elements in deque : n_frame*((c,h,w))
#     (obs_lst[0] for _ in range(4)),
#     maxlen=4)
# print(np.array(stacked_obs).shape) # (n_f,c,h,w)
# for o, a, r, d in zip(obs_lst[1:], action_lst, rew_lst, done_lst):
#     stacked_obs.append(o)  # doesnt change shape
#
#     replay_buffer.add(stacked_obs, a, r, d)
#
# b = replay_buffer.sample(24)
# print(b[0].shape)
#print(np.concatenate(b[0], axis=0).shape)
# a = np.transpose(obs_lst[20], (1, 2, 0))
# plt.imshow(a)
# plt.show()


# b = torch.randn((2, 4, 5))
# b = torch.unsqueeze(b, 0)
# a = torch.randn((1, 5, 4, 5))
# print(np.concatenate((a, b), axis=1)[:,2:,:,:].shape)

# a = [1, 2, [3, 4], [45]]
# np.array(a, copy=True, dtype=np.float32)
# import numpy as np
#
# a = np.ones((2, 2, 3))
# print(a.flatten())
