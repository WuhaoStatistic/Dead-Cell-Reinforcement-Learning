# Dead-Cell-Reinforcement-Learning

**On going project**

This is the repo for reinforcement learning. Play around with Dead Cell this game!  
**please also star [ailec0623](https://github.com/ailec0623/DQN_HollowKnight) and [seemer](https://github.com/seermer/HollowKnight_RL), part of the skeleton are from these two repos.** 

### Update Log

1/4/2023

start the project, successfully get the game data (character location health;boss location health) through Cheat Engine.

1/5/2023

add necessary tools 

1/6 - 1/9 2023

1 Try to use rgb data to train, but later found it unnecessary. Since the needed features from images are only boss and character movement, which have nothing to do with color. Meanwhile, using rgb data will cause cauculations increase three times than before.  

2 fix `reset()` function in game_environment.py (also pay attention to this function even when you apply this repo on Dead Cell, it behaves differently on different computer)

3 adding more comments within code lines for better reading experience.

4 updating resnet extractor.

5 add more pring info to inspect training.

1/10 2023

1 change the logic of `load_exploration()` and `learn()` in trainer.py

1/11 2023

fix all bugs now it can run whole forward and backward adding necessary comments for better reading.

### Important Features

1/12-1/13 2023
Very happy to have a chat with seemer. Updating saving and loading and buffer sample logic/data type. Only concate when using buffer data.
Used new reward system.

here contains some points you may need to concern. All those things are mentioned in code comments.

1 n_frame is the parameter for MultistepBuffer, it will concatnate n_frame picture together and then feed into network forward. 

2 The whole trajectory ios saved in .npz file where each step is one data in the .npz file. One step includes (obs,action,next_obs,reward,done). Obs and next_obs here
is just a single image (gray image in my case).  

3 reset() function may not work well on your local computer. The logical here is checking character x position and doing action accordingly, if computer response time
differ, the time for holding the key will also different (**especially when quiting boss room!!**). You need to test it out by yourself if my version can not work.

4 when running simple extractor & duling mlp with noisy linear and batch size 32, it only posses around half of GPU memory in my computer (RTX3090), which is roughly 14GB, you can adjust model and batch size according to this.

5 pay attention to action class, name prefix has impact on action behave (how to press key) in game_environment.py
