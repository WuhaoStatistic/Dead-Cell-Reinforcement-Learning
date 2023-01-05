import win32api
import time


def key_check():
    operations = []
    if win32api.GetAsyncKeyState(0x52):
        operations.append("R")
    if win32api.GetAsyncKeyState(0xA0):
        operations.append("L_SHIFT")
    if win32api.GetAsyncKeyState(0x4A):
        operations.append("J")
    if win32api.GetAsyncKeyState(0x4B):
        operations.append("K")

    direction = []
    if win32api.GetAsyncKeyState(0x57):
        direction.append("W")
    if win32api.GetAsyncKeyState(0x23):
        direction.append("S")
    if win32api.GetAsyncKeyState(0x41):
        direction.append("A")
    if win32api.GetAsyncKeyState(0x44):
        direction.append("D")

    return operations, direction


def is_end(self_location, boss_location):
    if self_location >= 122:
        return True
    elif boss_location == 0:
        return True
    return False


def mean_reward(d):
    return sum(d) / len(d)


def self_hp_reward(next_self_hp, current_self_hp):
    if next_self_hp - current_self_hp < 0:
        return 11 * (next_self_hp - current_self_hp)
    return 0


def boss_hp_reward(next_boss_hp, current_boss_hp):
    if next_boss_hp - current_boss_hp < 0:
        return int((current_boss_hp - next_boss_hp) / 9)
    return 0


# in the boss room
# x range is from 46 to 60
def direction_reward(move, player_x, hornet_x):
    base = 5
    if abs(player_x - hornet_x) < 2.5:
        dis = -1
    else:
        dis = 1
    if player_x - hornet_x > 0:
        s = -1
    else:
        s = 1
    if move == 0 or move == 2:
        dire = -1
    else:
        dire = 1

    return dire * s * dis * base


def distance_reward(move, next_player_x, next_hornet_x):
    if abs(next_player_x - next_hornet_x) < 2.5:
        return -6
    elif abs(next_player_x - next_hornet_x) < 4.8:
        return 4
    else:
        if move < 2:
            return 4
        else:
            return -2


def move_judge(self_blood, next_self_blood, player_x, next_player_x, hornet_x, next_hornet_x, move, hornet_skill1):
    if hornet_skill1:
        # run away while distance < 5
        if abs(player_x - hornet_x) < 6:
            # change direction while hornet use skill
            if move == 0 or move == 2:
                dire = 1
            else:
                dire = -1
            if player_x - hornet_x > 0:
                s = -1
            else:
                s = 1
            # if direction is correct and use long move
            if dire * s == 1 and move < 2:
                return 10
        # do not do long move while distance > 5
        else:
            if move >= 2:
                return 10
        return -10

    dis = abs(player_x - hornet_x)
    dire = player_x - hornet_x
    if move == 0:
        if (dis > 5 and dire > 0) or (dis < 2.5 and dire < 0):
            return 10
    elif move == 1:
        if (dis > 5 and dire < 0) or (dis < 2.5 and dire > 0):
            return 10
    elif move == 2:
        if dis > 2.5 and dis < 5 and dire > 0:
            return 10
    elif move == 3:
        if dis > 2.5 and dis < 5 and dire < 0:
            return 10

    # reward = direction_reward(move, player_x, hornet_x) + distance_reward(move, player_x, hornet_x)
    return -10


def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, next_player_x, next_boss_x, action):
    # Player dead
    if next_self_blood <= 0 and self_blood != 100:
        self_blood_reward = self_hp_reward(next_self_blood, self_blood)
        boss_blood_reward = boss_hp_reward(next_boss_blood, boss_blood)
        reward = self_blood_reward + boss_blood_reward
        if action == 4:
            reward *= 1.5
        elif action == 5:
            reward *= 0.5
        done = 1
        return reward, done

    # boss dead
    elif next_boss_blood <= 0 or next_boss_x == 0:
        self_blood_reward = self_hp_reward(next_self_blood, self_blood)
        boss_blood_reward = boss_hp_reward(next_boss_blood, boss_blood)
        reward = self_blood_reward + boss_blood_reward
        if action == 4:
            reward *= 1.5
        elif action == 5:
            reward *= 0.5
        done = 2
        return reward, done
    # playing
    else:
        self_blood_reward = self_hp_reward(next_self_blood, self_blood)
        boss_blood_reward = boss_hp_reward(next_boss_blood, boss_blood)

        reward = self_blood_reward + boss_blood_reward
        if action == 4:
            reward *= 1.5
        elif action == 5:
            reward *= 0.5
        done = 0
        return reward, done


# Paused training
def pause_game(paused):
    op, d = key_check()
    if 'ESC' in op:
        if paused:
            paused = False
            print('restart game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            op, d = key_check()
            # pauses game and can get annoying.
            if 'ESC' in op:
                if paused:
                    paused = False
                    print('restart game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return
