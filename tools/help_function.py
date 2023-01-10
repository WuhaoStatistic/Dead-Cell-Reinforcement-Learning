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

def distance_reward(distance):
    if distance < 6:
        return 4
    else:
        return -1


def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, next_player_x, next_boss_x, action):
    # Player dead
    if next_self_blood <= 0 and self_blood != 100:
        self_blood_reward = self_hp_reward(next_self_blood, self_blood)
        boss_blood_reward = boss_hp_reward(next_boss_blood, boss_blood)
        reward = self_blood_reward + boss_blood_reward
        done = 1
        return reward, done

    # boss dead
    elif next_boss_blood <= 0 or next_boss_x == 0:
        self_blood_reward = self_hp_reward(next_self_blood, self_blood)
        boss_blood_reward = boss_hp_reward(next_boss_blood, boss_blood)
        reward = self_blood_reward + boss_blood_reward
        done = 2
        return reward, done
    # playing
    else:
        self_blood_reward = self_hp_reward(next_self_blood, self_blood)
        boss_blood_reward = boss_hp_reward(next_boss_blood, boss_blood)

        reward = self_blood_reward + boss_blood_reward
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
