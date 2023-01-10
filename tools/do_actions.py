from send_key import PressKey, ReleaseKey
import time
import threading

# Hash code for key we may use: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes?redirectedfrom=MSDN
W = 0x57  # jump
S = 0x23  # down
A = 0x41  # left
D = 0x44  # right
R = 0x52  # interact

L_SHIFT = 0xA0  # roll
J = 0x4A  # main weapon attack
K = 0x4B  # second weapon attack

ESC = 0x1B  # pause


# characters will equip a short range weapon(and fast attack speed) and a shield
# no healing allowed, no left nor right skills
# since jump dash can not dizzy boss, it is skipped

# move actions
# 0
def Nothing():
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(J)
    ReleaseKey(K)
    ReleaseKey(L_SHIFT)
    pass


# Move
# 0
def Move_Left():
    PressKey(A)
    time.sleep(0.01)


# 1
def Move_Right():
    PressKey(D)
    time.sleep(0.01)


# 2
def Single_Jump():
    PressKey(W)
    time.sleep(0.02)
    ReleaseKey(W)
    Nothing()
    time.sleep(0.01)


# 3
def Double_Jump():
    PressKey(W)
    time.sleep(0.02)
    ReleaseKey(W)
    time.sleep(0.02)
    PressKey(W)
    time.sleep(0.02)
    ReleaseKey(W)
    Nothing()


# ----------------------------------------------------------------------

#
# 0
def Attack():
    PressKey(J)
    time.sleep(0.15)
    ReleaseKey(J)
    Nothing()
    time.sleep(0.01)


# 1
def Shield():
    PressKey(K)
    time.sleep(0.1)
    ReleaseKey(K)
    time.sleep(0.01)


# 2
def Roll():
    PressKey(L_SHIFT)
    time.sleep(0.01)
    ReleaseKey(L_SHIFT)
    Nothing()
    time.sleep(0.02)


# Restart function
# it restart a new game
# it is not in actions space

# when boss fight is finished
# the character will be sent to entrance and x location is 123 (this is fixed)
# we will start at x = 20 at boss room
# not until x = 46 will the boss fight start
def restart(location):
    while 120 < location <= 129:
        Move_Right()
    PressKey(R)
    time.sleep(0.02)
    ReleaseKey(R)
    time.sleep(1)
    while location <= 46:
        Move_Right()


# List for action functions
Actions = [Attack, Shield, Roll]
Directions = [Move_Left, Move_Right, Turn_Left, Turn_Right]


# Run the action
def take_action(action):
    Actions[action]()


def take_direction(direc):
    Directions[direc]()


class TackAction(threading.Thread):
    def __init__(self, threadID, name, direction, action):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.direction = direction
        self.action = action

    def run(self):
        take_direction(self.direction)
        take_action(self.action)
