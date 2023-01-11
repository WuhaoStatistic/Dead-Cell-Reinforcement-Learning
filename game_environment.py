import gc
import gym
import cv2
import time
import enum
import random
import pyautogui
import threading
from tools.get_hp_location import Hp_getter
from tools.get_screen import grab_screen
# from mss.windows import MSS as mss
import pywintypes
import numpy as np
import win32gui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.


class Actions(enum.Enum):
    @classmethod
    def random(cls):
        return random.choice(list(cls))


class Move(Actions):
    NO_OP = 0
    HOLD_LEFT = 1
    HOLD_RIGHT = 2
    ROLL = 3


class Attack(Actions):
    NO_OP = 0
    ATTACK = 1
    SHIELD = 2


class Jump(Actions):
    NO_OP = 0
    SINGLE_JUMP = 1
    # DOUBLE_JUMP = 2


class DCEnv(gym.Env):
    """
    environment that interacts with Dead Cell game,
    implementation follows the gym custom environment API
    """
    # in the function reset d and r is used (r is not in the key map)
    KEYMAPS = {  # map each action to its corresponding key to press
        Move.HOLD_LEFT: 'a',
        Move.HOLD_RIGHT: 'd',
        Move.ROLL: 'i',
        Jump.SINGLE_JUMP: 'w',
        # Jump.DOUBLE_JUMP: ('w', 'w'),
        Attack.ATTACK: 'j',
        Attack.SHIELD: 'k'
    }
    # HP_CKPT = [64, 99, 135, 171, 207, 242, 278, 314, 352]
    ACTIONS = [Move, Attack, Jump]

    def __init__(self, obs_shape=(224, 224), w1=1., w2=1., w3=-0.0001,rgb=False):
        """
        :param obs_shape: the shape of observation returned by step and reset
        :param w1: the weight of negative reward when being hit
                (for example, w1=1. means give -1 reward when being hit)
        :param w2: the weight of positive reward when hitting boss
                (for example, w2=1. means give +1 reward when hitting boss)
        :param w3: the weight of positive reward when not hitting nor being hit
                (for example, w3=-0.0001 means give -0.0001 reward when neither happens
        """
        # self.monitor = self._find_window()

        self.rgb = rgb
        self.holding = []
        self.prev_self_hp = None
        self.prev_boss_hp = None
        self.prev_action = -1
        total_actions = np.prod([len(Act) for Act in self.ACTIONS])
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                dtype=np.uint8, shape=(1,) + obs_shape)
        self.action_space = gym.spaces.Discrete(int(total_actions))

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.cl = Hp_getter()
        self._timer = None
        self._episode_time = None
        self.total_epi = 0
        self.win_epi = 0

    # @staticmethod
    # def _find_window():
    #     """
    #     find the location of Hollow Knight window
    #     :return: return the monitor location for screenshot
    #     """
    #     window = pyautogui.getWindowsWithTitle('Dead Cells')
    #     assert len(window) == 1, f'found {len(window)} windows called Dead Cells {window}'
    #     window = window[0]
    #     try:
    #         window.activate()
    #     except Exception:
    #         window.minimize()
    #         window.maximize()
    #         window.restore()
    #     window.resizeTo(1280, 720)
    #     window.moveTo(0, 0)
    #     geo = None
    #     while geo is None:
    #         geo = pyautogui.locateOnScreen('./locator/geo.png', confidence=0.9)
    #         time.sleep(0.2)
    #     loc = {
    #         'left': geo.left - 48,
    #         'top': geo.top - 78,
    #         'width': 986,
    #         'height': 640
    #     }
    #     return loc

    def _timed_hold(self, key, seconds):
        """
        use a separate thread to hold a key for given seconds
        if the key is already holding, do nothing and return 1,
        :param key: the key to be pressed
        :param seconds: time to hold the key
        :return: 1 if already holding, 0 when success
        """

        def timer_thread():
            pyautogui.keyDown(key)
            time.sleep(seconds)
            pyautogui.keyUp(key)
            time.sleep(0.001)

        if self._timer is None or not self._timer.is_alive():
            # timer available, do timed action
            # ignore if there is already a timed action in progress
            self._timer = threading.Thread(target=timer_thread)
            self._timer.start()
            return 0
        else:
            return 1

    def _step_actions(self, actions):
        """
        release all non-timed holding keys,
        press keys corresponding to given actions
        :param actions: a list of actions
        :return: reward for doing given actions
        """
        for key in self.holding:
            pyautogui.keyUp(key)
        self.holding = []
        action_rew = 0
        for act in actions:
            if not act.value:
                continue
            key = self.KEYMAPS[act]

            if act.name.startswith('HOLD'):
                pyautogui.keyDown(key)
                self.holding.append(key)
            elif act.name.startswith('TIMED'):
                action_rew += self._timed_hold(key, act.value * 0.3)
            elif isinstance(key, tuple):
                with pyautogui.hold(key[0]):
                    pyautogui.press(key[1])
            else:
                pyautogui.press(key)
        return action_rew * -1e-4

    def _to_multi_discrete(self, num):
        """
        interpret the single number to a list of actions
        :param num: the number representing an action combination
        :return: list of action enums
        """
        num = int(num)
        chosen = []
        for Act in self.ACTIONS:
            num, mod = divmod(num, len(Act))
            chosen.append(Act(mod))
        return chosen

    # def _find_menu(self):
    #     """
    #     locate the menu badge,
    #     when the badge is found, the correct game is ready to be started
    #     :return: the location of menu badge
    #     """
    #     monitor = self.monitor
    #     monitor = (monitor['left'] + monitor['width'] // 2,
    #                monitor['top'],
    #                monitor['width'] // 2,
    #                monitor['height'] // 2)
    #     return pyautogui.locateOnScreen(f'locator/menu_badge.png',
    #                                     region=monitor,
    #                                     confidence=0.85)

    def observe(self):
        """
        get boss and self HP & location
        :param
        :return: observation (a resized screenshot), self HP, boss HP, self location and boss location
        """

        hwnd1 = win32gui.FindWindow(None, 'Dead Cells')
        obs = grab_screen(hwnd1)
        obs = cv2.resize(obs, self.observation_space.shape[1:])  #
        if not self.rgb:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)  # return shape e.g. (1,H,W)
            return obs[np.newaxis, ...], self.cl.get_self_hp(), self.cl.get_boss_hp(), \
               self.cl.get_self_location(), self.cl.get_boss_loca()
        else:
            print('game_environment line 218')
            print(obs.shape)
            raise NotImplementedError

    def step(self, actions):
        action_rew = 0
        if actions == self.prev_action:
            action_rew -= 2e-5
        self.prev_action = actions
        actions = self._to_multi_discrete(actions)
        action_rew += self._step_actions(actions)
        obs, self_hp, boss_hp, self_location, boss_location = self.observe()
        # this win lose condition is tested from game and Cheat Engine,
        # you can use tools.get_hp_location.py and Cheat Engine to check
        win = boss_hp == 1 and self_hp >= 1
        lose = self_hp == 1 and self_location == 0
        done = win or lose

        hurt = self_hp < self.prev_self_hp
        hit = boss_hp < self.prev_boss_hp
        reward = (
                - self.w1 * hurt
                + self.w2 * hit
                + action_rew
        )
        if not (hurt or hit):
            reward += self.w3
        if win:  # extra reward for winning based on conditions
            time_rew = 5. / (time.time() - self._episode_time)
            reward += self_hp / 40. + time_rew
            print('-------------------------win this episode-------------------------')
            self.win_epi += 1
            self.total_epi += 1
        elif lose:
            reward -= boss_hp / 5.
            print('-------------------------lose this episode-------------------------')
            self.total_epi += 1
        self.prev_self_hp = self_hp
        self.prev_boss_hp = boss_hp
        reward = np.clip(reward, -1.5, 1.5)
        return obs, reward, done, False, None

    # when enter the boss room, player's location is 20. Walk to 46 to trigger boss fight
    # when boss fight is finished player's location is 123 boss location is 0
    # player walks to either 128 129 130, then press R to enter boss room

    # def reset(self, seed=None, options=None):
    #     super(DCEnv, self).reset(seed=seed)
    #     print('reset')
    #     self.cleanup()
    #     while self.cl.get_self_location() < 100:  # still in boss room waiting to quit
    #         time.sleep(3)
    #     while 100 < self.cl.get_self_location() <= 127:  # when finish boss fight player will start at position 123
    #         pyautogui.keyDown('d')
    #     pyautogui.keyUp('d')
    #     time.sleep(1)
    #     while 132 < self.cl.get_self_location():  # when finish boss fight player will start at position 123
    #         pyautogui.keyDown('a')
    #     pyautogui.keyUp('a')
    #     time.sleep(1)
    #     while 128 <= self.cl.get_self_location() <= 131:  # press r to enter boss room
    #         pyautogui.press('r')
    #         time.sleep(3)
    #     print('entered boss room')
    #     if self.cl.get_self_location() > 138:
    #         print('reset error')
    #     while self.cl.get_boss_hp() == 0:  # loading boss room
    #         time.sleep(3)
    #     while self.cl.get_self_location() < 46:  # need to move from 20 to 46
    #         pyautogui.keyDown('d')
    #     pyautogui.keyUp('d')
    #     self.prev_self_hp, self.prev_boss_hp = self.cl.get_self_hp(), self.cl.get_boss_hp()
    #     self._episode_time = time.time()
    #     return self.observe()[0], None

    def reset(self, seed=None, option=None):
        super(DCEnv, self).reset(seed=seed)
        print('-------------------------reset-------------------------')
        self.cleanup()
        while self.cl.get_self_location() < 100:  # still in boss room waiting to quit
            time.sleep(5)  # longer time here more stable this function

        # in the boss-selection room:
        # within 128-131 can press r for 127 sometimes cant press r
        # block here guarantee character will step into boss room
        while self.cl.get_self_location() > 50:
            while not 127 < self.cl.get_self_location() < 132:
                while self.cl.get_self_location() >= 132:
                    pyautogui.keyDown('a')
                pyautogui.keyUp('a')
                time.sleep(1.5)
                while self.cl.get_self_location() <= 127:
                    pyautogui.keyDown('d')
                pyautogui.keyUp('d')
                time.sleep(1.5)
            # in the region where player can press r
            if 128 <= self.cl.get_self_location() <= 131:  # press r to enter boss room
                pyautogui.press('r')
                time.sleep(3)
        print('--------------------------------------------------------')
        print('entered boss room')
        print('--------------------------------------------------------')
        while self.cl.get_boss_hp() == 0:  # loading boss room
            time.sleep(3)
        while self.cl.get_self_location() < 46:  # need to move from 20 to 46
            pyautogui.keyDown('d')
        pyautogui.keyUp('d')

        # when we get the correct location to trigger boss fight
        # there is a period of time watching boss doing something (like Concierge(看守者) would yell)
        # this time period is important since when function is returned
        # system will start capture screen to get action
        # if you dont want to load unnecessary image, stay careful about time below
        time.sleep(4.5)

        self.prev_self_hp, self.prev_boss_hp = self.cl.get_self_hp(), self.cl.get_boss_hp()
        self._episode_time = time.time()
        return self.observe()[0], None

    def close(self):
        self.cleanup()

    def cleanup(self):
        """
        do any necessary cleanup on the interaction
        should only be called before or after an episode
        """
        if self._timer is not None:
            self._timer.join()
        self.holding = []
        for key in self.KEYMAPS.values():
            if isinstance(key, tuple):
                for k in key:
                    pyautogui.keyUp(k)
            else:
                pyautogui.keyUp(key)
        self.prev_self_hp = None
        self.prev_boss_hp = None
        self.prev_action = -1
        self._timer = None
        self._episode_time = None
        gc.collect()
