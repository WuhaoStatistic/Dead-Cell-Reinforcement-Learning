import random
from collections import deque
import numpy as np


class SumTree:
    def __init__(self, maxlen: int,
                 alpha: float = 0.6, beta: float = 0.4,
                 beta_anneal: float = 0.):
        self.maxlen = maxlen
        self.leaves = [None for _ in range(maxlen)]
        self.tree = np.zeros((maxlen * 2 - 1,), dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        self.length = 0
        self.oldest = 0
        self.max_prio = 1. ** self.alpha

    def _get_idx(self, val: float):
        idx = 0
        while True:
            left_idx = 2 * idx + 1
            if left_idx >= len(self.tree):
                break  # leaf

            if val <= self.tree[left_idx]:
                # traverse left
                idx = left_idx
            else:
                # traverse right
                val -= self.tree[left_idx]
                idx = left_idx + 1
        return idx + 1 - self.maxlen

    def update_prio(self, prio: float, idx: int, return_w: bool = True):
        if prio > self.max_prio:
            self.max_prio = prio
        prio = (prio + 1e-7) ** self.alpha

        idx += self.maxlen - 1
        old_prio = self.tree[idx]
        total = self.tree[0]
        gap = prio - old_prio

        while idx >= 0:
            self.tree[idx] += gap
            idx = (idx - 1) // 2
        if return_w:
            w = (1. / (self.length * old_prio / total)) ** self.beta
            return w

    def append(self, element):
        idx = self.oldest
        self.leaves[idx] = element
        self.update_prio(self.max_prio, idx, return_w=False)
        self.oldest = (idx + 1) % self.maxlen
        if self.length < self.maxlen:
            self.length += 1
        return idx

    def sample(self, k: int):
        assert k <= self.length
        segment = self.tree[0] / k
        indices = []
        elements = []
        for i in range(k):
            val = random.uniform(i * segment, (i + 1) * segment)
            idx = self._get_idx(val)
            while self.leaves[idx] is None:
                # failsafe when the tree returns an invalid index
                # because of float point error
                val = random.uniform(i * segment, (i + 1) * segment)
                idx = self._get_idx(val)
            indices.append(idx)
            elements.append(self.leaves[idx])
        return elements, indices

    def step_beta(self):
        self.beta += self.beta_anneal

    def __len__(self):
        return self.length

    def __str__(self):
        s = [f'SumTree:{self.length}']
        prev = 0
        idx = 1
        while True:
            s.append(str(self.tree[prev:idx]))
            if idx >= len(self.tree):
                break
            prev = idx
            idx = idx * 2 + 1

        s.append('Leaves')
        s.append(str(self.leaves))
        return '\n'.join(s)


class Buffer:
    def __init__(self, size: int,
                 prioritized=None,
                 *args, **kwargs):
        assert size > 0
        self.buffer = (deque(maxlen=size) if prioritized is None
                       else SumTree(maxlen=size, **prioritized))
        self.maxlen = size
        self.prioritized = prioritized is not None
        self._temp_buffer = []

    @staticmethod
    def _to_numpy(batch):
        obs, act, rew, obs_next, done = [], [], [], [], []
        for o, a, r, o_, d in batch:
            obs.append(np.concatenate(o))
            act.append(a)
            rew.append(r)
            obs_next.append(np.concatenate(o_))
            done.append(d)

        return (np.array(obs, copy=True, dtype=np.float32),   # (batch,n_frame*channel,h,w)
                np.array(act, copy=True, dtype=np.int64)[:, np.newaxis],
                np.array(rew, copy=True, dtype=np.float32)[:, np.newaxis],
                np.array(obs_next, copy=True, dtype=np.float32),
                np.array(done, copy=True, dtype=bool)[:, np.newaxis])

    @property
    def is_full(self):
        return len(self.buffer) == self.maxlen

    def add(self, obs, act, rew, done):  # obs here is a deque (n_f,c,h,w)
        self._temp_buffer.append((obs, act, rew, done))
        if len(self._temp_buffer) == 2:
            obs_, act_, rew_, done_ = self._temp_buffer.pop(0)
            self.buffer.append((obs_, act_, rew_, obs, done_))
            if done:
                self.buffer.append((obs, act, rew, obs, done))
                self._temp_buffer = []

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return self._to_numpy(batch)

    def prioritized_sample(self, batch_size):
        batch, indices = self.buffer.sample(batch_size)
        return self._to_numpy(batch), indices

    def update_priority(self, priorities, indices):
        weights = []
        for prio, idx in zip(priorities, indices):
            weights.append(self.buffer.update_prio(prio, idx))
        # print(self.buffer)
        weights = np.array(weights, dtype=np.float32)
        return weights / np.max(weights)

    def step(self):
        if self.prioritized:
            self.buffer.step_beta()

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return '\n'.join(map(str, self.buffer))


class MultistepBuffer(Buffer):
    def __init__(self, size: int, n: int = 5, gamma: float = 0.9,
                 prioritized=None):
        super(MultistepBuffer, self).__init__(size, prioritized=prioritized)

        self.n = n
        self.gamma = gamma

    def _add_nstep(self, obs_next, done):
        record = self._temp_buffer.pop(0)
        obs, act, rew, _ = record
        for i, rec in enumerate(self._temp_buffer, 1):
            rew += (self.gamma ** i) * rec[2]
        self.buffer.append((obs, act, rew, obs_next, done))

    def add(self, obs, act, rew, done):
        self._temp_buffer.append((obs, act, rew, done))
        if len(self._temp_buffer) > self.n:
            self._add_nstep(obs, done)
        if done:
            while len(self._temp_buffer) > 0:
                self._add_nstep(obs, done)
