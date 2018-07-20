import numpy as np
from collections import deque


class ReplayMemory:
    def __init__(self, replay_memory_size, batch_size):
        self.memory = deque([], maxlen=replay_memory_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, 1.0 - done))

    def sample_memories(self):
        indices = np.random.permutation(len(self.memory))[:self.batch_size]
        batch_memories = [[], [], [], [], []]
        for i in indices:
            memory = self.memory[i]
            for category, value in zip(batch_memories, memory):
                category.append(value)
        batch_memories = [np.array(category) for category in batch_memories]
        return (batch_memories[0], batch_memories[1], batch_memories[2].reshape(-1, 1),
                batch_memories[3], batch_memories[4].reshape(-1, 1))
