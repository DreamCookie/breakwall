# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.95    # коэффициент дисконтирования
        self.epsilon = 1.0   # начальная вероятность случайного действия
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # ε
        self.learning_rate = 0.0001
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_steps = 0  # Счетчик шагов

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.policy_net(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.vstack([e[0] for e in minibatch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.vstack([e[3] for e in minibatch])).to(self.device)
        dones = torch.FloatTensor([e[4] for e in minibatch]).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = self.loss_fn(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Обновление ε для ε-жадной стратегии
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Обновление сети каждые 1000 шагов
        self.update_steps += 1
        if self.update_steps % 1000 == 0:
            self.update_target_network()
            print(f"Обновление сети на шаге {self.update_steps}")

        # Логирование потерь каждые 100 шагов
        if self.update_steps % 100 == 0:
            print(f"Шаг {self.update_steps}, Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")

    def save(self, filepath):
        """Сохранение весов модели."""
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Модель сохранена в {filepath}")

    def load(self, filepath):
        """Загрузка весов модели."""
        if os.path.isfile(filepath):
            self.policy_net.load_state_dict(torch.load(filepath))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = self.epsilon_min
            print(f"Модель загружена из {filepath}")
            return True
        else:
            print(f"Файл {filepath} не найден.")
            return False
