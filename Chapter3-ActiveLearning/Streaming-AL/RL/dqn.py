import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

# The environment here is the one class classifier
class SimpleEnvironment:
    def __init__(self, model, dataset):
        # the model to train on
        self.model = model
        # the model to save later
        self.original_model = copy.deepcopy(model) 
        # current instance
        self.inst = 0 
        # excluded instances 
        self.excluded = []
        # dataset
        self.dataset = dataset

    def reset(self):
        self.model = self.original_model
        self.original_model = copy.deepcopy(self.original_model)
    
    def step(self, action):
        if action == 0:
            # if not label
            self.excluded.append(self.inst)
        self.inst = self.inst + 1

    def train(self):
        instances, labels = self.dataset[:self.inst]
        self.model = self.model.fit(instances)
        self.model.predict(instances)
        
    

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = DQNNetwork(state_size, action_size).float()
        self.target_model = DQNNetwork(state_size, action_size).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).float().unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).float().unsqueeze(0)
            next_state = torch.FloatTensor(next_state).float().unsqueeze(0)
            target = self.model(state)
            with torch.no_grad():
                t = self.target_model(next_state)
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * torch.max(t)
            self.optimizer.zero_grad()
            outputs = self.model(state)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = SimpleEnvironment()
    state_size = 2  # Assume a state size of 2 for simplicity
    action_size = 1  # Assume an action size of 2 for simplicity
    agent = DQN(state_size, action_size)
    batch_size = 32
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        for time in range(200):  # 200 is the budget for the active learning 
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
