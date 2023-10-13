import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import random
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from collections import deque
import copy
from cnn import transform_epoch

# returns CNN (only convolutional layers) applied to the instances of the original dataset
class CNNTransformedDataset(Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        instance, label = self.original_dataset[index]
        label = -2*label + 1 # label = 0 is healthy, 1 is unhealthy but we want 1 healthy and -1 unhealthy
        return self.transform(instance), label


# The environment here is the one class classifier. Dataset here is the transformed dataset above
class TestingEnvironment:
    def __init__(self, model, dataset, offset, budget):
        # the model to train on
        self.model = model
        # the model to save later
        self.original_model = copy.deepcopy(model) 
        # current instance begins at an offset (we don't start cold)
        self.inst = offset
        # the offset for later resets
        self.offset = offset
        # excluded instances 
        self.excluded = []
        # dataset
        self.dataset = dataset
        # current accuracy
        self.acc = 0
        self.budget = budget

    def reset(self):
        # return first state
        self.excluded = []
        self.inst = self.offset
        self.model = self.original_model
        self.original_model = copy.deepcopy(self.original_model)
        next_inst, _ = self.dataset[self.inst]
        _, score = self.train()
        return (next_inst, score)
    
    def step(self, action):
        # return next_state, reward, done
        if action == 0:
            # if not label
            self.excluded.append(self.inst)

        self.inst = self.inst + 1 #  next instance
        next_inst, _ = self.dataset[self.inst]
        change_acc, score = self.train() # larger score more likely an inlier (score and change_acc produced by next state)
        done = 0
        if self.budget == self.inst:
            done = 1

        return [(next_inst, score), change_acc, done]
        

    def train(self):
        # allowed instances
        allowed_instances = np.setdiff1d(np.arange(self.inst), self.excluded)
        instances, labels = self.dataset[allowed_instances]
        self.model = self.model.fit(instances)
        new_acc = accuracy_score(labels, self.model.predict(instances)) 
        change_acc = new_acc - self.acc
        self.acc = new_acc

        # get the latest instance and get the score
        last_inst = allowed_instances[-1]
        score = self.model.score_samples(instances[last_inst: last_inst + 1, :])[0]
        return change_acc, score

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

from dataloader import ShearBuildingLoader, Z24Loader
from cnn import CustomResNet
if __name__ == "__main__":
    clf = LocalOutlierFactor(n_neighbors=16) # for our one-class classifier
    sampling_budget = 200 #  for active learning, this is the max nbr of samples we can query
    offset = 30 #  how many healthy samples we start off with

    # dataset loader for building
    z24_fs = 100
    z24_epoch_size = 16384
    dataset_train = Z24Loader(z24_epoch_size, lambda epoch: transform_epoch(epoch, z24_fs))

    # load the CNN which will give the feature vectors
    model = CustomResNet(version="50", num_classes=2).double()
    model.load_state_dict(torch.load('model_weights.pth'))

    dataset_transformed = CNNTransformedDataset(original_dataset=dataset_train, transform=model.features)
    env = TestingEnvironment(clf, dataset_transformed, offset, sampling_budget)
    state_size = 2  # Assume a state size of 2 for simplicity
    action_size = 2  # Assume an action size of 2 for simplicity
    agent = DQN(state_size, action_size)
    batch_size = 32
    episodes = 1000
    

    for e in range(episodes):
        state = env.reset()
        for time in range(sampling_budget):  # 200 is the budget for the active learning 
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
