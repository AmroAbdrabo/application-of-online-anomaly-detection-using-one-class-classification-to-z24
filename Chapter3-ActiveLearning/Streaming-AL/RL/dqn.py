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

# returns feature vectors obtained by CNN (only convolutional layers) applied to the instances of the original dataset
class CNNTransformedDataset(Dataset):
    def __init__(self, path, train_test_val, train_split):
        self.dataset = np.load(path)
        self.split = train_split
        tot_len = self.dataset.shape[0]
        val_split = (1-train_split)/2
        self.dataset[:, -1] = (-2*self.dataset[:, -1])  + 1 # make labels -1, 1 for OCC
        np.random.seed(42)

        # define splits (validation, test, train)
        train_indices = np.random.choice(tot_len, size=int(self.split*tot_len), replace=False)
        non_train_indices = np.setdiff1d(np.arange(tot_len), train_indices)
        val_indices = np.random.choice(non_train_indices, int(val_split*tot_len), replace=False) # hold out set for getting accuracy in reward
        test_indices  = np.setdiff1d(non_train_indices, val_indices)
        print(f"{len(train_indices)} train indices, {len(test_indices)} test indices, {len(val_indices)} val indices")
        if train_test_val == 0:
            self.dataset = self.dataset[train_indices]
        elif train_test_val == 1:
            self.dataset = self.dataset[test_indices]
        elif train_test_val == 2:
            self.dataset = self.dataset[val_indices]
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        instance_label = self.dataset[index]
        # the last element of each row, i.e. feature vector, is the label of that feature vector
        return instance_label[:-1], instance_label[-1]


# The environment here is the one class classifier. Dataset here is the transformed dataset above
class Environment:
    def __init__(self, model, dataset, train_env, offset, budget, val_set):
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
        # number of allowed queries
        self.budget = budget
        # is it train, 0, or validation, 1, environment?
        self.train_env = train_env
        # val set for accuracy in reward
        self.val_set = val_set
        # start with a "warm" model
        self.train(warm_start=True)
        
    def reset(self):
        # return first state and reset everything as in the constructor
        self.excluded = []
        self.inst = self.offset
        self.model = self.original_model
        self.original_model = copy.deepcopy(self.original_model)
        self.train(warm_start=True)
        next_inst, _ = self.dataset[self.inst]
        _, score = self.train(warm_start=True)
        next_st = np.append(next_inst, score)
        return next_st
    
    def step(self, action):
        # return next_state, reward, done
        if action == 0:
            # if not label
            self.excluded.append(self.inst)

        self.inst = self.inst + 1 #  next instance
        next_inst, _ = self.dataset[self.inst]
        change_acc, score = self.train(False) # larger score more likely an inlier (score and change_acc produced by next state)
        done = 0
        if self.budget == (self.inst - self.offset):
            done = 1
        next_st = np.append(next_inst, score)
        return [next_st, change_acc, done]
        
    def train(self, warm_start):
        # allowed instances
        allowed_instances = np.setdiff1d(np.arange(self.inst), self.excluded) if not(warm_start) else np.arange(self.offset)
        instances, labels = self.dataset.dataset[allowed_instances, :-1], self.dataset.dataset[allowed_instances, -1] 
        self.model.fit(instances)
        self.model.novelty = True 
        # get hold out accuracy
        instances_val, labels_val = self.val_set.dataset[:, :-1], self.val_set.dataset[:, -1] 
        #g = self.model.predict(instances_val)
        #print(g)
        acc1 = accuracy_score(labels_val.astype(int), self.model.predict(instances_val).astype(int))
        acc2 = accuracy_score((-labels_val).astype(int), self.model.predict(instances_val).astype(int))
        new_acc = max(acc1,  \
                      acc2)
        print(new_acc)
        change_acc = new_acc - self.acc
        self.acc = new_acc

        # get the latest instance and get the score
        last_inst = allowed_instances[-1]
        
        score = self.model.score_samples(self.dataset.dataset[last_inst: last_inst + 1, :-1])[0]
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
        self.epsilon = 0.9  # exploration rate
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

lumo_feat_path = "C:\\Users\\amroa\\Documents\\thesis\\resnet18_lumo_feat.npy"

if __name__ == "__main__":

    clf_train_lof = LocalOutlierFactor(n_neighbors=8, novelty=True) # for our one-class classifier
    clf_test_lof = LocalOutlierFactor(n_neighbors=8, novelty=True) # for our one-class classifier
    sampling_budget = 200 #  for active learning, this is the max nbr of samples we can query

    offset = 30 #  how many healthy samples we start off with
    train_split = 0.7 # 70% train, 15% test, 15% val

    dataset_train = CNNTransformedDataset(path=lumo_feat_path, train_test_val=0, train_split=train_split)
    dataset_test = CNNTransformedDataset(path=lumo_feat_path, train_test_val=1, train_split=train_split)
    dataset_val = CNNTransformedDataset(path=lumo_feat_path, train_test_val=2, train_split=train_split)

    # two environments one for training, the other validation
    env_train = Environment(model = clf_train_lof, dataset = dataset_train, train_env=0, offset= offset, budget= sampling_budget, val_set = dataset_val)
    env_test = Environment(model = clf_test_lof, dataset = dataset_test, train_env=1, offset= offset, budget= sampling_budget, val_set = dataset_val)
    
    state_size = 513  # Assume a state size of 512 (Resnet18 output + score from OCC)
    action_size = 2  # Assume an action size of 2 for simplicity
    
    agent = DQN(state_size, action_size)
    batch_size = 32
    episodes = 1000
    validation_interval = 50  # validate every 50 episodes
    

    for e in range(episodes):
        state = env_train.reset()
        for time in range(sampling_budget):  # 200 is the budget for the active learning 
            action = agent.act(state)
            next_state, reward, done = env_train.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # validation phase
        if e % validation_interval == 0:
            total_reward = 0
            state = env_test.reset()
            for time in range(sampling_budget):
                # Use the greedy policy (no exploration)
                action = np.argmax(agent.model(torch.FloatTensor(state).float().unsqueeze(0)).detach().numpy())
                next_state, reward, done = env_test.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            print(f"Episode: {e}/{episodes}, Validation Reward: {total_reward}")
        
