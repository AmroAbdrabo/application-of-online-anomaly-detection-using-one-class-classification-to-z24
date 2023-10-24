import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import random
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from collections import deque
from sklearn.svm import OneClassSVM
import copy
from sklearn.mixture import GaussianMixture  # Importing GaussianMixture instead of KMeans


# returns feature vectors obtained by CNN (only convolutional layers) applied to the instances of the original dataset
class CNNTransformedDataset(Dataset):
    def __init__(self, path, train_test_val, train_split):
        self.dataset = np.load(path)
        self.split = train_split
        tot_len = self.dataset.shape[0]
        val_split = (1-train_split)/2
        #self.dataset[:, -1] = (-2*self.dataset[:, -1])  + 1 # make labels -1, 1 for OCC
        np.random.seed(41)

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
        self.train(warm_start=False)
        
    def reset(self):
        # return first state and reset everything as in the constructor
        self.excluded = []
        self.inst = self.offset
        self.model = self.original_model
        self.acc = 0
        self.original_model = copy.deepcopy(self.original_model)
        #self.train(warm_start=True)
        next_inst, _ = self.dataset[self.inst]
        _, score, prob = self.train(warm_start=False)
        self.acc = 0
        next_st = np.hstack([next_inst, score, prob])
        return next_st
    
    def step(self, action):
        # return next_state, reward, done
        if action == 0:
            # if not label
            self.excluded.append(self.inst)

        self.inst = self.inst + 1 #  next instance
        next_inst, _ = self.dataset[self.inst]
        change_acc, score, prob = self.train(False) # larger score more likely an inlier (score and change_acc produced by next state)
        done = 0
        allowed_instances = np.setdiff1d(np.arange(self.inst), self.excluded) # instances we are "allowed" to label
        if self.budget == (len(allowed_instances) - self.offset):
            done = 1
        next_st = np.hstack([next_inst, score, prob])
        return [next_st, change_acc, done]
        
    def train(self, warm_start):
        # allowed instances
        allowed_instances = np.setdiff1d(np.arange(self.inst), self.excluded) if not(warm_start) else np.arange(self.offset)
        instances, labels = self.dataset.dataset[allowed_instances, :-1], self.dataset.dataset[allowed_instances, -1] 
        self.model.fit(instances)
        #self.model.novelty = True 
        # get hold out accuracy
        instances_val, labels_val = self.val_set.dataset[:, :-1], self.val_set.dataset[:, -1] 
        #g = self.model.predict(instances_val)
        #print(g)
        acc1 = accuracy_score(labels_val.astype(int), self.model.predict(instances_val).astype(int))
        acc2 = 1-acc1 #accuracy_score((-labels_val).astype(int)+1, self.model.predict(instances_val).astype(int))
        new_acc = max(acc1,  \
                      acc2)
        #print(new_acc)
        change_acc = new_acc - self.acc
        self.acc = new_acc

        # get the latest instance and get the score
        last_inst = allowed_instances[-1]
        
        score = self.model.score_samples(self.dataset.dataset[last_inst: last_inst + 1, :-1])
        prob = self.model.predict_proba(self.dataset.dataset[last_inst: last_inst + 1, :-1])[0]
        return change_acc, score, prob

import torch.nn.functional as F
class DQNNetwork(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.dropout = nn.Dropout(p=0.85)
        self.fc1 = nn.Linear(state_size, action_size)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
    """
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    """

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.05
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
        
        self.model.eval()
        q_values = self.model(state)
        self.model.train()
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

    #clf_train_lof = LocalOutlierFactor(n_neighbors=8, novelty=True) # for our one-class classifier
    #clf_test_lof = LocalOutlierFactor(n_neighbors=8, novelty=True) # for our one-class classifier
    clf_train_lof = GaussianMixture(n_components=2, random_state=42) # for our one-class classifier
    clf_test_lof = GaussianMixture(n_components=2, random_state=42)
    sampling_budget = 100 #  for active learning, this is the max nbr of samples we can query

    offset = 2 #  how many healthy samples we start off with
    train_split = 0.7 # 70% train, 15% test, 15% val

    dataset_train = CNNTransformedDataset(path=lumo_feat_path, train_test_val=0, train_split=train_split)
    dataset_test = CNNTransformedDataset(path=lumo_feat_path, train_test_val=1, train_split=train_split)
    dataset_val = CNNTransformedDataset(path=lumo_feat_path, train_test_val=2, train_split=train_split)

    # two environments one for training, the other validation
    env_train = Environment(model = clf_train_lof, dataset = dataset_train, train_env=0, offset= offset, budget= sampling_budget, val_set = dataset_val)
    env_test = Environment(model = clf_test_lof, dataset = dataset_test, train_env=1, offset= offset, budget= sampling_budget, val_set = dataset_val)
    
    state_size = 515  # Assume a state size of 512 (Resnet18 output + score fromGMM + probabilistic prediction  from GMM)
    action_size = 2  # Assume an action size of 2 for simplicity
    
    agent = DQN(state_size, action_size)
    batch_size = 32
    episodes = 1000
    validation_interval = 5  # validate every 50 episodes
    
    for e in range(episodes):
        state = env_train.reset()
        for time in range(sampling_budget+200):  # 200 is the budget for the active learning 
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
            agent.model.eval()
            # save it
            torch.save(agent.model.state_dict(), f'.\\RL_experiments\\agent_episode_{e}.pth')
            for time in range(sampling_budget+200):
                # Use the greedy policy (no exploration)
                action = np.argmax(agent.model(torch.FloatTensor(state).float().unsqueeze(0)).detach().numpy())
                next_state, reward, done = env_test.step(action)
                #print(f"Reward for time step {time} is {reward}")
                total_reward += reward
                state = next_state
                if done:
                    break
            agent.model.train() 
            print(f"Episode: {e}/{episodes}, Validation Reward: {total_reward}")

"""
Output:
Episode: 0/1000, Validation Reward: 0.8837209302325582
Episode: 5/1000, Validation Reward: 0.6871035940803383
Episode: 10/1000, Validation Reward: 0.638477801268499
Episode: 15/1000, Validation Reward: 0.6046511627906976
Episode: 20/1000, Validation Reward: 0.6849894291754757
Episode: 25/1000, Validation Reward: 0.6976744186046512
Episode: 30/1000, Validation Reward: 0.6046511627906976
Episode: 35/1000, Validation Reward: 0.813953488372093
Episode: 40/1000, Validation Reward: 0.733615221987315
Episode: 45/1000, Validation Reward: 0.8837209302325582
Episode: 50/1000, Validation Reward: 0.8837209302325582
Episode: 55/1000, Validation Reward: 0.5433403805496828
Episode: 60/1000, Validation Reward: 0.8837209302325582
Episode: 65/1000, Validation Reward: 0.6046511627906976
Episode: 70/1000, Validation Reward: 0.7463002114164905
Episode: 75/1000, Validation Reward: 0.8837209302325582
Episode: 80/1000, Validation Reward: 0.6194503171247357
Episode: 85/1000, Validation Reward: 0.8837209302325582
Episode: 90/1000, Validation Reward: 0.8837209302325582
Episode: 95/1000, Validation Reward: 0.6046511627906976
Episode: 100/1000, Validation Reward: 0.6046511627906976
Episode: 105/1000, Validation Reward: 0.8921775898520085
Episode: 110/1000, Validation Reward: 0.8118393234672304
Episode: 115/1000, Validation Reward: 0.8837209302325582
Episode: 120/1000, Validation Reward: 0.8837209302325582
Episode: 125/1000, Validation Reward: 0.6004228329809725
Episode: 130/1000, Validation Reward: 0.8837209302325582
Episode: 135/1000, Validation Reward: 0.8837209302325582
Episode: 140/1000, Validation Reward: 0.8837209302325582
Episode: 145/1000, Validation Reward: 0.8837209302325582
Episode: 150/1000, Validation Reward: 0.8837209302325582
Episode: 155/1000, Validation Reward: 0.8837209302325582
Episode: 160/1000, Validation Reward: 0.9492600422832981
Episode: 165/1000, Validation Reward: 0.8837209302325582
Episode: 170/1000, Validation Reward: 0.9873150105708245
Episode: 175/1000, Validation Reward: 1.0
Episode: 180/1000, Validation Reward: 0.8837209302325582
Episode: 185/1000, Validation Reward: 0.6046511627906976
Episode: 190/1000, Validation Reward: 0.6046511627906976
Episode: 195/1000, Validation Reward: 0.8583509513742071
Episode: 200/1000, Validation Reward: 0.8837209302325582
Episode: 205/1000, Validation Reward: 0.8837209302325582
Episode: 210/1000, Validation Reward: 0.8837209302325582
Episode: 215/1000, Validation Reward: 0.6046511627906976
Episode: 220/1000, Validation Reward: 0.8837209302325582
Episode: 225/1000, Validation Reward: 0.8837209302325582
Episode: 230/1000, Validation Reward: 0.6046511627906976
Episode: 235/1000, Validation Reward: 0.6257928118393234
Episode: 240/1000, Validation Reward: 0.6955602536997886
Episode: 245/1000, Validation Reward: 0.5983086680761099
Episode: 250/1000, Validation Reward: 0.7293868921775899
Episode: 255/1000, Validation Reward: 0.7716701902748414
Episode: 260/1000, Validation Reward: 1.0
Episode: 265/1000, Validation Reward: 0.6046511627906976
Episode: 270/1000, Validation Reward: 0.5539112050739958
Episode: 275/1000, Validation Reward: 0.5961945031712473
Episode: 280/1000, Validation Reward: 0.6046511627906976
Episode: 285/1000, Validation Reward: 0.6046511627906976
Episode: 290/1000, Validation Reward: 0.8837209302325582
Episode: 295/1000, Validation Reward: 0.8837209302325582
Episode: 300/1000, Validation Reward: 0.8837209302325582
Episode: 305/1000, Validation Reward: 0.8837209302325582
Episode: 310/1000, Validation Reward: 0.6046511627906976
Episode: 315/1000, Validation Reward: 0.8837209302325582
Episode: 320/1000, Validation Reward: 0.8837209302325582
Episode: 325/1000, Validation Reward: 0.8837209302325582
Episode: 330/1000, Validation Reward: 0.6046511627906976
Episode: 335/1000, Validation Reward: 0.6046511627906976
Episode: 340/1000, Validation Reward: 0.8837209302325582
Episode: 345/1000, Validation Reward: 0.8837209302325582
Episode: 350/1000, Validation Reward: 0.8837209302325582
Episode: 355/1000, Validation Reward: 0.6046511627906976
Episode: 360/1000, Validation Reward: 0.6553911205073996
Episode: 365/1000, Validation Reward: 0.53276955602537
Episode: 370/1000, Validation Reward: 0.8837209302325582
Episode: 375/1000, Validation Reward: 0.8837209302325582
Episode: 380/1000, Validation Reward: 0.8837209302325582
Episode: 385/1000, Validation Reward: 0.7674418604651163
Episode: 390/1000, Validation Reward: 0.8837209302325582
Episode: 395/1000, Validation Reward: 0.7315010570824525
Episode: 400/1000, Validation Reward: 0.8604651162790697
Episode: 405/1000, Validation Reward: 0.8837209302325582
Episode: 410/1000, Validation Reward: 0.6046511627906976
Episode: 415/1000, Validation Reward: 0.8837209302325582
Episode: 420/1000, Validation Reward: 0.8837209302325582
Episode: 425/1000, Validation Reward: 0.7315010570824525
Episode: 430/1000, Validation Reward: 0.6553911205073996
Episode: 435/1000, Validation Reward: 0.6046511627906976
Episode: 440/1000, Validation Reward: 0.8414376321353065
Episode: 445/1000, Validation Reward: 0.8837209302325582
Episode: 450/1000, Validation Reward: 0.6955602536997886
Episode: 455/1000, Validation Reward: 0.8054968287526427
Episode: 460/1000, Validation Reward: 0.5200845665961945
Episode: 465/1000, Validation Reward: 0.8837209302325582
Episode: 470/1000, Validation Reward: 0.6046511627906976
Episode: 475/1000, Validation Reward: 0.8837209302325582
Episode: 480/1000, Validation Reward: 0.6871035940803383
Episode: 485/1000, Validation Reward: 0.8837209302325582
Episode: 490/1000, Validation Reward: 0.8054968287526427
Episode: 495/1000, Validation Reward: 0.8837209302325582
Episode: 500/1000, Validation Reward: 0.8837209302325582
Episode: 505/1000, Validation Reward: 0.6046511627906976
Episode: 510/1000, Validation Reward: 0.8837209302325582
Episode: 515/1000, Validation Reward: 0.8837209302325582
Episode: 520/1000, Validation Reward: 0.7885835095137421
Episode: 525/1000, Validation Reward: 0.8837209302325582
Episode: 530/1000, Validation Reward: 0.8837209302325582
Episode: 535/1000, Validation Reward: 0.8837209302325582
Episode: 540/1000, Validation Reward: 0.6532769556025371
Episode: 545/1000, Validation Reward: 0.7780126849894292
Episode: 550/1000, Validation Reward: 0.8837209302325582
Episode: 555/1000, Validation Reward: 0.8837209302325582
Episode: 560/1000, Validation Reward: 0.8837209302325582
Episode: 565/1000, Validation Reward: 0.8837209302325582
Episode: 570/1000, Validation Reward: 0.8837209302325582
Episode: 575/1000, Validation Reward: 0.8837209302325582
Episode: 580/1000, Validation Reward: 0.8837209302325582
Episode: 585/1000, Validation Reward: 0.8837209302325582
Episode: 590/1000, Validation Reward: 0.8837209302325582
Episode: 595/1000, Validation Reward: 0.7293868921775899
Episode: 600/1000, Validation Reward: 0.8837209302325582
Episode: 605/1000, Validation Reward: 0.8837209302325582
Episode: 610/1000, Validation Reward: 0.8837209302325582
Episode: 615/1000, Validation Reward: 0.8837209302325582
Episode: 620/1000, Validation Reward: 0.8837209302325582
Episode: 625/1000, Validation Reward: 0.8837209302325582
Episode: 630/1000, Validation Reward: 0.8350951374207188
Episode: 635/1000, Validation Reward: 0.8837209302325582
Episode: 640/1000, Validation Reward: 0.6405919661733614
Episode: 645/1000, Validation Reward: 0.8837209302325582
Episode: 650/1000, Validation Reward: 0.8837209302325582
Episode: 655/1000, Validation Reward: 0.8837209302325582
Episode: 660/1000, Validation Reward: 0.8837209302325582
Episode: 665/1000, Validation Reward: 0.6046511627906976
Episode: 670/1000, Validation Reward: 0.8414376321353065
Episode: 675/1000, Validation Reward: 0.8837209302325582
Episode: 680/1000, Validation Reward: 0.8837209302325582
Episode: 685/1000, Validation Reward: 0.8837209302325582
Episode: 690/1000, Validation Reward: 0.8837209302325582
Episode: 695/1000, Validation Reward: 0.6046511627906976
Episode: 700/1000, Validation Reward: 0.6088794926004228
Episode: 705/1000, Validation Reward: 0.8372093023255814
Episode: 710/1000, Validation Reward: 0.53276955602537
Episode: 715/1000, Validation Reward: 0.8054968287526427
Episode: 720/1000, Validation Reward: 0.7441860465116279
Episode: 725/1000, Validation Reward: 0.8837209302325582
Episode: 730/1000, Validation Reward: 0.8837209302325582
Episode: 735/1000, Validation Reward: 0.8837209302325582
Episode: 740/1000, Validation Reward: 0.8837209302325582
Episode: 745/1000, Validation Reward: 0.6955602536997886
Episode: 750/1000, Validation Reward: 0.7928118393234672
Episode: 755/1000, Validation Reward: 0.8837209302325582
Episode: 760/1000, Validation Reward: 0.6532769556025371
Episode: 765/1000, Validation Reward: 0.8837209302325582
Episode: 770/1000, Validation Reward: 0.8837209302325582
Episode: 775/1000, Validation Reward: 0.8837209302325582
Episode: 780/1000, Validation Reward: 0.8837209302325582
Episode: 785/1000, Validation Reward: 0.7780126849894292
Episode: 790/1000, Validation Reward: 0.8837209302325582
Episode: 795/1000, Validation Reward: 0.8837209302325582
Episode: 800/1000, Validation Reward: 0.8837209302325582
Episode: 805/1000, Validation Reward: 0.8837209302325582
Episode: 810/1000, Validation Reward: 0.6046511627906976
Episode: 815/1000, Validation Reward: 0.8837209302325582
Episode: 820/1000, Validation Reward: 0.8837209302325582
Episode: 825/1000, Validation Reward: 0.7822410147991543
Episode: 830/1000, Validation Reward: 0.6046511627906976
Episode: 835/1000, Validation Reward: 0.8054968287526427
Episode: 840/1000, Validation Reward: 0.8837209302325582
Episode: 845/1000, Validation Reward: 0.773784355179704
Episode: 850/1000, Validation Reward: 0.8837209302325582
Episode: 855/1000, Validation Reward: 0.8837209302325582
Episode: 860/1000, Validation Reward: 0.6849894291754757
Episode: 865/1000, Validation Reward: 0.6046511627906976
Episode: 870/1000, Validation Reward: 0.8837209302325582
"""