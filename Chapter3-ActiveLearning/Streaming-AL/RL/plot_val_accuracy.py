import re
import matplotlib.pyplot as plt
import numpy as np

def extract_rewards_from_file(filename):
    validation_rewards = []

    # Open the file for reading
    with open('valacc.txt', 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Check if the line contains "Validation Reward"
            if "Validation Reward" in line:
                # Split the line by comma and extract the reward value
                reward = float(line.split(':')[-1].strip())
                validation_rewards.append(reward)
    return validation_rewards

def plot_rewards(rewards):
    plt.plot(np.arange(0, 875, 5), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Validation Reward')
    plt.title('Evolution of Validation Reward')
    plt.grid(True)
    plt.show()

import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
print("Current Working Directory:", current_directory)


filename = 'valacc.txt'
rewards = extract_rewards_from_file(filename)
#print(rewards)
plot_rewards(rewards)