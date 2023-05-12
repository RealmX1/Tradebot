
# Importing libraries
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# Importing PyTorch here

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

env = gym.make('CartPole-v1')

print("There are {} actions".format(env.action_space.n))

gamma = 0.99
eps = np.finfo(np.float32).eps.item()

# You can move either left or right to balance the pole
# Lets implement the Actor critic network
class ActorCritic(nn.Module):
    def __init__(self, device, feature_num, hidden_num, action_num):
        super(ActorCritic, self).__init__()
        self.feature_num = feature_num
        self.hidden_num = hidden_num
        self.action_num = action_num
        
        self.fc1 = nn.Linear(feature_num, hidden_num) # 4 because there are 4 parameters as the observation space
        self.actor = nn.Linear(hidden_num, action_num) # 2 for the number of actions
        self.critic = nn.Linear(hidden_num, 1) # Critic is always 1
        self.saved_actions = []
        self.rewards = []
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_prob, state_values
    
    def step(self, optimizer):
        # We calculate the losses and perform backprop in this function
        R = 0
        policy_losses = []
        value_losses =[]
        returns = []
        
        for r in self.rewards[::-1]: # this reverses the list
            R = r + gamma * R # 0.99 is our gamma number
            returns.insert(0, R)
        # this is such a weird operation... why not start from front to back? The reverse doens't work at all... why?
        # the reward of an action is calculated using reward of all the states after the action is taken. and the closer the state is to the action, the larger weight it has.
        # for r in model.rewards:
        #     R = r + gamma * R
        #     returns.append(R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()
            
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))
        
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        loss.backward()
        optimizer.step()
        
        del self.rewards[:]
        del self.saved_actions[:]


    def select_action(self, state):
        # print(state)
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()
# In this function, we decide whehter we want the block to move left or right,based on what the model decided

























def train(model, optimizer, device):
    running_reward = 10
    for i_episode in count(): # We need around this much episodes
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):
            print(state.shape)
            action = model.select_action(torch.from_numpy(state).float().to(device))
            state, reward, termination, _, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if termination:
                break
        running_reward = 0.05 * ep_reward + (1-0.05) * running_reward

        model.step(optimizer)


        if i_episode % 10 == 0:
            print("Episode {}\tLast Reward: {:.2f}\tAverage reward: {:.2f}".format(
                i_episode, ep_reward, running_reward
            ))
        if running_reward > env.spec.reward_threshold:
            print("Solved, running reward is now {} and the last episode runs to {} time steps".format(
                    running_reward, t
            ))
            break
            # This means that we solved cartpole and training is complete
                  

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_num = env.observation_space.shape[0]
    hidden_num = 128
    action_num = env.action_space.n

    model = ActorCritic(device, feature_num, hidden_num, action_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train(model, optimizer, device)

    # There. we finished
    # Lets see it in action
    termination = False
    cnt = 0

    env2 = gym.make('CartPole-v1', render_mode='human')
    observation, _ = env2.reset()
    while not termination:
        cnt += 1
        env2.render()
        action = model.select_action(torch.from_numpy(observation).float().to(device))
        observation, reward, termination, _, done = env2.step(action)
        # Lets see how long it lasts until failing
    print(f"Game lasted {cnt} moves")


if __name__ == "__main__":
    main()