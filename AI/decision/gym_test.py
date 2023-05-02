import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode = 'human')

# Define the neural network architectures for the actor and critic models
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = Actor()
        self.critic = Critic()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.fc2(x), dim=0)
        value = self.fc3(x)
        return policy, value

# Define the hyperparameters
num_episodes = 10000
gamma = 0.99
learning_rate = 0.03

# Initialize the actor and critic models and the optimizer
actor = Actor()
critic = Critic()
optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate)

for episode in range (num_episodes):
    state, _ = env.reset()
    temination = False
    total_reward = 0

    while not temination:
        # Use the actor model to choose an action
        action_probs = actor(torch.FloatTensor(state))
        # print(action_probs.detach().numpy())
        action = np.random.choice(env.action_space.n, p=action_probs.detach().numpy())

        # Take the chosen action and observe the new state and reward
        next_state, reward, temination, info, done = env.step(action)
        env.render()
        
        # Use the critic model to estimate the state value function
        state_value = critic(torch.FloatTensor(state))
        next_state_value = critic(torch.FloatTensor(next_state))
        td_error = reward + gamma * next_state_value - state_value
        
        # Update the critic model
        critic_loss = td_error.pow(2).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        
        # Update the actor model using the TD error as a baseline
        actor_loss = -torch.log(action_probs[action]) * td_error.detach()
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    print("Episode:", episode + 1, "Total Reward:", total_reward)