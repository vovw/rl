import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
SHOW_EVERY = 1000

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

CELL_SIZE = 30
class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

def get_state(player, food, enemy):
    return (player-food) + (player-enemy)

def render_env(player, food, enemy):
    env = np.zeros((SIZE*CELL_SIZE, SIZE*CELL_SIZE, 3), dtype=np.uint8)
    
    # Draw grid lines
    env[:, ::CELL_SIZE] = [100, 100, 100]  # Vertical lines
    env[::CELL_SIZE, :] = [100, 100, 100]  # Horizontal lines
    
    # Draw food
    env[food.y*CELL_SIZE:(food.y+1)*CELL_SIZE, food.x*CELL_SIZE:(food.x+1)*CELL_SIZE] = d[FOOD_N]
    
    # Draw player
    env[player.y*CELL_SIZE:(player.y+1)*CELL_SIZE, player.x*CELL_SIZE:(player.x+1)*CELL_SIZE] = d[PLAYER_N]
    
    # Draw enemy
    env[enemy.y*CELL_SIZE:(enemy.y+1)*CELL_SIZE, enemy.x*CELL_SIZE:(enemy.x+1)*CELL_SIZE] = d[ENEMY_N]
    
    return env

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def update_policy(log_probs, rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

input_size = 4  # (player-food_x, player-food_y, player-enemy_x, player-enemy_y)
output_size = 4  # 4 possible actions
policy_net = PolicyNetwork(input_size, output_size)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

episode_rewards = []

cv2.namedWindow("Blob World", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blob World", SIZE*CELL_SIZE, SIZE*CELL_SIZE)

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}")
        if episode_rewards:
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    log_probs = []
    rewards = []

    for i in range(200):
        state = get_state(player, food, enemy)
        
        action, log_prob = select_action(state)
        log_probs.append(log_prob)
        
        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        rewards.append(reward)

        if show:
            env = render_env(player, food, enemy)
            cv2.imshow("Blob World", env)
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    update_policy(log_probs, rewards)
    episode_rewards.append(episode_reward)

    if episode % 100 == 0:
        print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:])}")

