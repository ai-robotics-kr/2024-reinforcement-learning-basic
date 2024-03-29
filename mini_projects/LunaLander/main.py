'''
https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html
'''

import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from collections import deque, namedtuple

import configparser

# For visualization
import cv2
import skvideo.io

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64) # 8 -> 64
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size) # 64 -> 4
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

    
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, device, cfg=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            device (torch.device): set torch device type
            cfg ()
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.lr = cfg.getfloat('DQN', 'LR')
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.buffer_size = cfg.getint('DQN', 'BUFFER_SIZE')
        self.batch_size = cfg.getint('DQN', 'BATCH_SIZE')
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed, device)
        # Initialize time step (for updating every UPDATE_PERIOD steps)
        self.update_period = cfg.getint('DQN', 'UPDATE_PERIOD')
        self.t_step = 0

        # hyper params
        self.tau = cfg.getfloat('DQN', 'TAU')
        self.gamma = cfg.getfloat('DQN', 'GAMMA')
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_PERIOD time steps.
        self.t_step = (self.t_step + 1) % self.update_period
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): set torch device type
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
def dqn(env: gym.Env, agent: Agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

def train(cfg: configparser.ConfigParser):
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = Agent(state_size=8, action_size=4, seed=0, device=device, cfg=cfg)
    scores = dqn(env, agent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('DQN_train_result.png')
    plt.show()
    

def visualize_replay(mat, txt, org: tuple, fontscale:float=0.75, color:tuple=(0, 0, 0)):
    cv2.putText(mat, txt, org, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontscale, color, 1, cv2.LINE_AA)
    return mat

def draw_infos(frame, information_tab, state, action=None, reward=None, total_reward=None, action_cnt=None, done=None):
    if action_cnt is not None:
        h, w = frame.shape[:2]
        cv2.putText(frame, f"actions: {action_cnt}", (w//2-60, h-20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    
    if done is not None and done == True:
        h, w = frame.shape[:2]
        if total_reward >= 200:
            cv2.putText(frame, f"DONE!", (w//2-40, h-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Failed!", (w//2-40, h-50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    # obs
    visualize_replay(information_tab, "observations", (5, 15), color=(0, 0, 0))
    x, y = state[0], state[1]
    visualize_replay(information_tab, f"x: {x:0.4f}, y: {y:0.4f}", (5, 35), color=(255, 255, 0))
    v_x, v_y = state[2], state[3]
    visualize_replay(information_tab, f"x_vel: {v_x:0.4f}", (5, 50), color=(255, 255, 0))
    visualize_replay(information_tab, f"y_vel: {v_y:0.4f}", (5, 65), color=(255, 255, 0))
    angle, v_ang = state[4], state[5]
    visualize_replay(information_tab, f"angle(rad): {angle:0.4f}", (5, 80), color=(255, 255, 0))
    visualize_replay(information_tab, f"angle(deg): {np.rad2deg(angle):0.4f}", (5, 95), color=(255, 255, 0))
    visualize_replay(information_tab, f"a_vel(deg): {np.rad2deg(v_ang):0.4f}", (5, 110), color=(255, 255, 0))
    left_leg = int(state[6])
    right_leg = int(state[7])
    if left_leg == 0 and right_leg == 0:
        visualize_replay(information_tab, f"left: {left_leg} right: {right_leg}", (5, 130), color=(255, 0, 0))
    elif left_leg == 1 and right_leg == 1:
        visualize_replay(information_tab, f"left: {left_leg} right: {right_leg}", (5, 130), color=(0, 255, 0))
    elif left_leg == 1 and right_leg == 0:
        visualize_replay(information_tab, f"left: {left_leg}", (5, 130), color=(255, 0, 0))
        visualize_replay(information_tab, f"right: {right_leg}", (71, 130), color=(0, 255, 0))
    else:
        visualize_replay(information_tab, f"left: {left_leg}", (5, 130), color=(0, 255, 0))
        visualize_replay(information_tab, f"right: {right_leg}", (71, 130), color=(255, 0, 0))

    if action is None:
        return

    # action
    visualize_replay(information_tab, f"actions: {action}", (5, 170), color=(0, 0, 0))
    if action == 0:
        visualize_replay(information_tab, "<=", (5, 190), color=(100, 100, 100))
        visualize_replay(information_tab, "=>", (50, 190), color=(100, 100, 100))
        visualize_replay(information_tab, "||", (35, 210), color=(100, 100, 100))
        visualize_replay(information_tab, "V", (34, 230), color=(100, 100, 100))
    elif action == 1:
        visualize_replay(information_tab, "<=", (5, 190), color=(0, 255, 0))
        visualize_replay(information_tab, "=>", (50, 190), color=(100, 100, 100))
        visualize_replay(information_tab, "||", (35, 210), color=(100, 100, 100))
        visualize_replay(information_tab, "V", (34, 230), color=(100, 100, 100))
    elif action == 2:
        visualize_replay(information_tab, "<=", (5, 190), color=(100, 100, 100))
        visualize_replay(information_tab, "=>", (50, 190), color=(100, 100, 100))
        visualize_replay(information_tab, "||", (35, 210), color=(0, 255, 0))
        visualize_replay(information_tab, "V", (34, 230), color=(0, 255, 0))
    elif action == 3:
        visualize_replay(information_tab, "<=", (5, 190), color=(100, 100, 100))
        visualize_replay(information_tab, "=>", (50, 190), color=(0, 255, 0))
        visualize_replay(information_tab, "||", (35, 210), color=(100, 100, 100))
        visualize_replay(information_tab, "V", (34, 230), color=(100, 100, 100))
    else:
        raise ValueError(f"Invalid action")

    # reward
    if reward is None:
        return
    
    visualize_replay(information_tab, f"reward: {reward:.4f}", (5, 270), color=(0, 0, 0))
    visualize_replay(information_tab, f"total: {total_reward:.4f}", (5, 285), color=(0, 0, 0))



class TestNet:
    def __init__(self, env: gym.Env, seed: int, device: torch.device, cfg: configparser.ConfigParser):
        self.env = env
        self.seed = random.seed(seed)
        self.device = device
        self.qnetwork = QNetwork(state_size=8, action_size=4, seed=seed).to(device)
        ckpt = torch.load(cfg.get('DQN', 'CHECKPOINT'))
        self.qnetwork.load_state_dict(ckpt)
        self.qnetwork.eval()

    def inference(self, video_path:str):
        state = self.env.reset()
        score = 0
        done = False

        writer = skvideo.io.FFmpegWriter(
            video_path,
            outputdict={'-vcodec': 'libx264', '-crf': str(18), '-preset': 'veryslow', '-r': str(30)}
            )

        # first state
        frame = np.asarray(self.env.render('rgb_array'))
        h, w = frame.shape[:2]
        cv2.waitKey(1) # to render frame in env module

        total_reward = 0
        action_cnt = 0

        
        with torch.no_grad():
            while not done:
                information_tab = np.full((h, 200, 3), fill_value=120, dtype=np.uint8)

                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action = self.qnetwork(state)
                action = np.argmax(action.cpu().data.numpy())
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

                frame = np.asarray(self.env.render('rgb_array'))
                draw_infos(frame, information_tab, state, action, reward, total_reward, action_cnt, done)
                frame = np.concatenate([frame, information_tab], axis=1)

                writer.writeFrame(frame)
                action_cnt += 1
                

        writer.close()
            

def test(cfg: configparser.ConfigParser):
    env = gym.make('LunarLander-v2')
    env.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testnet = TestNet(env, 1, device, cfg)

    video_path = "output.mp4"
    testnet.inference(video_path)


if __name__ == "__main__":
    cfg = configparser.ConfigParser()
    cfg.read('cfg.ini')

    train(cfg)
    test(cfg)