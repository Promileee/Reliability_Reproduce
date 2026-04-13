import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque

from config import Config
from network import QNetwork

class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, payload_failed, next_payload_failed):
        self.buffer.append((state, action, reward, next_state, done, payload_failed, next_payload_failed))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, p_fails, next_p_fails = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), np.array(p_fails), np.array(next_p_fails))
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN 智能体
    """
    def __init__(self, state_dim):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q网络与目标网络
        self.policy_net = QNetwork(state_dim=state_dim, action_dim=2).to(self.device)
        self.target_net = QNetwork(state_dim=state_dim, action_dim=2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.LR)
        self.memory = ReplayBuffer(self.config.MEMORY_CAPACITY)
        self.loss_fn = nn.MSELoss()
        
        self.steps_done = 0

    def select_action(self, state, payload_failed):
        """epsilon-greedy 动作选择"""
        # 计算探索率 epsilon
        eps_threshold = self.config.EPSILON_END + (self.config.EPSILON_START - self.config.EPSILON_END) * \
                        np.exp(-1. * self.steps_done / self.config.EPSILON_DECAY)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            # 贪婪策略: 基于当前 Q 网络选择
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.BoolTensor([payload_failed]).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor, payload_failed_mask=mask_tensor)
                return q_values.argmax(dim=1).item()
        else:
            # 随机探索: 受限于 Mask
            if payload_failed:
                return 0 # DA
            else:
                return random.choice([0, 1])

    def update(self):
        """单步梯度下降更新网络参数"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return
            
        # 采样
        states, actions, rewards, next_states, dones, p_fails, next_p_fails = self.memory.sample(self.config.BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        p_fails = torch.BoolTensor(p_fails).to(self.device)
        next_p_fails = torch.BoolTensor(next_p_fails).to(self.device)
        
        # 1. 计算 Q(s_t, a_t)
        q_values = self.policy_net(states, payload_failed_mask=p_fails)
        state_action_values = q_values.gather(1, actions)
        
        # 2. 计算 Target Q(s_{t+1}, a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states, payload_failed_mask=next_p_fails)
            # 取最大 Q 值作为下一步的预估价值
            next_state_values = next_q_values.max(1)[0].unsqueeze(1)
            
        # 3. 期望的 Q 值 = r + gamma * max_a' Q(s', a')
        expected_state_action_values = rewards + (self.config.GAMMA * next_state_values * (1 - dones))
        
        # 4. 计算损失并反向传播
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
    def update_target_network(self):
        """硬更新 Target Network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())