import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

from config import Config
from network import UnifiedNetwork

class RolloutBuffer(Dataset):
    """用于存储 PPO 交互轨迹的数据集"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.logprobs = []
        self.values = []
        self.payload_failed_masks = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx], self.actions[idx], self.rewards[idx],
            self.next_states[idx], self.dones[idx], self.logprobs[idx],
            self.values[idx], self.payload_failed_masks[idx]
        )

class PPOAgent:
    """
    定制化深度强化学习智能体 (基于文献 Section 4.2)
    """
    def __init__(self, state_dim):
        self.config = Config()
        
        # 实例化我们在上一步写的统一网络
        self.network = UnifiedNetwork(state_dim=state_dim, action_dim=2)
        # 根据文献建议，使用 Adam 优化器 (Eq. 29 之后的描述)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.LR)
        
        self.buffer = RolloutBuffer()
        
        # 自动选择运行设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def select_action(self, state, payload_failed):
        """
        根据当前状态选择动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor([payload_failed]).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor, payload_failed_mask=mask_tensor)
            
            # 从动作概率分布中采样
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        return action.item(), action_logprob.item(), state_value.item()

    def store_transition(self, state, action, reward, next_state, done, logprob, value, payload_failed):
        """记录轨迹"""
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.next_states.append(next_state)
        self.buffer.dones.append(done)
        self.buffer.logprobs.append(logprob)
        self.buffer.values.append(value)
        self.buffer.payload_failed_masks.append(payload_failed)

    def compute_gae(self, rewards, values, dones):
        """
        计算广义优势估计 GAE (公式 26, 27) 和 目标价值 (Reward-to-go)
        """
        advantages = []
        gae = 0
        
        # 为了方便计算，我们将 values 向后错一位，最后一个 next_value 为 0
        values = values + [0.0]
        
        # 逆序计算 GAE
        for i in reversed(range(len(rewards))):
            is_terminal = 1.0 - dones[i]
            # 计算时序差分误差 TD error (公式 27)
            delta = rewards[i] + self.config.GAMMA * values[i + 1] * is_terminal - values[i]
            # 计算优势函数 (公式 26)
            gae = delta + self.config.GAMMA * self.config.LAMBDA_GAE * is_terminal * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        values_tensor = torch.tensor(values[:-1], dtype=torch.float32).to(self.device)
        
        # 计算 Returns (Target values): V_target = Advantage + V_old
        returns = advantages + values_tensor
        
        # 对 advantage 进行标准化，有利于网络训练的稳定性
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 对 advantage 进行标准化，有利于网络训练的稳定性
        # 安全防护：防止因为 Episode 只有 1 步导致 std() 除以 0 产生 NaN
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # 如果只有 1 步，均值就是它自己，减去均值等于 0 (梯度为0，不更新这个特例)
            advantages = advantages - advantages.mean()
        
        return returns, advantages
    
    def update_learning_rate(self, current_step, total_steps):
        """
        余弦退火动态学习率 (带最小截断)
        Cosine Annealing Learning Rate with Minimum Threshold
        """
        initial_lr = self.config.LR  # 初始最大学习率 (eta_max)，例如 3e-4
        min_lr = 1e-5                # 最小学习率截断底线 (eta_min)
    
        # 防止步数超限
        fraction = min(current_step / total_steps, 1.0)
    
        # 余弦退火公式
        lr_now = min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + math.cos(math.pi * fraction))
    
        # 手动更新优化器中所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_now
        
        return lr_now

    def update(self):
        """
        执行网络参数更新 (Algorithm 1: Training Phase 伪代码逻辑)
        """
        # 将 buffer 数据转换为 Tensors
        old_states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.buffer.logprobs)).to(self.device)
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        payload_masks = torch.BoolTensor(np.array(self.buffer.payload_failed_masks)).to(self.device)

        # 计算 returns 和 advantages
        returns, advantages = self.compute_gae(rewards, values, dones)

        # 构建 DataLoader 进行 Mini-batch 训练
        dataset = torch.utils.data.TensorDataset(old_states, old_actions, old_logprobs, returns, advantages, payload_masks)
        dataloader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)

        # 迭代 Z 个 epochs (文献设定)
        for _ in range(self.config.TRAIN_EPOCHS):
            for batch in dataloader:
                b_states, b_actions, b_old_logprobs, b_returns, b_advantages, b_masks = batch

                # 在当前网络下重新评估动作概率分布、价值和熵
                logprobs, state_values, dist_entropy = self.network.evaluate_actions(b_states, b_actions, payload_failed_mask=b_masks)
                state_values = state_values.squeeze(-1)

                # 计算概率比 r_t(theta) (公式 25)
                ratios = torch.exp(logprobs - b_old_logprobs)

                # 计算替代目标函数的两部分 (公式 24)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.config.CLIP_EPSILON, 1.0 + self.config.CLIP_EPSILON) * b_advantages
                
                # 策略损失 (Policy Loss) -> 我们在优化器中是执行梯度下降，所以取负号转化为最小化问题
                loss_policy = -torch.min(surr1, surr2).mean()

                # 价值损失 (Value Loss) (公式 29) -> 均方误差
                loss_value = nn.MSELoss()(state_values, b_returns)

                # 熵损失 (Entropy Loss) (公式 28) -> 取负号，因为我们希望熵越大越好（增加探索）
                loss_entropy = -dist_entropy.mean()

                # 最终的统一损失函数 (合并公式 23)
                # 注：文献中 Eq 23 是 L_CLIP + c1*L_ET - c2*L_V，但由于文献说“maximize reward”，
                # 转换到 Pytorch 标准的损失最小化形式： Loss = PolicyLoss + c2 * ValueLoss + c1 * EntropyLoss
                loss = loss_policy + self.config.C2_VALUE * loss_value + self.config.C1_ENTROPY * loss_entropy

                # 反向传播与优化
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪防爆炸
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

        # 更新完毕后清空 buffer
        self.buffer.clear()