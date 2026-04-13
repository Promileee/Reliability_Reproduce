import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedNetwork(nn.Module):
    """
    带参数共享和动作掩码的多头统一神经网络
    基于文献 Section 4.1 和 Figure 4 构建
    """
    def __init__(self, state_dim, action_dim=2):
        super(UnifiedNetwork, self).__init__()
        
        # ==========================================
        # 1. 共享参数层 (Shared parameter layer)
        # 负责提取底层通用的状态特征 S_n = (X_n^1, X_n^2, n)
        # ==========================================
        self.shared_fc1 = nn.Linear(state_dim, 128)
        self.shared_fc2 = nn.Linear(128, 128) # 删掉一层
        
        # ==========================================
        # 2. 策略输出头 (Actor Head - Output layer)
        # 负责输出执行 DA (Abort) 和 DC (Continue) 的倾向 (Logits)
        # ==========================================
        self.actor_fc = nn.Linear(128, 32)
        self.actor_out = nn.Linear(32, action_dim)
        
        # ==========================================
        # 3. 价值输出头 (Critic Head - Output layer)
        # 负责输出当前状态的价值评估 V(S_n)
        # ==========================================
        self.critic_fc = nn.Linear(128, 32)
        self.critic_out = nn.Linear(32, 1)

    def forward(self, state, payload_failed_mask=None):
        """
        前向传播
        :param state: 状态张量 (Batch_size, state_dim)
        :param payload_failed_mask: 布尔掩码张量 (Batch_size,)，指明各个样本中的任务载荷是否已失效
        :return: action_probs (动作概率分布), state_value (状态价值)
        """
        # --- 共享层特征提取 ---
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # --- Critic 头 ---
        v_hidden = F.relu(self.critic_fc(x))
        state_value = self.critic_out(v_hidden)
        
        # --- Actor 头与 Action Masking (公式 21 & 22) ---
        a_hidden = F.relu(self.actor_fc(x))
        actor_logits = self.actor_out(a_hidden)
        
        # 动作索引定义：0 -> DA (Abort), 1 -> DC (Continue)
        if payload_failed_mask is not None:
            # 如果载荷失效 (payload_failed_mask 为 True)，则将选择 DC (索引1) 的原始分值设为极小值
            # 这样在经过 Softmax 后，选择 DC 的概率就会趋近于 0
            # 使用 -1e9 模拟公式(21)中的 -∞
            actor_logits[payload_failed_mask, 1] = -1e9 
            
        # 经过 Softmax 得到最终的动作概率分布
        action_probs = F.softmax(actor_logits, dim=-1)
        
        return action_probs, state_value

    def evaluate_actions(self, state, action, payload_failed_mask=None):
        """
        在 PPO 更新网络时，用于评估旧轨迹在当前网络下的概率、状态价值和策略熵
        """
        action_probs, state_value = self.forward(state, payload_failed_mask)
        
        # 构建分类分布
        dist = torch.distributions.Categorical(action_probs)
        
        # 计算指定动作的对数概率
        action_log_probs = dist.log_prob(action)
        
        # 计算当前策略的熵 (用于计算公式 28 的 Entropy Loss)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_value, dist_entropy