import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    动作价值网络 (Q-Network)
    """
    def __init__(self, state_dim, action_dim=2):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, state, payload_failed_mask=None):
        """
        前向传播
        :param state: 状态张量 (Batch_size, state_dim)
        :param payload_failed_mask: 布尔掩码张量 (Batch_size,)
        :return: q_values (动作的预期价值)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        
        if payload_failed_mask is not None:
            # 如果载荷失效，强制 DA(索引0)可用，DC(索引1)的Q值被设为极小值以避免被选中
            q_values[payload_failed_mask, 1] = -1e9 
            
        return q_values