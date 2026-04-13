import torch
import numpy as np
import os
from environment import UAVEnvironment
from network import UnifiedNetwork
from config import Config

def get_normalized_state(state, config, max_steps):
    norm_factors = np.append(config.L_THRESHOLDS, max_steps)
    return state / norm_factors

# 【修改点 1】：给 test 函数增加 run_dir 参数
def test(run_dir, episodes=300): 
    print(f"\n==================================================")
    print(f"🚀 开始自动评估刚刚训练好的 PPO 策略...")
    print(f"📂 模型目录: {run_dir}")
    print(f"==================================================")
    
    config = Config()
    env = UAVEnvironment()
    state_dim = config.K + 1 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = UnifiedNetwork(state_dim=state_dim, action_dim=2).to(device)
    
    # 【修改点 2】：使用传入的 run_dir
    model_path = os.path.join(run_dir, "ppo_uav_model.pth")
    try:
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
    except FileNotFoundError:
        print(f"❌ 找不到模型文件: {model_path}")
        return
    
    total_rewards = []
    
    for ep in range(episodes):
        raw_state = env.reset()
        state = get_normalized_state(raw_state, config, env.max_steps)
        episode_reward = 0.0
        done = False
        
        while not done:
            payload_failed = env.is_mission_payload_failed
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask_tensor = torch.BoolTensor([payload_failed]).to(device)
            
            with torch.no_grad():
                action_probs, _ = network(state_tensor, payload_failed_mask=mask_tensor)
                # 评估时贪婪策略
                action = torch.argmax(action_probs, dim=-1).item()
                
            raw_next_state, reward, done, _ = env.step(action)
            state = get_normalized_state(raw_next_state, config, env.max_steps)
            episode_reward += reward
            
        total_rewards.append(episode_reward)

    print(f"✅ 评估完成！(测试 {episodes} 个回合)")
    print(f"🏆 平均累积奖励: {np.mean(total_rewards):.2f}")
    
if __name__ == "__main__":
    # 如果你想单独运行 test.py，可以在这里临时指定目录
    test(run_dir="result/20260413_102545")