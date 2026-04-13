import torch
import numpy as np
import os
from datetime import datetime
from environment import UAVEnvironment
from dqn_agent import DQNAgent  # 引入 DQN Agent
from config import Config
from test import test
import matplotlib.pyplot as plt

def plot_learning_curve(file_path, run_dir, window_size=50):
    try:
        rewards = np.load(file_path)
    except FileNotFoundError:
        return

    weights = np.repeat(1.0, window_size) / window_size
    sma = np.convolve(rewards, weights, 'valid')
    x_raw = np.arange(len(rewards))
    x_sma = np.arange(window_size - 1, len(rewards))

    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(x_raw, rewards, color='lightsteelblue', alpha=0.5, label='Raw Episode Reward')
    plt.plot(x_sma, sma, color='royalblue', linewidth=2.5, label=f'Moving Average (Window={window_size})')
    plt.title('DQN Learning Curve - Reward Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(run_dir, "dqn_learning_curve.png")
    plt.savefig(output_path, dpi=150)
    plt.close('all')

def get_normalized_state(state, config, max_steps):
    norm_factors = np.append(config.L_THRESHOLDS, max_steps)
    return state / norm_factors

def train():
    print("开始训练 DQN 智能体...")
    config = Config()
    env = UAVEnvironment()
    
    state_dim = config.K + 1 
    agent = DQNAgent(state_dim=state_dim)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("result_dqn", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    reward_history = []
    global_step = 0
    episodes = 0

    while global_step < config.MAX_TRAIN_STEPS:
        raw_state = env.reset()
        state = get_normalized_state(raw_state, config, env.max_steps)
        episode_reward = 0.0
        done = False
        payload_failed = env.is_mission_payload_failed
        
        while not done:
            # 选择动作
            action = agent.select_action(state, payload_failed)
            
            # 环境交互
            raw_next_state, reward, done, info = env.step(action)
            next_state = get_normalized_state(raw_next_state, config, env.max_steps)
            next_payload_failed = info.get("payload_failed", False)
            
            # 缩放奖励存入回放池
            scaled_reward = reward / config.REWARD_SCALE
            agent.memory.push(state, action, scaled_reward, next_state, done, payload_failed, next_payload_failed)
            
            # 每一步都尝试优化模型
            agent.update()
            
            # 定期更新目标网络
            if global_step % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
            
            state = next_state
            payload_failed = next_payload_failed
            episode_reward += reward
            global_step += 1
            
            if global_step >= config.MAX_TRAIN_STEPS:
                break

        reward_history.append(episode_reward)
        episodes += 1

        # 打印日志
        if episodes % 50 == 0:
            avg_reward = np.mean(reward_history[-50:])
            eps = agent.config.EPSILON_END + (agent.config.EPSILON_START - agent.config.EPSILON_END) * \
                  np.exp(-1. * agent.steps_done / agent.config.EPSILON_DECAY)
            print(f"步骤 {global_step}/{config.MAX_TRAIN_STEPS} (Episode {episodes}) \t "
                  f"近期平均奖励: {avg_reward:.2f} \t 当前 Epsilon: {eps:.3f}")

    # 保存模型
    model_path = os.path.join(run_dir, "dqn_uav_model.pth")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"\n训练完成！模型权重已保存至 {model_path}")

    reward_path = os.path.join(run_dir, "reward_history.npy")
    np.save(reward_path, reward_history)

    plot_learning_curve(reward_path, run_dir, window_size=50)

    # 训练完自动测试
    test(run_dir=run_dir, episodes=300)

if __name__ == "__main__":
    train()