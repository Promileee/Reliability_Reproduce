import torch
import numpy as np
import os
from datetime import datetime
from environment import UAVEnvironment
from ppo_agent import PPOAgent
from config import Config
from test import test

def train():
    print("开始训练 PPO 智能体...")
    config = Config()
    env = UAVEnvironment()
    
    state_dim = config.K + 1 
    agent = PPOAgent(state_dim=state_dim)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("result", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    reward_history = []
    
    global_step = 0
    episodes = 0
    print_interval = int(config.UPDATE_TIMESTEPS / 10)

    while global_step < config.MAX_TRAIN_STEPS:
        raw_state = env.reset()
        state = get_normalized_state(raw_state, config, env.max_steps)
        episode_reward = 0.0
        done = False
        
        while not done:
            payload_failed = env.is_mission_payload_failed
            
            # 选择动作
            action, logprob, value = agent.select_action(state, payload_failed)
            
            # 环境步进
            raw_next_state, reward, done, _ = env.step(action)
            next_state = get_normalized_state(raw_next_state, config, env.max_steps)
            
            # 存入 Buffer (极其重要：存入的是缩放后的 Reward！)
            scaled_reward = reward / config.REWARD_SCALE
            agent.store_transition(state, action, scaled_reward, next_state, done, logprob, value, payload_failed)
            
            state = next_state
            episode_reward += reward
            global_step += 1

        reward_history.append(episode_reward)
        episodes += 1

        # 【核心修正】：只在 Episode 结束时，并且 Buffer 攒够了足够的数据时才更新！
        # 这避免了截断 GAE 的跨回合计算错误
        if len(agent.buffer) >= config.UPDATE_TIMESTEPS:
            agent.update()
            
            # 更新 learning rate
            current_lr = agent.update_learning_rate(global_step, config.MAX_TRAIN_STEPS)

        # 打印日志
        if episodes % print_interval == 0:
            avg_reward = np.mean(reward_history[-print_interval:])
            lr_display = current_lr if 'current_lr' in locals() else config.LR
            print(f"步骤 {global_step}/{config.MAX_TRAIN_STEPS} (Episode {episodes}) \t "
                  f"近期平均奖励: {avg_reward:.2f} \t 当前LR: {lr_display:.2e}")
    # 保存模型
    model_path = os.path.join(run_dir, "ppo_uav_model.pth")
    torch.save(agent.network.state_dict(), model_path)
    print(f"\n训练完成！模型权重已保存至 {model_path}")

    reward_path = os.path.join(run_dir, "reward_history.npy")
    np.save(reward_path, reward_history)
    print(f"奖励历史数据已保存至 {reward_path}")

    plot_learning_curve(reward_path, run_dir, window_size=1000)

    test(run_dir=run_dir, episodes=1000)

def plot_learning_curve(file_path, run_dir, window_size=50):
    """
    绘制 PPO 训练过程的奖励曲线，包含原始数据和滑动平均线
    """
    import matplotlib.pyplot as plt
    
    try:
        # 加载训练时保存的奖励历史数据
        rewards = np.load(file_path)
    except FileNotFoundError:
        print(f"找不到文件 {file_path}，获取奖励数据失败。")
        return

    # 计算滑动平均 (Moving Average) 以平滑曲线
    weights = np.repeat(1.0, window_size) / window_size
    sma = np.convolve(rewards, weights, 'valid')
    
    # 构建 x 轴 (Iterations)
    x_raw = np.arange(len(rewards))
    x_sma = np.arange(window_size - 1, len(rewards))

    # 设置论文级别的画图风格
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制原始波动的奖励曲线
    plt.plot(x_raw, rewards, color='lightsteelblue', alpha=0.5, marker='.', linestyle='none', markersize=2, label='Raw Episode Reward')
    
    # 绘制平滑后的奖励曲线
    plt.plot(x_sma, sma, color='royalblue', linewidth=2.5, label=f'Moving Average (Window={window_size})')

    # 设置图表元素
    plt.title('PPO Learning Curve - Reward Convergence', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    
    # 调整坐标轴边距并保存
    plt.tight_layout()
    output_path = os.path.join(run_dir, "ppo_learning_curve.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n学习曲线已自动保存为 {output_path}")
    plt.close('all')  # 关闭图表以释放内存 

def get_normalized_state(state, config, max_steps):
    """状态归一化，将输入压缩到 [0, 1] 附近"""
    norm_factors = np.append(config.L_THRESHOLDS, max_steps)
    return state / norm_factors

if __name__ == "__main__":
    train()