import numpy as np
import time
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入你的模块
from config import Config
from environment import UAVEnvironment
from ppo_agent import PPOAgent

# ==========================================
# 1. 统一的纯净 CRN 仿真函数
# ==========================================
def run_sim(env, policy_fn, num_episodes):
    rewards = []
    for ep in range(num_episodes):
        np.random.seed(10000 + ep)
        state = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            action = policy_fn(state)
            state, r, done, _ = env.step(action)
            ep_reward += r
            
        rewards.append(ep_reward)
        
    np.random.seed(None) 
    return np.mean(rewards)

# ==========================================
# 2. 启发式寻优模块 (关闭底层并行，workers=1)
# ==========================================
def optimize_heuristics(env, evolve_episodes=200):
    def obj_p1(x):
        def p1_policy(s):
            ratios = s[:Config.K] / Config.L_THRESHOLDS
            return 0 if np.any(ratios > x[0]) else 1
        return -run_sim(env, p1_policy, evolve_episodes)

    bounds_p1 = [(0.5, 1.5)]
    # 【注意】外层使用了进程池，这里的 workers 必须设为 1
    res_p1 = differential_evolution(obj_p1, bounds_p1, strategy='best1bin', 
                                    maxiter=15, popsize=10, polish=False, workers=1)

    def obj_p2(x):
        def p2_policy(s):
            return 0 if np.any(s[:Config.K] > x) else 1
        return -run_sim(env, p2_policy, evolve_episodes)

    L = Config.L_THRESHOLDS
    bounds_p2 = [(L[0]*0.5, L[0]*1.2), (L[1]*0.5, L[1]*1.2), (L[2]*0.5, L[2]*1.2)]
    res_p2 = differential_evolution(obj_p2, bounds_p2, maxiter=20, popsize=10, 
                                    polish=False, workers=1)

    return res_p1.x[0], res_p2.x

# ==========================================
# 3. PPO 内存直训模块
# ==========================================
def train_ppo_in_memory(env, max_train_steps):
    agent = PPOAgent(state_dim=Config.K + 1)
    global_step = 0
    
    while global_step < max_train_steps:
        raw_state = env.reset()
        norm_factors = np.append(Config.L_THRESHOLDS, env.max_steps)
        state = raw_state / norm_factors
        done = False
        
        while not done:
            payload_failed = env.is_mission_payload_failed
            action, logprob, value = agent.select_action(state, payload_failed)
            
            raw_next_state, reward, done, _ = env.step(action)
            next_state = raw_next_state / norm_factors
            
            scaled_reward = reward / Config.REWARD_SCALE
            agent.store_transition(state, action, scaled_reward, next_state, done, logprob, value, payload_failed)
            
            state = next_state
            global_step += 1
            
        if len(agent.buffer) >= Config.UPDATE_TIMESTEPS:
            agent.update()
            agent.update_learning_rate(global_step, max_train_steps)
            
    return agent.network

# ==========================================
# 4. 【并行 Worker 函数】：执行独立实验
# ==========================================
def worker_task(task_spec):
    """
    运行在独立进程中的任务函数。
    Windows spawn 机制会为其分配完全独立的内存空间，所以直接修改 Config 是绝对安全的。
    """
    var_type, val, label, test_episodes = task_spec
    
    # 1. 强制复位 Baseline 参数
    Config.H_MATRIX = Config.H_2
    Config.MU = np.array([0.24, 0.45, 0.52])
    Config.C_M = 500.0
    Config.C_F = 1500.0

    # 2. 注入当前实验的特异性参数
    if var_type == 'H': Config.H_MATRIX = val
    elif var_type == 'MU': Config.MU[0] = val
    elif var_type == 'CM': Config.C_M = float(val)
    elif var_type == 'CS': Config.C_F = float(val)

    env = UAVEnvironment()
    norm_factors = np.append(Config.L_THRESHOLDS, env.max_steps)
    
    # 3. 寻优 & 训练
    best_ra0, best_th = optimize_heuristics(env, evolve_episodes=200)
    ppo_network = train_ppo_in_memory(env, max_train_steps=Config.MAX_TRAIN_STEPS)
    ppo_network.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. 定义四条策略
    def policy_1(s):
        return 0 if np.any(s[:Config.K] / Config.L_THRESHOLDS > best_ra0) else 1
    def policy_2(s):
        return 0 if np.any(s[:Config.K] > best_th) else 1
    def policy_3(s):
        return 1
    def policy_ppo(s):
        payload_failed = bool(s[0] >= Config.L_THRESHOLDS[0])
        s_tensor = torch.FloatTensor(s / norm_factors).unsqueeze(0).to(device)
        m_tensor = torch.BoolTensor([payload_failed]).to(device)
        with torch.no_grad():
            probs, _ = ppo_network(s_tensor, payload_failed_mask=m_tensor)
            # 置信度大于 60% 才中止，突破 PTSD 瓶颈
            return 0 if probs[0][0].item() > 0.6 else 1

    # 5. 在 CRN 考场终极对决
    s1 = run_sim(env, policy_1, test_episodes)
    s2 = run_sim(env, policy_2, test_episodes)
    s3 = run_sim(env, policy_3, test_episodes)
    s_ppo = run_sim(env, policy_ppo, test_episodes)
    
    # 打印单项实验完成日志
    print(f"✅ [完成] {var_type}={label} | PPO: {s_ppo:.2f} | P2: {s2:.2f} | P1: {s1:.2f} | P3: {s3:.2f}")
    
    return (var_type, label, s1, s2, s3, s_ppo)

# ==========================================
# 5. 主控调度与画图模块
# ==========================================
if __name__ == "__main__":
    # >>> 创建三级嵌套输出目录 result -> sens -> timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("result", "sens", run_timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print("=====================================================")
    print("🚀 启动全面敏感度分析自动化流水线 (顶层并行加速版)")
    print(f"📂 结果将保存至: {save_dir}")
    print("=====================================================\n")
    
    test_ep = 3000 # 终极评估次数
    
    # 构建所有需要独立运行的任务列表 [var_type, value, label, test_episodes]
    tasks = []
    
    # 1. H 矩阵实验 (4个)
    h_matrices = [Config.H_0, Config.H_1, Config.H_2, Config.H_3]
    for i, h_val in enumerate(h_matrices):
        tasks.append(('H', h_val, f'H{i}', test_ep))
        
    # 2. MU 实验 (4个)
    for mu in [0.20, 0.22, 0.24, 0.26]:
        tasks.append(('MU', mu, mu, test_ep))
        
    # 3. CM 实验 (4个)
    for cm in [300, 500, 700, 900]:
        tasks.append(('CM', cm, cm, test_ep))
        
    # 4. CS (CF) 实验 (4个)
    for cs in [1000, 1500, 2000, 2500]:
        tasks.append(('CS', cs, cs, test_ep))

    # 初始化存储结构
    results = {
        'H':  {'x': ['H0', 'H1', 'H2', 'H3'], 'p1':[], 'p2':[], 'p3':[], 'ppo':[]},
        'MU': {'x': [0.20, 0.22, 0.24, 0.26], 'p1':[], 'p2':[], 'p3':[], 'ppo':[]},
        'CM': {'x': [300, 500, 700, 900],     'p1':[], 'p2':[], 'p3':[], 'ppo':[]},
        'CS': {'x': [1000, 1500, 2000, 2500], 'p1':[], 'p2':[], 'p3':[], 'ppo':[]}
    }

    print(f"⏳ 总计 {len(tasks)} 个独立实验条件，已抛入进程池同时计算...")
    start_time = time.time()
    
    # 开启宏观多进程并行
    # 建议保留2个核心给系统，防止电脑卡死，同时避免瞬间拉起太多进程导致 GPU OOM
    cores = max(1, mp.cpu_count() - 2) 
    
    # 使用 ProcessPoolExecutor 分发任务
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = [executor.submit(worker_task, task) for task in tasks]
        
        # 收集结果
        for future in as_completed(futures):
            try:
                var_type, label, s1, s2, s3, spp = future.result()
                
                # 获取该 label 在 x 轴列表中的索引，以保证顺序写入
                idx = results[var_type]['x'].index(label)
                
                # 因为列表不能按索引随便 append，所以先占位，再替换
                if len(results[var_type]['p1']) == 0:
                    results[var_type]['p1'] = [0] * 4
                    results[var_type]['p2'] = [0] * 4
                    results[var_type]['p3'] = [0] * 4
                    results[var_type]['ppo'] = [0] * 4
                    
                results[var_type]['p1'][idx] = s1
                results[var_type]['p2'][idx] = s2
                results[var_type]['p3'][idx] = s3
                results[var_type]['ppo'][idx] = spp
                
            except Exception as e:
                print(f"❌ 某个进程发生错误: {e}")

    print(f"\n🎉 16 组实验全部并行计算完毕！总耗时: {(time.time() - start_time)/60:.2f} 分钟")

    # ================= 绘图阶段 =================
    print(f"📊 正在生成图表并保存至 {save_dir} ...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    axs = axs.flatten()
    
    configs_plot = [
        ('H', 'Degradation Interaction Matrix', axs[0]),
        ('MU', 'Degradation Rate of Payload ($\mu_{E11}$)', axs[1]),
        ('CM', 'Task Failure Cost ($C_m$)', axs[2]),
        ('CS', 'System Failure Cost ($C_s$)', axs[3])
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for key, xlabel, ax in configs_plot:
        x = results[key]['x']
        ax.plot(x, results[key]['ppo'], color=colors[3], marker=markers[3], linewidth=2.5, label='Proposed Policy (PPO)')
        ax.plot(x, results[key]['p2'],  color=colors[2], marker=markers[2], linewidth=2, label='Policy 2 (Multi-threshold)', linestyle='--')
        ax.plot(x, results[key]['p1'],  color=colors[1], marker=markers[1], linewidth=2, label='Policy 1 (Fixed Ratio)', linestyle='--')
        ax.plot(x, results[key]['p3'],  color=colors[0], marker=markers[0], linewidth=2, label='Policy 3 (Never Abort)', linestyle=':')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Expected Cumulative Reward', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        
    plt.suptitle("Sensitivity Analysis of Mission Abort Policies", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 存入你指定的三级嵌套文件夹
    output_png = os.path.join(save_dir, "sensitivity_analysis_results.png")
    plt.savefig(output_png, bbox_inches='tight')
    print(f"✅ 图表已完美保存！路径: {output_png}")