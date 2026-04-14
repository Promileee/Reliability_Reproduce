import numpy as np
import time
from environment import UAVEnvironment

# ==========================================
# 1. 启发式策略
# ==========================================
def policy_1(state, config, ra0):
    X = state[:config.K]
    ratios = X / config.L_THRESHOLDS
    if np.any(ratios > ra0):
        return 0 # DA (Abort)
    return 1     # DC (Continue)

def policy_2(state, config, custom_thresholds):
    X = state[:config.K]
    if np.any(X > custom_thresholds):
        return 0 # DA
    return 1     # DC

def policy_3(state, config):
    return 1

# ==========================================
# 2. 蒙特卡洛评估核心
# ==========================================
def evaluate_policy_core(env, policy_fn, num_episodes):
    """底层的仿真循环 (引入 CRN 技术消除运气偏差)"""
    total_rewards = []
    
    for ep in range(num_episodes):
        # 【核心魔法】：公共随机数 (CRN)
        # 强制每一局都使用固定的初始种子，保证各策略面临相同的退化轨迹
        np.random.seed(10000 + ep)
        
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action = policy_fn(state, env.config)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        
    # 评估结束后，解除种子锁定，防止污染外部其他的随机过程
    np.random.seed(None)
    
    return np.mean(total_rewards)

# ==========================================
# 3. 主程序 (用户自定义参数评估)
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("🚀 启动蒙特卡洛评估计算引擎 (用户自定义阈值版)")
    print("==================================================\n")

    env_main = UAVEnvironment()
    
    # ------------------------------------------------
    # 👇 用户在此处手动设定阈值参数 👇
    # ------------------------------------------------
    
    # Policy 1: 固定比例阈值 
    # (例如 0.8 表示任意部件退化量达到自身失效阈值的 80% 即中止)
    user_ra0 = 0.4  
    
    # Policy 2: 多维独立阈值 
    # (针对不同部件设定不同的绝对退化量阈值。这里暂以 L_THRESHOLDS 的特定比例为例，你可以改成任意绝对数值)
    L = env_main.config.L_THRESHOLDS
    user_custom_thresholds = np.array([L[0] * 0.4, L[1] * 0.5, L[2] * 0.5]) 
    
    # 评估回合数 (建议 1000 - 5000 次)
    eval_episodes = 1000 
    
    # ------------------------------------------------
    
    print(f"🔹 Policy 1 设定的比例阈值 (ra0): {user_ra0}")
    print(f"🔹 Policy 2 设定的多维阈值: {np.round(user_custom_thresholds, 3)}")
    print(f"🔹 评估样本量: {eval_episodes} 次")
    print("\n⏳ 正在评估中，请稍候...")
    
    start_time = time.time()

    # 包装策略以注入用户自定义的参数
    def final_p1_wrapper(s, c): return policy_1(s, c, user_ra0)
    def final_p2_wrapper(s, c): return policy_2(s, c, user_custom_thresholds)

    # 执行评估
    mean_1 = evaluate_policy_core(env_main, final_p1_wrapper, eval_episodes)
    mean_2 = evaluate_policy_core(env_main, final_p2_wrapper, eval_episodes)
    mean_3 = evaluate_policy_core(env_main, policy_3, eval_episodes)

    print(f"✅ 评估完成 (耗时: {time.time()-start_time:.1f}s)")

    print("\n==================================================")
    print("🏆 评估结果对比 (基于公共随机数):")
    print(f"Policy 2 (用户自定义多维阈值) : {mean_2:.2f}")
    print(f"Policy 1 (用户自定义比例阈值) : {mean_1:.2f}")
    print(f"Policy 3 (不实施中止策略)     : {mean_3:.2f}")
    print("==================================================")