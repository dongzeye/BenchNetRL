import gymnasium as gym
import numpy as np
import envs.finite_pomdp # Ensure the custom environments are registered

def test_env(env_id, num_steps=10):
    print(f"Testing {env_id}...")
    try:
        env = gym.make(env_id)
        obs, info = env.reset()
        print(f"  Reset successful. Initial observation: {obs}")
        
        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"    Step {i+1}: Action={action}, Reward={reward:.4f}, Terminated={terminated}, Truncated={truncated}")
            if terminated or truncated:
                print(f"  Episode finished at step {i+1}")
                obs, info = env.reset()
        
        env.close()
        print(f"  {env_id} PASSED.")
    except Exception as e:
        print(f"  {env_id} FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    envs_to_test = [
        "Tiger-Theta2-v0",
        "Tiger-Theta3-v0",
        "Tiger-Theta4-v0",
        "RiverSwim-Hard-v0",
        "SparseRewardPOMDP-Random0-v0",
        "SparseRewardPOMDP-Random1-v0",
        "SparseRewardPOMDP-Random2-v0",
        "SparseRewardPOMDP-Random3-v0",
        "SparseRewardPOMDP-Random4-v0",
    ]
    
    for env_id in envs_to_test:
        test_env(env_id)
        print("-" * 40)
