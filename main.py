from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO,DDPG,DQN,A2C
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import gym
import numpy as np
import time
from stable_baselines3.common.callbacks import EvalCallback
import env
env = env.ElectricVehicleEnv()
#vec_env = make_vec_env(env, n_envs=4)
#check_env(env)
env = DummyVecEnv([lambda: env])




# 设置日志目录和TensorBoard日志
log_dir = "log"
# 配置logger，使其在log目录下存储TensorBoard日志

model =A2C("MlpPolicy", env, verbose=1, tensorboard_log= log_dir, device="cuda")
model.learn(total_timesteps=100000 )

# 保存模型
model.save("dqn_electric_vehicle")

# 加载模型（如果需要）
# model = DQN.load("dqn_electric_vehicle")

# 评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


