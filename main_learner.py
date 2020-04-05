from stable_baselines.sac.policies import LnMlpPolicy
from sac import CommSAC
from callback import CustomCallback
from stable_baselines import DDPG

import gym

if __name__ == "__main__":
    env = gym.make('LunarLanderContinuous-v2')
    print(env.observation_space.shape)
    print(env.action_space.shape)
    agent = CommSAC(LnMlpPolicy, env, verbose=1, tensorboard_log='./mb_tensorboard')
    agent.learn(total_timesteps=1000000, log_interval=10, callback=CustomCallback())