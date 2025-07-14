from datetime import datetime
import numpy as np
import gym
from rbf_layer import RBF, RBF_eval
import envs
from scipy.stats import qmc
from utils import load_st_idt, split_exp_group
from ppo import PPO
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

################################### Training ###################################
def train():
    print("============================================================================================")
    ####### initialize environment hyperparameters ######
    env_name = "NH3AutoIgnEnv-v0"
    max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps

    action_std = 0.05  # starting std for action distribution (Multivariate Normal)

    update_timestep = 100 # update policy every n timesteps
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    lr_actor = 0.0007  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    print("training environment name : " + env_name)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    mech_path = './nh3_mech/NH3_Liao2023_THU.yaml'
    st_idt_paths = ['./exp_data/st_idt{}.csv'.format(i + 1) for i in range(3)]
    exp_XFO, exp_TP, exp_phi, exp_idt = load_st_idt(st_idt_paths, mech_path)

    # UF analysis
    UF_path = './UF_rxns.csv'
    UF = np.array(pd.read_csv(UF_path))[:, 1:]
    UF = np.ravel(UF)
    UF = UF[~np.isnan(UF)]
    UF_E = np.ones(275) * 0.2
    UF = np.concatenate((UF, UF_E))

    group_indices = split_exp_group(exp_XFO, exp_TP, exp_idt)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, K_epochs, eps_clip, action_std)

    sampler = qmc.LatinHypercube(d=len(UF))
    sam = -1 + 2 * sampler.random(n=1000)

    fit = np.sum((sam-0.5)**2, axis=-1)/10

    ppo_agent.buffer.all_states_ = sam.copy()
    ppo_agent.buffer.all_rewards = -fit

    all_rewards = -fit.reshape(-1, 1)
    flag = 'cubic'
    lambda_, gamma = RBF(sam, all_rewards, flag)
    FUN = lambda x: RBF_eval(x, sam, lambda_, gamma, flag)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)

    time_step = 0
    # training loop

    while time_step <= max_training_timesteps:
        state = env.reset()
        # select action with policy
        action = ppo_agent.select_action(state)
        state_, reward, done, _ = env.step(action)
        ppo_agent.buffer.states_.append(state_)

        time_step += 1

        # update PPO agent
        if time_step % update_timestep == 0:
            old_states_ = np.stack(ppo_agent.buffer.states_, axis=0)
            rewards = FUN(old_states_).flatten()
            indices = np.argsort(rewards)[-100:]
            fit = np.sum((old_states_[indices]-0.5)**2, axis=-1)/10

            ppo_agent.buffer.all_states_ = np.vstack([ppo_agent.buffer.all_states_, old_states_[indices]])
            ppo_agent.buffer.all_rewards = np.concatenate([ppo_agent.buffer.all_rewards, -fit])

            FUN = ppo_agent.update(time_step)
            ppo_agent.buffer.clear()

    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
