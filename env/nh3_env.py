import numpy as np
from gym import spaces
import gym
import pandas as pd
import envs
import cantera as ct
import models
from scipy.stats import multivariate_normal
import torch
from utils import load_st_idt, get_A, split_exp_group
import time


class NH3AutoIgnEnv(gym.Env):
    def __init__(self, args):
        # environment
        self.mech_path = args.mech_path
        ign_files = ['./exp_data/st_idt{}.csv'.format(i + 1) for i in range(3)]
        self.exp_XFO, self.exp_TP, self.exp_phi, self.exp_idt = load_st_idt(ign_files,
                                                                            self.mech_path)

        exp_logmean = np.log(self.exp_idt)
        exp_cov = np.diag(np.full(len(exp_logmean), args.exp_sigma**2))
        self.likelihood = multivariate_normal(exp_logmean, exp_cov)

        # UF analysis
        UF_path = './UF_rxns.csv'
        UF = np.array(pd.read_csv(UF_path))[:, 1:]
        UF = np.ravel(UF)
        UF = UF[~np.isnan(UF)]

        # load sl models
        self.sl_models = []
        for i in range(len(self.exp_idt)):
            model = models.SL_Network(len(UF), 1)
            model_path = './sl_results/sl_models/exp_IDT{:d}.pt'.format(i)
            model.load_state_dict(
                torch.load(model_path, map_location=lambda storage, loc: storage
                           )['model'])
            model.eval()
            self.sl_models.append(model)

        self.A_step = args.step_length

        # build the parameter space
        gas = ct.Solution(self.mech_path)
        A0 = get_A(gas, UF)
        self.model_mean = np.log(A0)
        self.model_sigma = np.log(UF) / 2
        self.prior_dist = multivariate_normal(self.model_mean, np.diag(self.model_sigma**2))

        self.observation_space = spaces.Box(0, 1, shape=(len(UF) + 1,))
        self.action_space = spaces.Box(0, 1, shape=(len(UF),))

    def step(self, action):
        self.A += (2 * action - 1) * self.A_step
        self.A = self.A.clip(-1, 1)

        y_pred = np.zeros(len(self.exp_idt))
        for i in range(len(self.exp_idt)):
            y_pred[i] = self.sl_models[i](torch.tensor(self.A, dtype=torch.float32)).detach().numpy()

        logA_prop = 2 * self.model_sigma * self.A + self.model_mean
        P_prop = self.prior_dist.logpdf(logA_prop) + self.likelihood.logpdf(y_pred)

        reward = P_prop / 500
        done = True

        state = np.concatenate([self.A, [P_prop / 500]])
        self.P_prev = P_prop

        return state, reward, done, {}

    def reset_A(self, A):
        self.A = A.copy()

        y_pred = np.zeros(len(self.exp_idt))
        for i in range(len(self.exp_idt)):
            y_pred[i] = self.sl_models[i](torch.tensor(self.A, dtype=torch.float32)).detach().numpy()

        logA_prop = 2 * self.model_sigma * self.A + self.model_mean
        P_prop = self.prior_dist.logpdf(logA_prop) + self.likelihood.logpdf(y_pred)

        state = np.concatenate([self.A, [P_prop / 500]])
        return state

    def reset(self):
        self.A = np.random.normal(0, 0.5,
                                  size=self.observation_space.shape[0]-1).clip(-1, 1)
        # self.A = np.random.uniform(-1, 1, size=self.observation_space.shape[0]-1)

        y_pred = np.zeros(len(self.exp_idt))
        for i in range(len(self.exp_idt)):
            y_pred[i] = self.sl_models[i](torch.tensor(self.A, dtype=torch.float32)).detach().numpy()

        logA_prop = 2 * self.model_sigma * self.A + self.model_mean
        P_prop = self.prior_dist.logpdf(logA_prop) + self.likelihood.logpdf(y_pred)

        self.P_prev = P_prop

        state = np.concatenate([self.A, [P_prop / 500]])

        return state

