import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from rbf_layer import RBF, RBF_eval
import numpy as np
from scipy.spatial.distance import cdist

################################## set device ##################################
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.states_ = []
        self.all_rewards = []
        self.FUN = lambda x: np.zeros((x.shape[0], 1))

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.states_[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        dist = Independent(Normal(action_mean, self.action_var), 1)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        dist = Independent(Normal(action_mean, action_var), 1)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, K_epochs, eps_clip,
                 action_std_init=0.1):

        self.action_std = action_std_init
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.lr_critic = lr_critic
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten()

    def update(self, id):
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_states_ = np.stack(self.buffer.states_, axis=0)

        all_rewards = self.buffer.all_rewards.reshape(-1, 1)
        Ns = 2 * (275*2 + 1)

        phdis = cdist(old_states_, self.buffer.all_states_)
        nidx = np.argsort(phdis, axis=1)[:, :Ns]
        nid = np.unique(nidx)
        lhx = self.buffer.all_states_[nid, :]
        lhf = all_rewards[nid, :]

        flag = 'cubic'
        lambda_, gamma = RBF(lhx, lhf, flag)
        FUN = lambda x: RBF_eval(x, lhx, lambda_, gamma, flag)
        rewards = FUN(old_states_).flatten()

        print("Timestep : {} \t\t Average Reward : {:e}".format(id + 1,
                                                                np.max(self.buffer.all_rewards)),
              Ns)

        # calculate advantages
        rewards = torch.squeeze(torch.tensor(rewards, dtype=torch.float32), dim=-1).to(device)
        # print(rewards, self.buffer.rewards)
        advantages = rewards.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return FUN
