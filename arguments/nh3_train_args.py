import argparse

# define some arguments that will be used...
def achieve_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--mech_path', type=str, default='./nh3_mech/NH3_Liao2023_THU.yaml')
    parse.add_argument('--episode_length', type=int, default=1, metavar='LENGTH', help='Episode length')
    parse.add_argument('--step_length', type=float, default=0.02, metavar='LENGTH', help='Episode length')
    parse.add_argument('--model_path', type=str, default=None)
    parse.add_argument('--exp_sigma', type=float, default=0.1)
    parse.add_argument('--seed', type=int, default=123, help='the random seed')
    parse.add_argument('--policy_lr', type=float, default=3e-4, help='the learning rate of actor network')
    parse.add_argument('--value_lr', type=float, default=3e-4, help='the learning rate of critic network')
    parse.add_argument('--batch_size', type=int, default=1, help='the batch size of the training')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount ratio...')
    parse.add_argument('--policy_update_step', type=int, default=10, help='the update number of actor network')
    parse.add_argument('--value_update_step', type=int, default=10, help='the update number of critic network')
    parse.add_argument('--epsilon', type=float, default=0.2, help='the clipped ratio...')
    parse.add_argument('--tau', type=float, default=0.95, help='the coefficient for calculate GAE')
    parse.add_argument('--env_name', default="NH3AutoIgnEnv-v0", help='environments name')
    parse.add_argument('--collection_length', type=int, default=10, help='the sample collection length(episodes)')
    parse.add_argument('--cores', type=int, default=20, help='number of parallel cpus')

    args = parse.parse_args()

    return args
