import argparse

# define some arguments that will be used...
def achieve_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mech', type=str, default='./NH3_Mech/reduced_17_SI.yaml')
    parse.add_argument('--key_rxns_idx', type=int, default=[0, 65, 56, 63, 66])
    #0, 59, 65, 64, 13, 73, 126, 60, 130, 125
    parse.add_argument('--ctrl_steps', type=int, default=2000)
    parse.add_argument('--ctrl_dt', type=float, default=1E-6)

    parse.add_argument('--seed', type=int, default=123, help='the random seed')
    parse.add_argument('--episode_length', type=int, default=10, metavar='LENGTH', help='Episode length')
    parse.add_argument('--env_name', default="NH3AutoIgnEnv-v0", help='environments name')

    args = parse.parse_args()

    return args


