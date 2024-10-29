import numpy as np


class Envs:

    def __init__(self, name: str, args: dict):
        self.name = name.lower()
        if name.lower() == "magent":
            pass
        if name.lower() == "lbf":
            pass
        if name.lower() == "gridworld":
            from gridworld.gridworld_env import GridworldEnv

            self.env = GridworldEnv(args["plan"], separated_rewards=True)
            self.n_agents = self.env.n_agents

        if name.lower() == "mpe":
            from MPE.MPE_env import MPEEnv
            from argparse import Namespace

            args = Namespace(**args)

            self.env = MPEEnv(args)
            self.n_agents = len(self.env.agents)

        if name.lower() == "overcook":
            pass

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        assert (
            self.action_space[0].__class__.__name__ == "Discrete"
        ), "Only support discrete action spaces"

        self.image_obs = len(self.observation_space[0].shape) >= 3

    def reset(self):
        adj = np.ones((self.n_agents, self.n_agents), dtype=bool)
        if self.name == "gridworld":
            obses, state, avail = self.env.reset()
        return (obses, adj)

    def step(self, action):
        next_state_returned = self.env.step(action)

        next_adj = np.ones((1, self.n_agents, self.n_agents), dtype=bool)

        if self.name == "gridworld":
            next_obs, states, rewards, dones, infos, avail_ = next_state_returned
        if self.name == "mpe":
            next_obs, rewards, dones, infos
        if self.name == "overcook":
            pass
        if self.name == "lbf":
            pass
        if self.name == "magent":
            pass

        next_obs = np.array(next_obs)
        rewards = np.array(rewards).squeeze()
        dones = np.array(dones)
        if self.image_obs:
            print(len(next_obs.shape))
            next_obs = np.transpose(next_obs, [0, 3, 1, 2])

        return next_obs, next_adj, rewards, dones
