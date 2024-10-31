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
            or self.action_space[0].__class__.__name__ == "MultiDiscrete"
        ), "Only support discrete action spaces"

        self.image_obs = len(self.observation_space[0].shape) >= 3

        self.num_actions = []
        for ac_space in self.action_space:
            if ac_space.__class__.__name__ == "MultiDiscrete":
                action_dims = ac_space.high - ac_space.low + 1
                # print("action_dim ", action_dims)
                n_actions = np.prod(action_dims)

                self.num_actions.append(n_actions)
            else:
                self.num_actions.append(ac_space.n)

    def reset(self):
        adj = np.ones((self.n_agents, self.n_agents), dtype=bool)
        if self.name == "gridworld":
            obses, state, avail = self.env.reset()
        if self.name == "mpe":
            obses = self.env.reset()
        return (obses, adj)

    def step(self, action):

        if self.name == "mpe":
            temp_actions_env = []
            for agent_id in range(self.n_agents):
                if self.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    action_dims = (
                        self.action_space[agent_id].high
                        - self.action_space[agent_id].low
                        + 1
                    )
                    action_id = action[agent_id]
                    div = 1
                    action_cvt = [int(action_id % action_dims[-1])]
                    div *= action_dims[-1]
                    for i in range(len(action_dims) - 2, -1, -1):
                        action_cvt.append(int(action_id // div))
                        div *= action_dims[i]
                    action_cvt.reverse()

                    for i in range(self.env.action_space[agent_id].shape):
                        uc_action_env = np.eye(
                            self.env.action_space[agent_id].high[i] + 1
                        )[action_cvt[i]]
                        if i == 0:
                            action_env = uc_action_env
                        else:
                            action_env = np.concatenate(
                                (action_env, uc_action_env), axis=0
                            )
                elif self.action_space[agent_id].__class__.__name__ == "Discrete":
                    action_env = np.squeeze(
                        np.eye(self.env.action_space[agent_id].n)[action[agent_id]]
                    )
                temp_actions_env.append(action_env)
            action = temp_actions_env

        # print("action", action)

        next_state_returned = self.env.step(action)

        next_adj = np.ones((1, self.n_agents, self.n_agents), dtype=bool)

        if self.name == "gridworld":
            next_obs, states, rewards, dones, infos, avail_ = next_state_returned
        if self.name == "mpe":
            next_obs, rewards, dones, infos = next_state_returned
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
