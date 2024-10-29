import math, random, copy
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from DGN import DGN
from buffer import ReplayBuffer
from envs import Envs
from config import *


def set_all_seeds(seed):
    random.seed(seed)
    # os.environ('PYTHONHASHSEED') = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_dir(name, seed):
    """Init directory for saving results."""
    import time

    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(
        "result",
        name,
        "DGN",
        "report",
        "-".join(["seed-{:0>5}".format(seed), hms_time]),
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter

    writter = SummaryWriter(log_path)
    # models_path = os.path.join(results_path, "models")
    # os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, writter


def print_all_shape(kwargs):
    print("=" * 10)
    for key, val in kwargs.items():
        print(f"{key}: {val.shape}")


def train(args):
    print(args)
    USE_CUDA = torch.cuda.is_available()
    set_all_seeds(args["seed"])

    device = "cuda" if USE_CUDA else "cpu"

    env = Envs(args["name"], args=args)
    n_ant = env.n_agents
    print("num agent:", n_ant)
    observation_space = env.observation_space[0]
    n_actions = env.action_space[0].n
    print("num actions:", n_actions)

    buff = ReplayBuffer(args["capacity"], observation_space.shape, n_actions, n_ant)
    model = [
        DGN(n_ant, observation_space, args["hidden_dim"], n_actions)
        for _ in range(n_ant)
    ]
    model_tar = [
        DGN(n_ant, observation_space, args["hidden_dim"], n_actions)
        for _ in range(n_ant)
    ]
    model = [m.to(device) for m in model]
    model_tar = [m.to(device) for m in model_tar]
    optimizer = [optim.Adam(m.parameters(), lr=0.0001) for m in model]

    n_episode = (args["num_env_steps"] + args["max_step"] - 1) // args["max_step"]
    GAMMA = args["GAMMA"]
    i_episode = 0

    epsilon = args["epsilon"]
    tau = args["tau"]

    results_path, log_path, writter = init_dir(args["name"], args["seed"])

    f = open(os.path.join(log_path, "r.txt"), "w")
    total_step = 0
    all_scores = []
    while total_step < args["num_env_steps"]:

        if i_episode > 40:
            epsilon *= 0.995
            if epsilon < 0.01:
                epsilon = 0.01
        i_episode += 1
        steps = 0
        score = 0

        obs, adj = env.reset()

        while steps < args["max_step"]:
            steps += 1
            action = []

            for i in range(n_ant):
                with torch.no_grad():
                    q = model[i](
                        torch.Tensor(np.array([obs])).to(device),
                        torch.Tensor(adj).to(device),
                    )[0]
                if np.random.rand() < epsilon:
                    a = np.random.randint(n_actions)
                else:
                    a = q[i].argmax().item()
                action.append(a)

            next_obs, next_adj, reward, done = env.step(action)
            terminated = done.all()
            # print_all_shape(
            #     {
            #         "obs": next_obs,
            #         "adj": next_adj,
            #         "rewards": reward,
            #         "done": terminated,
            #     }
            # )

            # print("reward", reward.shape)
            # exit()

            buff.add(
                np.array(obs),
                action,
                reward,
                np.array(next_obs),
                adj,
                next_adj,
                terminated,
            )
            obs = next_obs
            adj = next_adj
            score += np.mean(reward)

            if terminated.all():
                break

        all_scores.append(score)

        total_step += steps

        if i_episode % 20 == 0:
            avg_score = np.mean(all_scores)
            writter.add_scalar(f"train_episode_rewards", avg_score, total_step)
            writter.add_scalar("epsilon", epsilon, total_step)

            f.write(str(total_step) + ", " + str(avg_score) + "\n")
            print(
                f"episode: {i_episode}, total steps {total_step}, score:",
                str(avg_score) + f", epsilon: {epsilon}" + "\n",
            )
            all_scores = []

        if i_episode < 20:
            continue

        # avg_loss = []
        for e in range(args["n_epoch"]):

            O, A, R, Next_O, Matrix, Next_Matrix, D = buff.getBatch(args["batch_size"])
            O = torch.Tensor(O).to(device)
            Matrix = torch.Tensor(Matrix).to(device)
            Next_O = torch.Tensor(Next_O).to(device)
            Next_Matrix = torch.Tensor(Next_Matrix).to(device)
            A = torch.Tensor(A).long().to(device)
            R = torch.Tensor(R).to(device)

            for i_agent, (m, tar_m, op) in enumerate(zip(model, model_tar, optimizer)):
                # print(m(O, Matrix).shape)
                q_values = m(O, Matrix)[:, i_agent]
                with torch.no_grad():
                    target_q_values = tar_m(Next_O, Next_Matrix)
                    target_q_values = target_q_values.max(dim=2)[0][:, i_agent]
                target_q_values = target_q_values.cpu().data.numpy()
                # expected_q = q_values.copy().detach()

                # q_values = m(
                #     torch.Tensor(O).to(device),
                #     torch.Tensor(Matrix).to(device),
                # )[:, i_agent]
                # target_q_values = tar_m(
                #     torch.Tensor(Next_O).to(device),
                #     torch.Tensor(Next_Matrix).to(device),
                # ).max(dim=2)[0][:, i_agent]
                # target_q_values = np.array(target_q_values.cpu().data)
                # expected_q = np.array(q_values.cpu().data)

                # for j in range(args["batch_size"]):
                #     sample = batch[j]
                #     # for i in range(n_ant):
                #     expected_q[j][sample[1][i_agent]] = (
                #         sample[2][i_agent]
                #         + (1 - sample[6]) * GAMMA * target_q_values[j]
                # )

                expected_q = (
                    R[:, i_agent]
                    + GAMMA * (1 - D.squeeze()) * target_q_values.squeeze()
                )

                q_values = q_values.gather(1, A[:, i_agent].unsqueeze(-1)).squeeze()

                # for j in range(args["batch_size"]):
                #     # for i in range(n_ant):
                #     expected_q[j][A[j][i_agent]] = (
                #         R[j][i_agent]
                #         + (1 - D[j].squeeze()) * GAMMA * target_q_values[j]
                #     )
                # print("expected_q", expected_q.shape)
                # print("A", A.shape)
                # print("target_q_values", target_q_values.shape)
                # print("R", R.shape)
                # print("D", D.shape)
                # exit()

                # print(
                #     "q_values", q_values.shape, "expected_q", expected_q.shape
                # )  # batch x (n_ant) x n_action
                # print("rewards", sample[2][i])
                # exit()
                assert q_values.shape == expected_q.shape
                loss = (q_values - torch.Tensor(expected_q).to(device)).pow(2).mean()
                op.zero_grad()
                loss.backward()
                op.step()

                #     avg_loss.append(loss.item())
                # writter.add_scalar("TD error", np.mean(avg_loss), total_step)
                # if i_episode % 20 == 0:
                #     print("TD error", np.mean(avg_loss))

                with torch.no_grad():
                    for p, p_targ in zip(m.parameters(), tar_m.parameters()):
                        p_targ.data.mul_(tau)
                        p_targ.data.add_((1 - tau) * p.data)
        # if i_episode % 20 == 0:
        # for m_tar, m in zip(model_tar, model):
        #     m_tar.load_state_dict(m.state_dict())


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    import argparse

    parser = argparse.ArgumentParser(
        description="DGN", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    all_args = parser.parse_known_args(args)[0]

    all_args = vars(all_args)

    from config import read_dict_from_json

    all_configs = read_dict_from_json(all_args["config"])
    all_configs.update(all_args)

    train(all_configs)
