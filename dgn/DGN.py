import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: (
    autograd.Variable(*args, **kwargs).cuda()
    if USE_CUDA
    else autograd.Variable(*args, **kwargs)
)


class CNNLayer(nn.Module):
    def __init__(self, obs_shape):
        super(CNNLayer, self).__init__()

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, input_channel * 2, 2, 2),
            nn.ReLU(),
            nn.Conv2d(input_channel * 2, input_channel * 2, 2, 2),
            nn.ReLU(),
            nn.Conv2d(input_channel * 2, input_channel * 2, 2, 2),
            nn.ReLU(),
        )

        dummy_input = torch.randn(obs_shape)
        dummy_output = self.cnn(dummy_input)
        self.flatten_dim = dummy_output.view(-1).shape[0]

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return x


class Encoder(nn.Module):
    def __init__(self, observation_shape=32, hidden_dim=128):
        super(Encoder, self).__init__()
        # print("observation_shape", observation_shape)

        self.img_obs = len(observation_shape) >= 3
        if self.img_obs:
            self.cnn = CNNLayer(observation_shape)

        din = self.cnn.flatten_dim if self.img_obs else observation_shape[0]
        self.fc = nn.Sequential(
            nn.Linear(din, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.img_obs:
            x = self.cnn(x)
        embedding = self.fc(x)
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        h = torch.clamp(torch.mul(torch.bmm(q, k), mask), 0, 9e13) - 9e15 * (1 - mask)
        att = F.softmax(h, dim=2)

        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        # out = F.relu(self.fcout(out))
        return out


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


class DGN(nn.Module):
    def __init__(self, n_agent, observation_space, hidden_dim, num_actions):
        super(DGN, self).__init__()

        self.encoder = Encoder(observation_space.shape, hidden_dim)
        self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.q_net = Q_Net(hidden_dim * 2, num_actions)

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        # h3 = self.att_2(h2, mask)

        h = torch.cat([h1, h2], dim=-1)

        q = self.q_net(h)
        return q
