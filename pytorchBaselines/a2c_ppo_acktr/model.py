import torch
import torch.nn as nn
import numpy as np

from pytorchBaselines.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from pytorchBaselines.a2c_ppo_acktr.srnn_model import SRNN


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base == 'srnn':
            base=SRNN
            self.base = base(obs_shape, base_kwargs)
            self.srnn = True
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]

            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        if not hasattr(self, 'srnn'):
            self.srnn = False
        if self.srnn:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, infer=True)

        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        # inputs is a dictionary of keys: ['robot_node', 'spatial_edges', 'temporal_edges']
        #print("input robot_node shape", inputs['robot_node'].shape)
        #print("input robot_node: ", inputs['robot_node'])
        # actor_features is a tensor of shape [1, 256]
        # Dist is a pytorchBaselines.a2c_ppo_acktr.distributions.FixedNormal class
        # dist has 
        #print("dist variables: ", dist.__dict__.keys())
        #print("dist: ", dist)
        #print("dist: ", dist["loc"])

        is_safety_shield = True
        if is_safety_shield: 
            human_rad = 0.3 # [m] default
            human_vmax = 1  # [m/s] default
            dt = 0.25 # [s] default
            k = 1 # time steps in the future 
            num_agents = len(inputs["spatial_edges"][0]) 
            robot_pos = inputs['robot_node'][0, 0, 0:2].numpy()  # robot px, py
            counter = 0
            action = dist.mode().numpy()
            #print("robot_pos: ", robot_pos)
            for ag in range(num_agents):
                counter = 0
                human_pos = inputs['spatial_edges'][0, ag, 0:2].numpy() + robot_pos # human px, py
                while counter < 10:
                    fut_pos = robot_pos + action*dt*k
                    # distance between robot and human
                    distance = np.linalg.norm(fut_pos - human_pos)
                    if distance < human_rad + human_vmax*dt:
                        # action is unsafe
                        print("action is unsafe!")
                        action = dist.sample().numpy()
                    else:
                        break
                    counter += 1
                    print("Counter threshold reached!") if counter == 10 else None
        
            action = torch.from_numpy(action)
        else:
            # maybe change the action after taking the mode or sample

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        # resulted action is a tensor of shape [1, 2]
        #print("action: ", action)
        #print(" ")
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):

        value, _, _ = self.base(inputs, rnn_hxs, masks, infer=True)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs



