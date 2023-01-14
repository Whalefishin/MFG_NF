import torch
import torch.nn as nn
import torch.distributions as D
from mfg_bilevel_models import Obstacle_gaussian, Obstacle, Obstacle_true
import copy 
import argparse
from mfp_utils import sanitize_args, eval_obstacles_on_grid, create_grid_for_eval, plot_obstacles
from arguments import parse_arguments


d_in = 2
N = 10000
# N = 10
B = 32
args = parse_arguments()
device = torch.device('cuda')

class model_to_copy(nn.Module):
    def __init__(self):
        super(model_to_copy, self).__init__()
        self.mean = nn.Parameter(torch.zeros(2))
        self.diag = nn.Parameter(torch.Tensor([1., 0.5]))

    def forward(self, x):
        B = D.MultivariateNormal(self.mean, torch.diag_embed(self.diag)) 
        return torch.exp(B.log_prob(x[..., :2]))

grid, grid_pad, grid_x, grid_y, dx = create_grid_for_eval(args, d_in)
grid = grid.to(device)
grid_pad = grid_pad.to(device)

obs = Obstacle(d_in, args).to(device)
obs_true = Obstacle_true(args, device)
# obs = Obstacle_gaussian().to(device)

optimizer = torch.optim.Adam(obs.parameters())

for i in range(N):
    # x = 8. * torch.rand(B, 2) - 4. # x in [-4,4]^2
    x = 2. * torch.rand(B, 2) # x in [0,2]^2
    x = x.to(device)

    optimizer.zero_grad()
    loss = torch.mean((obs(x).reshape(-1) - obs_true.eval(x))**2)
    loss.backward()
    optimizer.step()

    if (i+1) % args.monitor_interval == 0:
        print ("Loss value: {:.5f}".format(float(loss)))
        # print ("Mean: {}; Diag: {}".format(obs.mean.data, obs.diag.data))


path = './results/crowd_motion_two_bars_bilevel/pretrain_obs.t'
torch.save(obs.state_dict(), path)

B_val, B_true_val = eval_obstacles_on_grid(obs, obs_true, grid, grid_pad)
fig_name = 'obs_comparison'
plot_dir = './grapher/images'
plot_obstacles(B_val, B_true_val, grid_x, grid_y, plot_dir, None, False, fig_name)