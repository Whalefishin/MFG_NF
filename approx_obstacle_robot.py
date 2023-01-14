import klampt
from klampt.plan import cspace,robotplanning
from klampt.plan.robotcspace import RobotCSpace
from klampt.io import resource
from klampt.model import collide
import time
from klampt.model.trajectory import RobotTrajectory

import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np


### world & robot setup ###
world = klampt.WorldModel()
world.readFile("./Klampt-examples/data/tx90cuptable_2.xml")
robot = world.robot(0)

# this is the C-space that will be used. Standard collision and joint limit constraints will be checked
space = RobotCSpace(robot, collide.WorldCollider(world))

#fire up a visual editor to get some start and goal configurations
# qstart = resource.get("under_table_start_4.config")
qstart = resource.get("default_start.config")
qgoal  = resource.get("cup_end_2.config")


### path-planning ###
settings = {'type':'rrt',
    'perturbationRadius':0.25,
    'bidirectional':True,
    'shortcut':True,
    'restart':True,
    'restartTermCond':"{foundSolution:1,maxIters:1000}"
}
t0 = time.time()
print("Creating planner...")
#Manual construction of planner
planner = cspace.MotionPlan(space, **settings)
planner.setEndpoints(qstart, qgoal)


torch.manual_seed(1)

# use NN to fit the obs
d = 12
d_in = 10
n_epochs = 2000
N = 1000
# B = 32
# B = 256
B = 1024
device = torch.device('cuda')
obs_val = 1.
# h = 512
# l = 1
# lr = 1e-3

h = 512
l = 3
lr = 1e-3

# h = 64
# l = 3
# lr = 1e-4

I_dummy = [0, 7]
I_keep = [i for i in range(d) if i not in I_dummy]

# # model
# class Obstacle(nn.Module):
#     def __init__(self, d_in, h):
#         super(Obstacle, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_in, h),
#             # nn.ReLU(),
#             nn.Tanh(),
#             nn.Linear(h, h),
#             # nn.ReLU(),
#             nn.Tanh(),
#             nn.Linear(h, 1)
#         )
    
#     def forward(self, x):
#         # return self.net(x)
#         return torch.exp(self.net(x))

class Obstacle(nn.Module):
    def __init__(self, d_in, h, l):
        super(Obstacle, self).__init__()
        self.linear_in     = nn.Linear(d_in, h)
        self.linear_layers = nn.ModuleList([nn.Linear(h,h) for i in range(l)])
        self.linear_out    = nn.Linear(h, 1) 
        # self.act_fun = nn.Tanh()
        self.act_fun = nn.ReLU()
    
    def forward(self, x):
        x = self.act_fun(self.linear_in(x))
        for l in self.linear_layers:
            x = self.act_fun(l(x)) + x

        # out = torch.exp(self.linear_out(x))
        out = torch.sigmoid(self.linear_out(x))

        return out

obs = Obstacle(d_in, h, l).to(device)
optimizer = torch.optim.Adam(obs.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

# data

def process_robot_data(x, mode='preprocess'):
    # x: B x d
    d = 12
    I_dummy = [0, 7]
    I_keep = [i for i in range(d) if i not in I_dummy]
    if mode == 'preprocess':
        x_hat = x[..., I_keep]
    else:
        x_hat = torch.zeros(x.shape[0], d)
        x_hat[..., I_keep] = x
        # x_hat[:, I_insert] 

    return x_hat


A    = torch.tensor(space.bound)
low  = A[:,0]
high = A[:,1]
low  = process_robot_data(low)
high = process_robot_data(high)
dist = D.Uniform(low, high)

width = 5

# debugging
# x_pre = dist.sample((B,))
# x = process_robot_data(x_pre)
# x = x.to(device)
# y = (1. - torch.tensor([space.isFeasible(a.numpy()) for a in x_pre]).to(device).float())

for epoch in range(n_epochs):
    loss_list = []
    for i in range(N):
        optimizer.zero_grad()
        # create training data
        # x_pre = dist.sample((B,))
        # x_pre[..., I_keep] += 0.1 * torch.randn_like(x_pre[..., I_keep])
        # x = process_robot_data(x_pre)
        # x = x.to(device)

        x = dist.sample((B,))
        x += 0.1 * torch.randn_like(x)
        x_pre = process_robot_data(x, mode='postprocess')
        x = x.to(device)

        # y = obs_val * (1 - torch.tensor([space.isFeasible(a.numpy()) for a in x_pre]).to(device).float())
        # loss = torch.mean((obs(x).reshape(-1) - y)**2)

        y = (1. - torch.tensor([space.isFeasible(a.numpy()) for a in x_pre]).to(device).float())
        loss = torch.nn.functional.binary_cross_entropy(obs(x).reshape(-1), y)
        # loss = torch.mean((obs(x).reshape(-1) - y)**2)
        # loss = torch.mean(torch.abs(obs(x).reshape(-1) - y))

        loss.backward()
        optimizer.step()
        loss_list.append(float(loss))

    loss_mean = np.mean(loss_list)
    # scheduler.step(loss_mean)
    scheduler.step()
    print ("Epoch: {}, loss: {:.4f} += {:.4f}".format(epoch, loss_mean, 2*np.std(loss_list)/np.sqrt(len(loss_list))))
    

# saving models
path = './results/robot_1/obs_NN_thick_sigmoid_B=1024.t'
torch.save(obs.state_dict(), path)

