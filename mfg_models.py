import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
from torch.nn.functional import relu, softplus, elu, leaky_relu


def construct_Q(args, device):
    if args.dataset_name == 'crowd_motion_gaussian':
        mean = torch.zeros(2).cuda(device)
        diag = torch.Tensor([1., 0.5]).cuda(device)
        cov  = torch.diag_embed(diag)
        Q = D.MultivariateNormal(mean, cov)
    elif args.dataset_name == 'drones_22_obs':
        mean = torch.Tensor([args.obs_mean_x_22, args.obs_mean_y_22]).cuda(device)
        diag = torch.Tensor([args.obs_var_x_22, args.obs_var_y_22]).cuda(device)
        cov  = torch.diag_embed(diag)
        Q = D.MultivariateNormal(mean, cov)
    elif args.dataset_name == 'robot_1':
        d_in = 10 # removed 2 dummy dimensions from the original 12
        h = 512
        Q = Obstacle_robot(d_in, h ,args).to(device)
        Q.load_state_dict(torch.load(args.obs_dir))
        # disable training for the obstacle
        for p in Q.parameters():
            p.requires_grad = False
    else:
        Q = None

    return Q

class Obstacle_true():
    def __init__(self, args, device):
        super(Obstacle_true, self).__init__()
        self.args = args
        self.device = device
        self.construct_obstacle()
        
    def construct_obstacle(self):
        self.B = construct_Q(self.args, self.device)

    def eval(self, x):
        if x.shape[0] == 0:
            return torch.zeros(0).to(x.device)
        if self.args.Q_true_is_dist:
            return 50 * torch.exp(self.B.log_prob(x[..., :2]))
        else:
            return self.B(x[..., :2])

    def __call__(self, x):
        return self.eval(x)

def sigmoid(x,a,b):
    return 1 / (1 + torch.exp(-a*(x-b)) )

def sigmoid_2D(x, y, a, b_x, b_y):
    # return 1/torch.exp(-a*(x-b_x)) * 1/torch.exp(-a*(y-b_y))
    return sigmoid(x,a,b_x) * sigmoid(y,a,b_y)



def relu_sqr(x):
    return relu(x)**2

def mish(x):
    return x * torch.tanh(softplus(x))

def swish(x):
    return x * torch.sigmoid(x)


class POU(nn.Module):
    def __init__(self, d_in, args):
        super(POU, self).__init__()
        self.args = args
        h = args.h_pou_obs

        self.linear_in     = nn.Linear(d_in, h)
        self.linear_layers = nn.ModuleList([nn.Linear(h,h) for _ in range(args.l_pou_obs)])
        self.linear_out    = nn.Linear(h, args.n_pou_obs) 

        self.act_fun = {'relu': relu, 'tanh': torch.tanh, 'softplus':nn.Softplus(beta=args.softplus_beta), \
                        'elu': elu, 'leaky_relu': leaky_relu, 'relu_sqr': relu_sqr, 'mish': mish, \
                        'swish': swish}[args.act_pou_obs]
        
    def forward(self, x):
        y = self.act_fun(self.linear_in(x))
        for i,l in enumerate(self.linear_layers):
            y_in = y
            y = self.act_fun(l(y))
            if self.args.res_link_pou_obs:
                y = y + y_in
        y = nn.functional.softmax(self.linear_out(y), dim=-1)

        return y


class Obstacle_single(nn.Module):
    def __init__(self, d_in, args):
        super(Obstacle_single, self).__init__()
        self.args = args
        h = args.h_obs

        self.linear_in     = nn.Linear(d_in, h)
        self.linear_layers = nn.ModuleList([nn.Linear(h,h) for _ in range(args.l_obs)])
        # NOTE: for a vanilla MLP, this bias will not accumulate any gradients here per our algorithm
        self.linear_out    = nn.Linear(h, 1, bias=args.last_bias_obs) 
        
        self.act_fun = {'relu': relu, 'tanh': torch.tanh, 'softplus':nn.Softplus(beta=args.softplus_beta), \
                        'elu': elu, 'leaky_relu': leaky_relu, 'relu_sqr': relu_sqr, 'mish': mish, \
                        'swish': swish}[args.act_obs]
        
        if args.bn_obs:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(h) for i in range(args.l_obs)])

    def forward(self, x_init):
        x = self.act_fun(self.linear_in(x_init))
        for i,l in enumerate(self.linear_layers):
            x_in = x
            x = self.act_fun(l(x))
            if self.args.bn_obs:
                bn = self.bn_layers[i]
                x = bn(x)
            if self.args.res_link_obs:
                x = x + x_in
        out = self.linear_out(x)

        if self.args.act_obs_out == 'exp':
            out = torch.exp(out)
        elif self.args.act_obs_out == 'relu':
            out = relu(out)
        elif self.args.act_obs_out == 'sqr':
            out = out**2
        elif self.args.act_obs_out == 'sigmoid':
            out = torch.sigmoid(out)

        return out

class Obstacle(nn.Module):
    def __init__(self, d_in, args):
        super(Obstacle, self).__init__()
        self.args = args
        if args.pou_obs:
            self.obs_nn = nn.ModuleList([Obstacle_single(d_in, args) \
                                            for _ in range(args.n_pou_obs)])
            self.POU = POU(d_in, args)
        else:
            self.obs_nn = Obstacle_single(d_in, args)

    def forward(self, x):
        if self.args.pou_obs:
            # POU
            out = torch.cat([N(x) for N in self.obs_nn], dim=-1)
            psi = self.POU(x)
            # apply POU on the previous NN output
            out = torch.sum(out*psi, dim=-1, keepdim=True)
        else:
            # map through obstacle NN to get its value at x
            out = self.obs_nn(x)
        
        return out


class Obstacle_indicator(nn.Module):
    def __init__(self, args):
        super(Obstacle_indicator, self).__init__()
        self.setup(args)

    def setup(self, args):
        if args.dataset_name == 'crowd_motion_gaussian_nonsmooth_obs':
            self.x_min = -3
            self.x_max = 3
            self.y_min = -1.5
            self.y_max = 1.5
            self.obs_val = args.obs_nonsmooth_val

            low  = torch.Tensor([self.x_min, self.y_min]).cuda()
            high = torch.Tensor([self.x_max, self.y_max]).cuda()
            self.sampler = D.Uniform(low, high)
        else:
            raise NotImplementedError()
    
    def forward(self, x):
        # x: N x d
        N = x.shape[0]
        I = (x[:, 0] >= self.x_min) * (x[:, 0] <= self.x_max) * (x[:, 1] >= self.y_min) * (x[:, 1] <= self.y_max)
        out = torch.zeros(N, 1).to(x.device)
        out[I] = self.obs_val

        return out

    def sample(self, n):
        return self.sampler.sample(n)


class Obstacle_robot(nn.Module):
    def __init__(self, d_in, h, args):
        super(Obstacle_robot, self).__init__()
        self.linear_in     = nn.Linear(d_in, h)
        self.linear_layers = nn.ModuleList([nn.Linear(h,h) for i in range(args.robot_1_obs_l)])
        self.linear_out    = nn.Linear(h, 1)
        if args.robot_1_obs_act == 'relu':
            self.act_fun = nn.ReLU()
        elif args.robot_1_obs_act == 'tanh':
            self.act_fun = nn.Tanh()
        
        self.args = args
    
    def forward(self, x):
        x = self.act_fun(self.linear_in(x))
        for l in self.linear_layers:
            x = self.act_fun(l(x)) + x

        out = self.linear_out(x)

        if self.args.obs_robot_sig:
            out = torch.sigmoid(out)
        else:
            out = torch.exp(out)

        return out


class NF_Obstacle(nn.Module):
    def __init__(self, NF, obstacle, args=None):
        super().__init__()
        self.args  = args
        self.NF = NF
        self.obstacle = obstacle
        self.param_len_flow = len(list(NF.parameters()))
