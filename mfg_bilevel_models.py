import torch
import torch.nn as nn
import numpy as np
import torch.distributions as D
from torch.nn.functional import relu, softplus, elu, leaky_relu






def construct_Q(args, device):
    if args.dataset_name in ['crowd_motion_gaussian', 'crowd_motion_gaussian_bilevel', \
                            'crowd_motion_gaussian_bilevel_strong']:
        mean = torch.zeros(2).cuda(device)
        diag = torch.Tensor([1., 0.5]).cuda(device)
        cov  = torch.diag_embed(diag)
        Q = D.MultivariateNormal(mean, cov)
    elif args.dataset_name == 'crowd_motion_gaussian_nonsmooth_obs':
        Q = Obstacle_indicator(args).to(device)
    elif args.dataset_name in ['crowd_motion_two_bars', 'crowd_motion_two_bars_bilevel',\
                               'crowd_motion_two_bars_uniform', 'crowd_motion_two_bars_uniform_bilevel',
                               'crowd_motion_two_bars_gaussian']:
        # coordinates for the corners of the bars
        points = [(-5, 0.9, 1.15, 1.35), (1.1, 7, 0.65, 0.85)]
        # points = [(-2, 0.5, 1.1, 1.3), (0.5, 3, 0.7, 0.9)]
        Q = Obstacle_cuboid(points, args.two_bars_sharpness, args.two_bars_height)
    elif args.dataset_name in ['crowd_motion_gaussian_two_bars', 'crowd_motion_gaussian_two_bars_uniform',\
                            'crowd_motion_gaussian_two_bars_uniform_bilevel', 'crowd_motion_gaussian_two_bars_gaussian',
                            'crowd_motion_gaussian_two_bars_gaussian_bilevel']:
        num_gaussians = 2
        weight = D.Categorical(torch.ones(num_gaussians,).to(device))
        mean = torch.tensor([[-2, 1.25], [4, 0.75]]).to(device) # N_m x d
        cov  = torch.tensor([[1.5,0], [0, 1e-2]]).unsqueeze(0).repeat(num_gaussians,1,1).to(device)
        dist = D.MultivariateNormal(mean.to(device), cov)
        # mixture = D.MixtureSameFamily(weight, dist)
        # Q = distributions.Mixture((dim,), mixture)
        Q = D.MixtureSameFamily(weight, dist)
    elif args.dataset_name in ['crowd_motion_gaussian_one_bar_uniform']:
        mean = torch.tensor([-2, 1.25]).to(device)
        diag = torch.Tensor([1.5, 1e-2]).to(device)
        cov  = torch.diag_embed(diag)
        Q = D.MultivariateNormal(mean, cov)
    elif args.dataset_name == 'crowd_motion_flower':
        horizontal_pedals = 2
        vertical_pedals   = 2
        num_gaussians = horizontal_pedals + vertical_pedals
        weight = D.Categorical(torch.ones(num_gaussians,).to(device))

        # mean = torch.tensor([[-1.2, 0.], [1.2, 0.], [0., 1.2], [0., -1.2]]).to(device) # N_m x d
        # cov_ver  = torch.tensor([[0.2,0], [0, 0.5]]).unsqueeze(0).repeat(horizontal_pedals,1,1).to(device)
        # cov_hor  = torch.tensor([[0.5, 0], [0, 0.2]]).unsqueeze(0).repeat(vertical_pedals,1,1).to(device)

        # mean = torch.tensor([[-1, 0.], [1, 0.], [0., 1], [0., -1]]).to(device) # N_m x d
        # cov_ver  = torch.tensor([[0.2,0], [0, 0.3]]).unsqueeze(0).repeat(horizontal_pedals,1,1).to(device)
        # cov_hor  = torch.tensor([[0.3, 0], [0, 0.2]]).unsqueeze(0).repeat(vertical_pedals,1,1).to(device)

        mean = torch.tensor([[-1, 0.], [1, 0.], [0., 1], [0., -1]]).to(device) # N_m x d
        cov_ver  = torch.tensor([[0.2,0], [0, 0.4]]).unsqueeze(0).repeat(horizontal_pedals,1,1).to(device)
        cov_hor  = torch.tensor([[0.4, 0], [0, 0.2]]).unsqueeze(0).repeat(vertical_pedals,1,1).to(device)
        theta = np.pi / 4
        R = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).to(device).float()
        cov = torch.cat((cov_hor, cov_ver))
        
        # rotate the flower
        mean = mean @ R.transpose(0,1)
        a = torch.bmm(R.unsqueeze(0).repeat(num_gaussians,1,1), cov)
        cov = torch.bmm(a, R.transpose(0,1).unsqueeze(0).repeat(num_gaussians,1,1))

        dist = D.MultivariateNormal(mean.to(device), cov)
        Q = D.MixtureSameFamily(weight, dist)

        # mean = torch.zeros(2).cuda(device)
        # diag = torch.Tensor([1., 0.5]).cuda(device)
        # a  = torch.diag_embed(diag)
        # cov  = R @ a @ R.T
        # Q = D.MultivariateNormal(mean, cov)

    elif args.dataset_name == 'crowd_motion_gaussian_close':
        e_2    = torch.zeros(2).to(device)
        e_2[1] = 1.
        mean = 0.5 * e_2
        diag = 0.02 * torch.Tensor([1., 0.5]).cuda(device)
        cov  = torch.diag_embed(diag)
        Q = D.MultivariateNormal(mean, cov)
    elif args.dataset_name == 'crowd_motion_gaussian_NN_obs':
        Q = Obstacle(args.gaussian_multi_dim, args).to(device)
        pretrain_obs_dir = './results/crowd_motion_gaussian_bilevel/pretrain_obs.t'
        Q.load_state_dict(torch.load(pretrain_obs_dir))
        print ("Loaded obstacle from: {}".format(pretrain_obs_dir))
        # disable training for the obstacle
        for p in Q.parameters():
            p.requires_grad = False
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
        # if self.args.dataset_name in ['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong']:
        #     mean = torch.zeros(2).to(self.device)
        #     diag = torch.Tensor([1., 0.5]).to(self.device)
        #     cov  = torch.diag_embed(diag)
        #     self.B = D.MultivariateNormal(mean, cov)
        # elif self.args.dataset_name == 'crowd_motion_two_bars_bilevel':
        #     points = [(-5, 0.9, 1.15, 1.35), (1.1, 7, 0.65, 0.85)]
        #     self.B = Obstacle_cuboid(points, self.args.two_bars_sharpness)
        # else:
        #     raise NotImplementedError()
        self.B = construct_Q(self.args, self.device)

    def eval(self, x):
        if x.shape[0] == 0:
            return torch.zeros(0).to(x.device)

        # if self.args.dataset_name in ['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong',
        #                               'crowd_motion_gaussian_two_bars_uniform_bilevel']:
        #     return 50 * torch.exp(self.B.log_prob(x[..., :2]))
        # elif self.args.dataset_name in ['crowd_motion_two_bars_bilevel', 'crowd_motion_two_bars_uniform_bilevel']:
        #     return self.B(x[..., :2])
        # else:
        #     raise NotImplementedError()

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


class Obstacle_cuboid():
    def __init__(self, points, a, h):
        super(Obstacle_cuboid, self).__init__()
        self.cuboids = []
        self.h = h
        for p in points:
            x_1, x_2, y_1, y_2 = p
            self.cuboids.append(Obstacle_single_cuboid(x_1, x_2, y_1, y_2, a=a))
    
    def __call__(self, p):
        val = 0
        for cuboid in self.cuboids:
            val += cuboid.eval(p)
        
        return self.h * val

class Obstacle_single_cuboid():
    def __init__(self, x_1, x_2, y_1, y_2, a=10):
        super(Obstacle_single_cuboid, self).__init__()
        self.x_1 = x_1
        self.x_2 = x_2
        self.y_1 = y_1
        self.y_2 = y_2
        self.a   = a

    def smooth_cuboid(self, x,y):
        return sigmoid_2D(x, y, self.a, self.x_1, self.y_1) - sigmoid_2D(x, y, self.a, self.x_1, self.y_2) \
             - sigmoid_2D(x, y, self.a, self.x_2, self.y_1) + sigmoid_2D(x, y, self.a, self.x_2, self.y_2)

    def eval(self, p):
        x = p[..., 0] # assumes the last axis to be the spatial dimensions
        y = p[..., 1]
        return self.smooth_cuboid(x,y)



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
            # # n networks, each representing an obstacle
            # self.linear_in     = nn.ModuleList([nn.Linear(d_in, h) for _ in range(args.n_pou_obs)])
            # self.linear_layers = nn.ModuleList([
            #                             nn.ModuleList([nn.Linear(h,h) for _ in range(args.l_obs)]) \
            #                         for i in range(args.n_pou_obs)])
            # self.linear_out    = nn.ModuleList([nn.Linear(h, args.n_pou_obs) for _ in range(args.n_pou_obs)])
            # # parametrization for the partition of unity
            # self.linear_in_pou = nn.Linear(d_in, h_pou)
            # self.linear_layers_pou = nn.ModuleList([nn.Linear(h_pou, h_pou) for _ in range(args.l_pou_obs)])
            # self.linear_out_pou = nn.Linear(h_pou, args.n_pou_obs) 
            self.obs_nn = nn.ModuleList([Obstacle_single(d_in, args) \
                                            for _ in range(args.n_pou_obs)])
            self.POU = POU(d_in, args)
        else:
            # self.linear_in     = nn.Linear(d_in, h)
            # self.linear_layers = nn.ModuleList([nn.Linear(h,h) for _ in range(args.l_obs)])
            # # NOTE: this bias will not accumulate any gradients here per our algorithm
            # self.linear_out = nn.Linear(h, 1, bias=False)
            self.obs_nn = Obstacle_single(d_in, args)

    def forward(self, x):
        # # map through obstacle NN to get its value at x
        # out = self.forward_NN(x)
        
        # # optional partition of unity if enabled
        # if self.args.pou_obs:
        #     psi = self.forward_POU(x)
        #     # apply POU on the previous NN output
        #     out = torch.sum(out*psi, dim=-1, keepdim=True)

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

    # def forward(self, x):
    #     x_initial = x
    #     x = self.act_fun(self.linear_in(x))
    #     for i,l in enumerate(self.linear_layers):
    #         x_in = x
    #         x = self.act_fun(l(x))
    #         if self.args.bn_obs:
    #             bn = self.bn_layers[i]
    #             x = bn(x)
    #         if self.args.res_link_obs:
    #             x = x + x_in
    #     x = self.linear_out(x)

    #     if self.args.act_obs_out == 'exp':
    #         out = torch.exp(x)
    #     elif self.args.act_obs_out == 'relu':
    #         out = relu(x)
    #     elif self.args.act_obs_out == 'sqr':
    #         out = x**2
    #     elif self.args.act_obs_out == 'sigmoid':
    #         out = torch.sigmoid(x)
        
    #     # optional partition of unity if enabled
    #     if self.args.pou_obs:
    #         y = self.act_fun_pou(self.linear_in_pou(x_initial))
    #         for i,l in enumerate(self.linear_layers_pou):
    #             y_in = y
    #             y = self.act_fun_pou(l(y))
    #             if self.args.res_link_pou_obs:
    #                 y = y + y_in
    #         y = nn.functional.softmax(self.linear_out_pou(y), dim=-1)

    #         # apply POU on the previous NN output
    #         out = torch.sum(out*y, dim=-1, keepdim=True)
        
    #     return out

    # def forward_NN(self, x_init):
    #     if self.args.pou_obs:
    #         for i in range(self.args.n_pou_obs):
    #             x = self.act_fun(self.linear_in(x_init))
    #             for i,l in enumerate(self.linear_layers):
    #                 x_in = x
    #                 x = self.act_fun(l(x))
    #                 if self.args.bn_obs:
    #                     bn = self.bn_layers[i]
    #                     x = bn(x)
    #                 if self.args.res_link_obs:
    #                     x = x + x_in
    #             x = self.linear_out(x)
    #     else:
    #         x = self.act_fun(self.linear_in(x_init))
    #         for i,l in enumerate(self.linear_layers):
    #             x_in = x
    #             x = self.act_fun(l(x))
    #             if self.args.bn_obs:
    #                 bn = self.bn_layers[i]
    #                 x = bn(x)
    #             if self.args.res_link_obs:
    #                 x = x + x_in
    #         x = self.linear_out(x)

    #     if self.args.act_obs_out == 'exp':
    #         out = torch.exp(x)
    #     elif self.args.act_obs_out == 'relu':
    #         out = relu(x)
    #     elif self.args.act_obs_out == 'sqr':
    #         out = x**2
    #     elif self.args.act_obs_out == 'sigmoid':
    #         out = torch.sigmoid(x)

    #     return out

    # def forward_POU(self, x):
    #     y = self.act_fun_pou(self.linear_in_pou(x))
    #     for i,l in enumerate(self.linear_layers_pou):
    #         y_in = y
    #         y = self.act_fun_pou(l(y))
    #         if self.args.res_link_pou_obs:
    #             y = y + y_in
    #     y = nn.functional.softmax(self.linear_out_pou(y), dim=-1)

    #     return y


class Obstacle_gaussian(nn.Module):
    def __init__(self, init='rand'):
        super(Obstacle_gaussian, self).__init__()
        if init == 'true':
            self.mean = nn.Parameter(torch.zeros(2))
            # self.diag = nn.Parameter(torch.Tensor([1., 0.5]))
        elif init == 'rand':
            self.mean = nn.Parameter(torch.rand(2))
            # self.diag = nn.Parameter(torch.rand(2))
        elif init == 'close':
            self.mean = nn.Parameter(torch.zeros(2) + 0.1 * torch.rand(2))
            # self.diag = nn.Parameter(torch.Tensor([1., 0.5]) + 0.001 * torch.rand(2))
        else:
            raise NotImplementedError()
        # self.B    = D.MultivariateNormal(self.mean, torch.diag_embed(self.diag))   

    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros(0).to(x.device)

        B = D.MultivariateNormal(self.mean, torch.diag_embed(torch.Tensor([1., 0.5]).to(x.device))) 
        return 50 * torch.exp(B.log_prob(x[..., :2]))
        

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
