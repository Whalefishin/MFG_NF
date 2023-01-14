'''
Adapted From: https://github.com/vis-opt-group/IAPTT-GM/blob/main/experiment/Numerical.py
Author: Risheng Liu, Yaohua Liu, Shangzhi Zeng, Jin Zhang.
'''


import argparse
import time
import os, sys

# sys.path.append('../')
import numpy as np
import torch
import higher
import csv
from mfp_utils import set_grad
import copy
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser(description='Data HyperCleaner')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--load', default=False)
parser.add_argument('--dim', type=int, default=1, help="dimension")
parser.add_argument('--outer_loop', type=int, default=500, help="outer loop")
parser.add_argument('--inner_loop', type=int, default=40, help="inner loop")
parser.add_argument('--K', type=int, default=60, help="for CG")
parser.add_argument('--learning_rate', type=float, default=0.0005, help="learning rate for inner training steps")
parser.add_argument('--meta_learning_rate', type=float, default=0.1, help="learning rate for outer training steps")
parser.add_argument('--min_learning_rate', type=float, default=0.01, help="learning rate for inner training steps")
parser.add_argument('--x0', type=float, default=1, help="init x")
parser.add_argument('--y0', type=float, default=2, help="init y")

parser.add_argument('--u', type=float, default=0.4, help="ratio between inner and outer objectives")
parser.add_argument('--reg_param', type=float, default=0.25, help="coefficient of regularization part")
parser.add_argument('--exp_param', type=float, default=0.5, help="coefficient of regularization part")
parser.add_argument('--epi_param', type=float, default=0.1, help="coefficient of regularization part")
parser.add_argument('--regularization', type=str, default=None, help="distance_norm or p_norm")

parser.add_argument('--Notes', type=str, default='new_time_test', metavar='N', help='Additional Notes')
args = parser.parse_args()

cuda = False
double_precision = True
grad_clip = False
grad_div = False
grad_clip_val = 1e-3

default_tensor_str = 'torch.cuda' if cuda else 'torch'
default_tensor_str += '.DoubleTensor' if double_precision else '.FloatTensor'
torch.set_default_tensor_type(default_tensor_str)

cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

train_losses = []
val_losses = []
inner_losses = []

def grad_clip_callback(grads):
    max_norm = grad_clip_val
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        if g is not None:
            g.data = g.data * clip_coef_clamped

    return grads


def _grad_div(params, grad_div_val):
    max_norm = grad_div_val
    total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in params if p.grad is not None]))
    clip_coef = max_norm / (total_norm + 1e-6)
    for p in params:
        if p.grad.data is not None:
            p.grad.data = p.grad.data * clip_coef


def grad_div_callback(grads):
    max_norm = grad_clip_val
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
    clip_coef = max_norm / (total_norm + 1e-6)
    # clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        if g is not None:
            g.data = g.data * clip_coef

    return grads

def lim(x,min,max):
    return torch.min(torch.max(x,torch.ones_like(x)*min),torch.ones_like(x)*max)

class Toy_x(torch.nn.Module):
    def __init__(self, d):
        super(Toy_x, self).__init__()
        self.toy_x = torch.nn.Parameter(args.x0 * torch.ones(d).requires_grad_(True))
        self.e = torch.ones(d)

    def forward(self, y, x=None):
        if x is None:
            return self.toy_x + self.toy_x * y
        else:
            return  x + x * y

class Toy_y(torch.nn.Module):
    def __init__(self, d):
        super(Toy_y, self).__init__()
        self.toy_y = torch.nn.Parameter(args.y0 * torch.ones(d).requires_grad_(True))

    def forward(self, x, y=None):
        if y is None:
            return -torch.sin(x *self.toy_y)

        else:
            return -torch.sin(x * y)


x_solu = torch.ones(args.dim) * 5.5*np.pi/2
y_solu = -2*torch.ones(args.dim)


# def grad_FD(model_x, model_y, loss_outer_orig, loss_inner_orig, d=1e-5):
#     g_obs_FD = []
    
#     for k,p in enumerate(model_x.parameters()):
#         size = p.numel()
#         g = torch.zeros_like(p)
#         for j in range(size):
#             e = torch.zeros_like(p).flatten()
#             e[j] = 1.

#             model_x_copy  = copy.deepcopy(model_x)
#             model_y_copy  = copy.deepcopy(model_y)
#             model_y_copy2 = copy.deepcopy(model_y)
#             optimizer_copy = torch.optim.Adam(model_y_copy.parameters(), lr=args.learning_rate)
#             optimizer_copy2 = torch.optim.Adam(model_y_copy2.parameters(), lr=args.learning_rate)

#             loss_outer_base = 0
#             loss_outer_perturbed = 0

#             for _ in range(args.inner_loop):
#                 model_y_copy.zero_grad()
#                 loss_inner = model_y_copy(model_x_copy.toy_x)
#                 loss_inner.backward()
#                 # if grad_clip:
#                 #     clip_grad_norm_(model_y_copy.parameters(), grad_clip_val)
#                 # elif grad_div:
#                 #     grad_div(model_y_copy.parameters(), grad_clip_val)
#                 optimizer_copy.step()
#                 model_y_copy.toy_y.data = lim(model_y_copy.toy_y.data, -2, 2)
#                 loss_outer_base = model_x_copy(model_y_copy.toy_y)

#             p_copy = list(model_x_copy.parameters())[k]
#             p_copy.data = p_copy.data + d * e.reshape(p_copy.shape)
            
#             for _ in range(args.inner_loop):
#                 model_y_copy2.zero_grad()
#                 loss_inner2 = model_y_copy2(model_x_copy.toy_x)
#                 loss_inner2.backward()
#                 # if grad_clip:
#                 #     clip_grad_norm_(model_y_copy2.parameters(), grad_clip_val)
#                 # elif grad_div:
#                 #     grad_div(model_y_copy2.parameters(), grad_clip_val)
#                 optimizer_copy2.step()
#                 model_y_copy2.toy_y.data = lim(model_y_copy2.toy_y.data, -2, 2)
#                 loss_outer_perturbed = model_x_copy(model_y_copy2.toy_y)

#             g = g.flatten()
#             g[j] = (loss_outer_perturbed - loss_outer_base) / d
        
#         g_obs_FD.append(g.reshape(p.shape))

#     return g_obs_FD



def grad_FD(model_x, model_y, loss_outer_orig, loss_inner_orig, d=1e-5):
    g_obs_FD = []
    
    for k,p in enumerate(model_x.parameters()):
        size = p.numel()
        g = torch.zeros_like(p)
        for j in range(size):
            e = torch.zeros_like(p).flatten()
            e[j] = 1.

            model_x_copy  = copy.deepcopy(model_x)
            model_y_copy  = copy.deepcopy(model_y)
            model_y_copy2 = copy.deepcopy(model_y)
            optimizer_copy = torch.optim.Adam(model_y_copy.parameters(), lr=args.learning_rate)
            optimizer_copy2 = torch.optim.Adam(model_y_copy2.parameters(), lr=args.learning_rate)

            loss_outer_base = 0
            loss_outer_perturbed = 0

            with higher.innerloop_ctx(model_y_copy, optimizer_copy, copy_initial_weights=False) as (fmodel_1, f_opt_1):
                for _ in range(args.inner_loop):
                    loss_inner = fmodel_1(model_x_copy.toy_x)
                    if grad_clip:
                        f_opt_1.step(loss_inner, grad_callback=grad_clip_callback)
                    elif grad_div:
                        f_opt_1.step(loss_inner, grad_callback=grad_div_callback)
                    else:
                        f_opt_1.step(loss_inner)
                    fmodel_1.toy_y.data = lim(fmodel_1.toy_y.data, -2, 2)
                    loss_outer_base = model_x_copy(fmodel_1.toy_y) # should be close to loss_outer_orig on the last iter.

            p_copy = list(model_x_copy.parameters())[k]
            p_copy.data = p_copy.data + d * e.reshape(p_copy.shape)
            
            with higher.innerloop_ctx(model_y_copy2, optimizer_copy2, copy_initial_weights=False) as (fmodel_2, f_opt_2):
                for _ in range(args.inner_loop):
                    loss_inner2 = fmodel_2(model_x_copy.toy_x)
                    if grad_clip:
                        f_opt_2.step(loss_inner2, grad_callback=grad_clip_callback)
                    elif grad_div:
                        f_opt_2.step(loss_inner2, grad_callback=grad_div_callback)
                    else:
                        f_opt_2.step(loss_inner2)
                    fmodel_2.toy_y.data = lim(fmodel_2.toy_y.data, -2, 2)
                    loss_outer_perturbed = model_x_copy(fmodel_2.toy_y)

            g = g.flatten()
            g[j] = (loss_outer_perturbed - loss_outer_base) / d
        
        g_obs_FD.append(g.reshape(p.shape))

    return g_obs_FD


def main():
    # log_path = "toy_3_proj_lor_outerloop{}_innerloop{}_dim{}_inner_lr{}_outer_lr{}_x_init{:.1f}_y_init{:.1f}_Notes{}.csv".format(
    #     args.outer_loop, args.inner_loop, args.dim, args.learning_rate, args.meta_learning_rate, args.x0,
    #     args.y0,args.Notes)
    #     #time.strftime("%Y_%m_%d_%H_%M_%S"))
    # with open(log_path, 'a', encoding='utf-8', newline='') as f:
        # csv_writer = csv.writer(f)
        # #csv_writer.writerow([args])
        # csv_writer.writerow(
        #     ['Meta_iter', 'Inner_loss', 'Outer_loss', 'res_norm_y', 'res_norm_x', 'total_time', 'hyper_time',
        #      'lower_time', 'i_max', 'y new', 'y final', 'x'])
    model_x = Toy_x(args.dim)
    model_y = Toy_y(args.dim)

    x_opt = torch.optim.SGD(model_x.parameters(), lr=args.meta_learning_rate)
    y_opt = torch.optim.Adam(model_y.parameters(), lr=args.learning_rate)
    y_init_opt = torch.optim.SGD(model_y.parameters(), lr=args.meta_learning_rate)
    x_lr_schedular = torch.optim.lr_scheduler.StepLR(x_opt, 50, gamma=0.7, last_epoch=-1)

    forward_time, backward_time = 0,0

    for meta_iter in range(args.outer_loop):
        x_opt.zero_grad()
        y_opt.zero_grad() # same as y_init_opt.zero_grad()

        F_list = []
        f_list = []
        
        i_max = args.inner_loop
        start_time_task = time.time()
        F_max = -np.inf

        with higher.innerloop_ctx(model_y, y_opt, copy_initial_weights=False) as (fmodel, f_opt):
            forward_time_task = time.time()
            for _ in range(args.inner_loop):
                loss_inner = fmodel(model_x.toy_x)

                f_list.append(loss_inner.detach().cpu().numpy())
                if grad_clip:
                    f_opt.step(loss_inner, grad_callback=grad_clip_callback)
                elif grad_div:
                    f_opt.step(loss_inner, grad_callback=grad_div_callback)
                else:
                    f_opt.step(loss_inner)

                fmodel.toy_y = lim(fmodel.toy_y,-2,2)

                loss_outer = model_x(fmodel.toy_y)

                F_list.append(loss_outer.detach())

                if F_max < loss_outer:
                    F_max = loss_outer

            # logging
            forward_time_task = time.time() - forward_time_task
            forward_time += forward_time_task
            backward_time_task = time.time()

            # PTT
            i_max = F_list.index(max(F_list))
            params = fmodel.parameters(time=i_max+1)
            y_new = next(params)

            # logging
            y_new_log = y_new.detach()
            y_final_log = next(fmodel.parameters(args.inner_loop)).detach()
            x_log = model_x.parameters()
            x_log = next(x_log).detach()

            # UL gradient
            # F_loss = F_max
            F_loss = loss_outer
            grad_y_init = torch.autograd.grad(F_loss, fmodel.parameters(time=0), retain_graph=True,
                                              allow_unused=True)
            grad_x_init = torch.autograd.grad(F_loss, model_x.parameters(), retain_graph=True)


        # Use FD to check gradients
        d_list = [10**(-i) for i in range(6,7)]
        diff_list = []
        rel_diff_list = []
        g_obs_FD_list = []
        for d in d_list:
            g_obs_FD = grad_FD(model_x, model_y, loss_outer, loss_inner, d=d)

            norm_AD = torch.sqrt(sum([torch.sum(g**2) for g in grad_x_init])) 
            g_diff = [g_AD - g_FD for (g_AD, g_FD) in zip(grad_x_init, g_obs_FD)]
            norm_diff = torch.sqrt(sum([torch.sum(g**2) for g in g_diff]))
            norm_rel_diff = norm_diff / norm_AD
            diff_list.append(float(norm_diff))
            rel_diff_list.append(float(norm_rel_diff))
            g_obs_FD_list.append(g_obs_FD)

        print ("Delta: {}".format(d_list))
        print ("Norm diff: {}".format(diff_list))
        print ("Rel Norm diff: {}".format(rel_diff_list))
        print ("grad from FD: {}".format(g_obs_FD_list))
        print ("grad from AD: {}".format(grad_x_init))


        model_x = set_grad(model_x, g_obs_FD_list[-1])
        model_y = set_grad(model_y, grad_y_init)

        # update outer loop
        y_init_opt.step()
        x_opt.step()
        # x_lr_schedular.step()

        # logging
        backward_time_task = time.time() - backward_time_task
        backward_time += backward_time_task
        total_time_iter = time.time() - start_time_task

        # projection, delete this
        with torch.no_grad():
            for x in model_x.parameters():
                x.clamp_(1, 10)
            for y in model_y.parameters():
                y.clamp_(-2, 2)

        # logging
        print("Meta_Iter{},Inner_loss{}, Outer_loss{},x{},y{},forward_time{},backward_time{}"
              .format(meta_iter, model_y(x_log, y_final_log).detach().numpy(), model_x(y_final_log, x_log).detach().numpy(), x_log.data.detach().numpy(),
                      # model_x.toy_x.data,
                      y_final_log.data.detach().numpy(),
                      forward_time,
                      backward_time))
        print('learning_rate: {} \n'.format(x_lr_schedular.get_last_lr()))


if __name__ == '__main__':
    main()
