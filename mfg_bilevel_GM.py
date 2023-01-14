import argparse
import json
from matplotlib.pyplot import xlim
import numpy as np
import torch
import os, sys, shutil, math, pylab

from tensorboardX import SummaryWriter
from time import sleep
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm
import torch.distributions as D
from torch.autograd.functional import vhp, vjp, jacobian
from torch.autograd import grad, gradcheck

import data as data_
import nn as nn_
import utils
from mfp_utils import *

from experiments import cutils
from nde import distributions, flows, transforms
from mfg_bilevel_models import *
import copy
import higher
from NF_iterative import *
from arguments import parse_arguments



args = parse_arguments()

# =================================================================================== #
#                                       Meta                                          #
# =================================================================================== #

os.environ['DATAROOT']     = 'experiments/dataset/data/'
os.environ['SLURM_JOB_ID'] = '1'

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

assert torch.cuda.is_available()
device = torch.device('cuda')
device_cpu = torch.device('cpu')

if args.debug_tensor_type == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)

# data loading 
if args.dataset_name == 'crowd_motion_gaussian_bilevel':
    if args.gaussian_multi_dim == 2:
        if args.pretrain_obs_dir == '':
            args.pretrain_obs_dir = './results/crowd_motion_gaussian_bilevel/pretrain_obs.t'
        if args.pretrain_NF_dir == '':
            args.pretrain_NF_dir = 'results/crowd_motion_gaussian_bilevel/NF_UL_pretrain_30k.t'
        args.bilevel_training_data_dir = './results/crowd_motion_gaussian/NSF_CL_crowd_motion_2D_Jeff_OT_1e-1_F_2e-1_B=2048_lr=1e-3_N=50k_FD4Simp/train_traj_crowd_motion_gaussian.pt'
    else:
        if args.pretrain_obs_dir == '':
            args.pretrain_obs_dir = './results/crowd_motion_gaussian_bilevel/pretrain_obs_{}D.t'.format(args.gaussian_multi_dim)
        if args.pretrain_NF_dir == '':
            args.pretrain_NF_dir = './results/crowd_motion_gaussian_bilevel/NF_UL_pretrain_{}D_30k.t'.format(args.gaussian_multi_dim)
        args.bilevel_training_data_dir = './results/crowd_motion_gaussian/{}D_iterative/train_traj_crowd_motion_gaussian.pt'.format(args.gaussian_multi_dim)
elif args.dataset_name == 'crowd_motion_gaussian_bilevel_strong':
    if args.pretrain_obs_dir == '':
        args.pretrain_obs_dir = './results/crowd_motion_gaussian_bilevel/pretrain_obs.t'
    if args.pretrain_NF_dir == '':
        args.pretrain_NF_dir = './results/crowd_motion_gaussian/NSF_CL_2D_N=1M_OT_1e-1_F_1e0/crowd_motion_gaussian-best-val-NSF_CL_2D_N=1M_OT_1e-1_F_1e0.t'
    args.bilevel_training_data_dir = './results/crowd_motion_gaussian/NSF_CL_2D_N=1M_OT_1e-1_F_1e0/train_traj_crowd_motion_gaussian.pt'
elif args.dataset_name == 'crowd_motion_two_bars_bilevel':
    if args.pretrain_obs_dir == '':
        args.pretrain_obs_dir = './results/crowd_motion_two_bars_bilevel/pretrain_obs.t'
    if args.pretrain_NF_dir == '':
        args.pretrain_NF_dir = './results/crowd_motion_two_bars_bilevel/NF_UL_pretrain_30k.t'
        # args.pretrain_NF_dir = './results/crowd_motion_two_bars/F=1e1/crowd_motion_two_bars-best-val-plot.t'
    args.bilevel_training_data_dir = './results/crowd_motion_two_bars/F=1e1/train_traj_crowd_motion_two_bars.pt'
elif args.dataset_name == 'crowd_motion_two_bars_uniform_bilevel':
    if math.isclose(args.two_bars_height, 50.):
        if args.pretrain_obs_dir == '':
            args.pretrain_obs_dir = './results/crowd_motion_two_bars_uniform_bilevel/pretrain_obs.t'
        if args.pretrain_NF_dir == '':
            args.pretrain_NF_dir = './results/crowd_motion_two_bars_uniform_bilevel/NF_UL_pretrain_h=50_30k.t'
            # args.pretrain_NF_dir = './results/crowd_motion_two_bars_uniform_bilevel/NF_UL_pretrain_30k.t'
        # args.bilevel_training_data_dir = './results/crowd_motion_two_bars_uniform/F=1e1/train_traj_crowd_motion_two_bars_uniform.pt'
        args.bilevel_training_data_dir = './results/crowd_motion_two_bars_uniform/F=2e-1_h=50/train_traj_crowd_motion_two_bars_uniform.pt'
    else:
        if args.pretrain_obs_dir == '':
            args.pretrain_obs_dir = './results/crowd_motion_two_bars_uniform_bilevel/pretrain_obs.t'
        if args.pretrain_NF_dir == '':
            args.pretrain_NF_dir = './results/crowd_motion_two_bars_uniform_bilevel/NF_UL_pretrain_30k.t'
        args.bilevel_training_data_dir = './results/crowd_motion_two_bars_uniform/F=1e1/train_traj_crowd_motion_two_bars_uniform.pt'
elif args.dataset_name == 'crowd_motion_gaussian_two_bars_uniform_bilevel':
        if args.pretrain_obs_dir == '':
            args.pretrain_obs_dir = './results/crowd_motion_gaussian_two_bars_uniform/F=2e-1/pretrain_obs.t'
        if args.pretrain_NF_dir == '':
            args.pretrain_NF_dir = './results/crowd_motion_gaussian_two_bars_uniform/F=2e-1/NF_UL_pretrain.t'
        if args.bilevel_training_data_dir == '':
            args.bilevel_training_data_dir = './results/crowd_motion_gaussian_two_bars_uniform/F=2e-1/train_traj_crowd_motion_gaussian_two_bars_uniform.pt'
elif args.dataset_name == 'crowd_motion_gaussian_two_bars_gaussian_bilevel':
        if args.pretrain_obs_dir == '':
            args.pretrain_obs_dir = './results/crowd_motion_gaussian_two_bars_gaussian/F=2e-1_noClip/pretrain_obs.t'
        if args.pretrain_NF_dir == '':
            args.pretrain_NF_dir = './results/crowd_motion_gaussian_two_bars_gaussian/F=2e-1_noClip/NF_UL_pretrain.t'
        if args.bilevel_training_data_dir == '':
            args.bilevel_training_data_dir = './results/crowd_motion_gaussian_two_bars_gaussian/F=2e-1_noClip/train_traj_crowd_motion_gaussian_two_bars_gaussian.pt'
else:
    raise NotImplementedError()
    
X_train, _, _, train_loader, val_loader, test_loader, P_1 = load_bilevel_data(args, args.num_train_data, args.num_test_data, \
    args.gaussian_multi_dim, args.bilevel_training_data_dir, args.train_batch_size, args.val_batch_size, args.test_batch_size)
train_generator = data_.batch_generator(train_loader)
test_generator  = data_.batch_generator(test_loader)
test_batch      = next(iter(train_loader)).to(device)
features        = args.gaussian_multi_dim

# =================================================================================== #
#                                       Logging                                       #
# =================================================================================== #

# logging
timestamp = args.exp_name
log_dir   = os.path.join(cutils.get_log_root(), args.dataset_name, timestamp)

# remove previous TB records if desired
tbx_logging = True
if os.path.exists(log_dir):
    choice = input("Directory already exists. Use TBX to record again? (y/n)")
    if choice == 'y':
        choice = input("Do you wish to remove previous TB records at: " + log_dir + " ? (y/n)")
        if choice == 'y':
            shutil.rmtree(log_dir)
            sleep(5)
    else:
        tbx_logging = False
        writer      = None

while tbx_logging:
    try:
        writer = SummaryWriter(log_dir=log_dir, max_queue=20)
        writer.add_text(tag='args', text_string='CUDA_VISIBLE_DEVICES=0 python ' + " ".join(x for x in sys.argv))
        break
    except FileExistsError:
        sleep(5)
filename = os.path.join(log_dir, 'config.json')
with open(filename, 'w') as file:
    json.dump(vars(args), file)
print_args(sys.argv, save_dir=os.path.join(log_dir, 'args.txt'))

plot_data_dir = os.path.join(log_dir, 'plot_data/')
plot_dir      = os.path.join(log_dir, 'plot/')
model_dir     = os.path.join(log_dir, 'model/')
if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
plot_logger = Plot_logger(root_path=plot_data_dir)

tbar = tqdm(range(args.num_training_steps))
best_val_score = 1e10

# grid to evaluate obstacle on
grid, grid_pad, grid_x, grid_y, dx = create_grid_for_eval(args, features)
grid = grid.to(device)
grid_pad = grid_pad.to(device)


# =================================================================================== #
#                                       Model                                         #
# =================================================================================== #


# methods for chaining together the flow transformations
def create_linear_transform():
    if args.linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif args.linear_transform_type == 'lu_no_perm':
        return transforms.LULinear(features, identity_init=True)
    elif args.linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif args.linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError

def create_base_transform(i):
    if args.base_net_act == 'relu':
        act = F.relu
    elif args.base_net_act == 'tanh':
        act = F.tanh
    elif args.base_net_act == 'mish':
        act = mish
    else:
        raise NotImplementedError()

    if args.base_transform_type == 'affine-coupling':
        return transforms.AffineCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            )
        )
    elif args.base_transform_type == 'quadratic-coupling':
        return transforms.PiecewiseQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'rq-coupling':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=act,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'quadratic-autoregressive':
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=act,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    else:
        raise ValueError

def create_transform():
    if args.linear_transform_type == 'lu_no_perm' and 'coupling' in args.base_transform_type:
        flows = [
            transforms.CompositeTransform([
                create_linear_transform(),
                create_base_transform(i),
                create_base_transform(i+1)
            ]) for i in range(0, 2*args.num_flow_steps, 2)
        ]
    else:
        flows = [
            transforms.CompositeTransform([
                create_linear_transform(),
                create_base_transform(i)
            ]) for i in range(args.num_flow_steps)
        ]
        
    K = args.num_flow_steps
    if args.LU_last:
        flows += [create_linear_transform()]
        K     += 1
    transform = transforms.CompositeTransform(flows)

    return transform, K

def create_transform_iterative():
    transform = []
    for i in range(0, 2*args.num_flow_steps, 2):
        transform.append(create_linear_transform())
        transform.append(create_base_transform(i))
        transform.append(create_base_transform(i+1))

    # flows = [create_linear_transform(),
    #         create_base_transform(i),
    #         create_base_transform(i+1)
    #      for i in range(0, 2*args.num_flow_steps, 2)
    # ]

    K = args.num_flow_steps
    if args.LU_last:
        flows += [create_linear_transform()]
        K     += 1
    # transform = transforms.CompositeTransform(flows)

    return transform, K

# prior
P_0 = create_base_dist(args, features)

# flows
transform, K = create_transform_iterative()
NF = NF_iterative_flatVar(transform, P_0, K).to(device)
num_params_NF = utils.get_num_parameters(NF)
print('There are {} trainable parameters in the flow model.'.format(num_params_NF))
if args.algo in ['penalty', 'penalty_fixNF']:
    transform_hat, K = create_transform_iterative()
    NF_hat = NF_iterative_flatVar(transform_hat, P_0, K).to(device)

# obstacle
# ground truth obstacle
B_true = Obstacle_true(args, device)
# NN-parametrized obstacle
if args.debug_obs == 'NN':
    obstacle = Obstacle(features, args).to(device)
elif args.debug_obs == 'gaussian':
    obstacle = Obstacle_gaussian(init=args.debug_obs_init).to(device)
num_params_obs = utils.get_num_parameters(obstacle)
print('There are {} trainable parameters in the obstacle model.'.format(num_params_obs))

# optimizer
optimizer_dict = {'adam': optim.Adam, "sgd": optim.SGD}
optim_to_use_inner = optimizer_dict[args.optimizer_inner]

if args.optimizer_obs == 'adam':
    optimizer_obs = optim.Adam(obstacle.parameters(), lr=args.lr_obs, weight_decay=args.l2_reg_obs, \
                                    betas=(args.adam_beta1_obs, args.adam_beta2_obs))
elif args.optimizer_obs == 'sgd':
    optimizer_obs = optim.SGD(obstacle.parameters(), lr=args.lr_obs, weight_decay=args.l2_reg_obs, \
                        nesterov=args.sgd_obs_nesterov, momentum=args.sgd_obs_momentum)

optimizer_inner = optim_to_use_inner(NF.parameters(), lr=args.step_size_inner, weight_decay=args.l2_reg_inner)

optim_to_use_NF = optimizer_dict[args.optimizer_NF]
optimizer_NF    = optim_to_use_NF(NF.parameters(), lr=args.lr_NF, weight_decay=args.l2_reg_NF)

# if args.scheduler_NF == 'adaptive':
#     scheduler_NF = optim.lr_scheduler.ReduceLROnPlateau(optimizer_NF, mode='min', factor=0.5, verbose=True, patience=args.patience_NF)
# elif args.scheduler_NF == 'cyclic':
#     scheduler_NF = optim.lr_scheduler.CosineAnnealingLR(optimizer_NF, args.num_training_steps, 0)
# elif args.scheduler_NF == 'multi_step':
#     scheduler_NF = optim.lr_scheduler.MultiStepLR(optimizer_NF, milestones=args.multi_lr_milestones, gamma=args.multi_lr_gamma)

# if args.scheduler_obs == 'adaptive':
#     scheduler_obs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_obs, mode='min', factor=0.5, verbose=True, patience=args.patience_obs)
# elif args.scheduler_obs == 'step':
#     scheduler_obs = optim.lr_scheduler.StepLR(optimizer_obs, step_size=args.step_lr_size, gamma=args.step_lr_gamma)
# elif args.scheduler_obs == 'cyclic':
#     scheduler_obs = optim.lr_scheduler.CosineAnnealingLR(optimizer_obs, args.num_training_steps, args.lr_obs_min)
# elif args.scheduler_obs == 'multi_step':
#     scheduler_obs = optim.lr_scheduler.MultiStepLR(optimizer_obs, milestones=args.multi_lr_milestones, gamma=args.multi_lr_gamma)

# if args.scheduler_inner == 'cyclic':
#     scheduler_inner = optim.lr_scheduler.CosineAnnealingLR(optimizer_inner, args.num_training_steps, args.lr_LL_min)
# elif args.scheduler_inner == 'step':
#     scheduler_inner = optim.lr_scheduler.StepLR(optimizer_inner, step_size=args.step_lr_size, gamma=args.step_lr_gamma)
# elif args.scheduler_inner == 'multi_step':
    # scheduler_inner = optim.lr_scheduler.MultiStepLR(optimizer_inner, milestones=args.multi_lr_milestones, gamma=args.multi_lr_gamma)

scheduler_NF    = make_scheduler(optimizer_NF, args, option='NF')
scheduler_obs   = make_scheduler(optimizer_obs, args, option='obs')
scheduler_inner = make_scheduler(optimizer_inner, args, option='inner')

def grad_FD(NF, obstacle, x, x_list, z_list, loss_outer_orig, d=1e-6):
    g_obs_FD = []
    
    for k,p in enumerate(obstacle.parameters()):
        size = p.numel()
        g = torch.zeros_like(p)
        for j in range(size):
            e = torch.zeros_like(p).flatten()
            e[j] = 1.

            obs_copy = copy.deepcopy(obstacle)
            NF_copy  = copy.deepcopy(NF)
            NF_copy2 = copy.deepcopy(NF)
            optimizer_copy = optim_to_use_inner(NF_copy.parameters(), lr=args.step_size_inner, weight_decay=args.l2_reg_inner)
            optimizer_copy2 = optim_to_use_inner(NF_copy2.parameters(), lr=args.step_size_inner, weight_decay=args.l2_reg_inner)

            for i in range(args.num_training_steps_inner):
                NF_copy.zero_grad()
                # LL objective
                x_i = x_list[i]
                z_i = z_list[i]
                # the lower level (MFG) objective
                loss_inner = compute_mfg_obj(NF_copy, obs_copy, x_i, z_i, args)['loss']
                # LL gradient step
                loss_inner.backward()
                optimizer_copy.step()
                # UL objective: used for PTT
                loss_outer, _ = compute_l(NF_copy, x)

            # obs_copy = copy.deepcopy(obstacle)
            p_copy = list(obs_copy.parameters())[k]
            p_copy.data = p_copy.data + d * e.reshape(p_copy.shape)
            
            for i in range(args.num_training_steps_inner):
                NF_copy2.zero_grad()
                # LL objective
                x_i = x_list[i]
                z_i = z_list[i]
                # the lower level (MFG) objective
                loss_inner2 = compute_mfg_obj(NF_copy2, obs_copy, x_i, z_i, args)['loss']
                # LL gradient step
                loss_inner2.backward()
                optimizer_copy2.step()
                # UL objective: used for PTT
                loss_outer_perturbed, _ = compute_l(NF_copy2, x)

            g = g.flatten()
            g[j] = (loss_outer_perturbed - loss_outer) / d
        
        g_obs_FD.append(g.reshape(p.shape))

    # print ("Obs param at the end: {}".format(list(obstacle.parameters())[0]))
    
    return g_obs_FD

def grad_FD_randcheck(NF, obstacle, x, x_list, z_list, l_base, g_AD_all, d=1e-6, \
                        model_to_check='obs', num_entries_checked=3, higher_opt=False):
    
    g_FD = torch.zeros(num_entries_checked).to(x.device)
    g_AD = torch.zeros(num_entries_checked).to(x.device)

    if model_to_check == 'NF':
        model = NF
    else:
        model = obstacle

    params = list(model.parameters())
    n_params = len(params)

    for n_check in range(num_entries_checked):
        obs_copy = copy.deepcopy(obstacle)
        NF_copy  = copy.deepcopy(NF)
        optimizer_copy = optim_to_use_inner(NF_copy.parameters(), lr=args.step_size_inner, weight_decay=args.l2_reg_inner)

        # select a random entry to perturb
        k = np.random.randint(n_params)
        p = params[k]
        j = np.random.randint(p.numel())
        e = torch.zeros_like(p).flatten()
        e[j] = 1.

        if model_to_check == 'NF':
            p_copy = list(NF_copy.parameters())[k]
        else:
            p_copy = list(obs_copy.parameters())[k]
        # perturb input
        p_copy.data = p_copy.data + d * e.reshape(p_copy.shape)

        # AD grad
        I = e.reshape(p_copy.shape) != 0
        g_AD[n_check] = g_AD_all[k][I]

        # FD grad
        l_perturbed = -np.inf
        if higher_opt:
            with higher.innerloop_ctx(NF_copy, optimizer_copy, copy_initial_weights=False) as (NF_inner, NF_inner_opt):
                for i in range(args.num_training_steps_inner):
                    # LL objective
                    x_i = x_list[i]
                    z_i = z_list[i]
                    # the lower level (MFG) objective
                    if args.algo == 'BDA':
                        loss_inner = args.BDA_alpha * compute_mfg_obj(NF_inner, obs_copy, x_i, z_i, args)['loss'] + \
                                     (1-args.BDA_alpha) * compute_l(NF_inner, x)[0]
                    else:
                        loss_inner = compute_mfg_obj(NF_inner, obs_copy, x_i, z_i, args)['loss']
                    ## LL gradient step
                    if args.grad_clip_LL:
                        NF_inner_opt.step(loss_inner, grad_callback=grad_clip_callback)
                    else:
                        NF_inner_opt.step(loss_inner)
                    # NF_inner_opt.step(loss_inner)
                    # UL objective
                    NF_inner(x[:,0,:])
                    l_val, _ = compute_l(NF_inner, x)
                    if args.algo == 'IAPTT':
                        if l_val > l_perturbed:
                            l_perturbed = l_val
                    else:
                        l_perturbed = l_val
        else:
            for i in range(args.num_training_steps_inner):
                NF_copy.zero_grad()
                # LL objective
                x_i = x_list[i]
                z_i = z_list[i]
                # the lower level (MFG) objective
                if args.algo == 'BDA':
                    loss_inner = args.BDA_alpha * compute_mfg_obj(NF_copy, obs_copy, x_i, z_i, args)['loss'] + \
                                 (1-args.BDA_alpha) * compute_l(NF_copy, x)[0]
                else:
                    loss_inner = compute_mfg_obj(NF_copy, obs_copy, x_i, z_i, args)['loss']
                # LL gradient step
                loss_inner.backward()
                if args.grad_clip_LL:
                    clip_grad_norm_(NF_copy.parameters(), 5.)
                optimizer_copy.step()
                # UL objective
                l_val, _ = compute_l(NF_copy, x)
                if args.algo == 'IAPTT':
                    if l_val > l_perturbed:
                        l_perturbed = l_val
                else:
                    l_perturbed = l_val
        
        g_FD[n_check] = (l_perturbed - l_base) / d


    return torch.allclose(g_AD, g_FD, rtol=1e-4, atol=1e-6), torch.norm(g_AD - g_FD)


# =================================================================================== #
#                                      Training                                       #
# =================================================================================== #

def compute_l(NF, x):
    # x: B x K x d
    # compute upper level objective at the specified parameter values
    B = x.shape[0]
    _, _, _, hist_gen, _ = NF.inverse(x[:,0,:])
    # hist_gen = partition_hist(hist_gen, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)
    F = hist_gen.reshape(B,-1) # B x K*d
    x_flat = x.reshape(B, -1)
    l_val = 1/2 * torch.mean(torch.sum((F - x_flat)**2, dim=-1))

    return l_val, hist_gen

def compute_mfg_obj(NF, obstacle, x, z, args, mode='train'):
    # forward
    log_density, _, _, hist_norm, _, _, _ = NF.log_prob(x)
    # log_density = torch.mean(log_density)

    # backward
    z_K, ld_gen, _, hist_gen, hist_ld_gen = NF.inverse(z)

    # distribution matching / terminal cost (G)
    KL_density  = torch.Tensor([0]).to(device)
    KL_sampling = torch.Tensor([0]).to(device)
    if args.NF_loss == 'KL_sampling':
        # G_cost = -log_density
        # if P_1 is not None:
        #     log_prob_1  = torch.mean(P_1.log_prob(x))
        #     KL_sampling = log_prob_1 - log_density
        log_density = torch.mean(log_density)
        if P_1 is None:
            G_cost = -log_density
        else:
            log_prob_1  = torch.mean(P_1.log_prob(x))
            KL_sampling = log_prob_1 - log_density
            G_cost = KL_sampling
    elif args.NF_loss == 'KL_density':
        assert P_1 is not None
        log_prob_0   = torch.mean(P_0.log_prob(z))
        log_prob_gen = torch.mean(P_1.log_prob(z_K))
        ld_gen       = torch.mean(ld_gen)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = KL_density
    elif args.NF_loss == 'jeffery':
        log_density = torch.mean(log_density)
        assert P_1 is not None
        log_prob_1   = torch.mean(P_1.log_prob(x))
        log_prob_0   = torch.mean(P_0.log_prob(z))
        log_prob_gen = torch.mean(P_1.log_prob(z_K))
        ld_gen       = torch.mean(ld_gen)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = log_prob_1 - log_density + KL_density # equal to KL_sampling + KL_density
    else:
        raise NotImplementedError()

    # OT regularization (L)
    L_cost = torch.Tensor([0]).to(device)
    if args.lbd_OT != 0:
        if args.reg_OT_dir == 'gen':
            hist   = hist_gen
        else:
            hist   = hist_norm
        L_cost, _ = compute_OT_cost(hist, args, mode=args.reg_OT_dir, partition_mode=args.OT_part, LU_last=args.LU_last,
                                        scheme=args.disc_scheme, part_hist=False)
        L_cost = torch.mean(L_cost)

    # interaction (F)
    F_P    = torch.Tensor([0]).to(device)
    F_E    = torch.Tensor([0]).to(device)
    F_cost = torch.Tensor([0]).to(device)
    if args.interaction:
        # hist_part    = partition_hist(hist_gen, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last)
        # hist_ld_part = partition_hist(hist_ld_gen, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, hist_type='ld')
        log_prob_0   = P_0.log_prob(z)
        F_E, F_P     = compute_F(args, obstacle, hist, log_prob_0, hist_ld_gen, \
                                Q_is_dist=False, scheme=args.F_disc_scheme, pad_ld=False)
        F_E          = torch.mean(F_E)
        F_P          = torch.mean(F_P)
        F_cost       = args.lbd_F_E * F_E + args.lbd_F_P * F_P

    # Overall loss
    loss = G_cost + args.lbd_OT * L_cost + args.lbd_F * F_cost

    if tbx_logging:
        writer.add_scalar(tag='OT_cost' + '_' + mode, scalar_value=L_cost.item(), global_step=step_inner)
        writer.add_scalar(tag='G_cost'+ '_' + mode, scalar_value=G_cost.item(), global_step=step_inner)
        writer.add_scalar(tag='loss'+ '_' + mode, scalar_value=loss.item(), global_step=step_inner)
        writer.add_scalar(tag='KL_density'+ '_' + mode, scalar_value=KL_density.item(), global_step=step_inner)
        writer.add_scalar(tag='KL_sampling'+ '_' + mode, scalar_value=KL_sampling.item(), global_step=step_inner)
        if args.interaction:
            writer.add_scalar(tag='F_P'+ '_' + mode, scalar_value=F_P.item(), global_step=step_inner)
            writer.add_scalar(tag='F_E'+ '_' + mode, scalar_value=F_E.item(), global_step=step_inner)
            writer.add_scalar(tag='F_cost'+ '_' + mode, scalar_value=F_cost.item(), global_step=step_inner)

    return {'loss': loss, 'G': G_cost, 'L': L_cost, 'F': F_cost}


def solve_MFG(NF, NF_hat_prev, obs, args):
    if args.penalty_LL_init == 'last_min':
        NF_hat = copy.deepcopy(NF_hat_prev)
    elif args.penalty_LL_init == 'last_iter':
        NF_hat = copy.deepcopy(NF)
    else:
        transform, K = create_transform_iterative()
        NF_hat = NF_iterative_flatVar(transform, P_0, K).to(device)
    
    obstacle = copy.deepcopy(obs)
    # disable grad on obs for faster AD in the LL loops
    disable_grad(obstacle)

    # TODO: we could use the same optimizer and keep historical info. Not sure if that'd be good
    # since the obs is constantly changing.
    optimizer_hat = optim_to_use_inner(NF_hat.parameters(), lr=args.step_size_inner, weight_decay=args.l2_reg_inner)
    scheduler_hat = make_scheduler(optimizer_hat, args, option='inner')

    # for tracking historical params
    NF_hat_list = []
    param_names = []
    for name, p in list(NF_hat.named_parameters()):
        param_names.append(name)

    for i in range(args.num_training_steps_inner):
        optimizer_hat.zero_grad()
        x_samples = P_1.sample(args.train_batch_size_inner)
        z_samples = P_0.sample(args.train_batch_size_inner)
        mfg_loss = compute_mfg_obj(NF_hat, obstacle, x_samples, z_samples, args)['loss']

        mfg_loss.backward()
        if args.grad_clip_LL:
            clip_grad_norm_(NF_hat.parameters(), args.grad_clip_value_LL)
        optimizer_hat.step()
        if scheduler_hat is not None:
            if args.scheduler_inner == 'adaptive':
                scheduler_hat.step(mfg_loss)
            else:
                scheduler_hat.step()

        # save the last few iterations and average them to approximate NF_hat
        if args.num_training_steps_inner - i <= args.penalty_H_approx_iter:
            params = tuple(NF_hat.parameters())
            params = tuple([torch.Tensor(p.detach().clone().cpu()).to(device) for p in params])
            NF_hat_list.append(params)

    if args.penalty_H_approx_iter > 1:
        load_weights(NF_hat, param_names, average_params(NF_hat_list), as_params=True)

    # disable NF_hat grad for faster AD on H later
    disable_grad(NF_hat)

    return NF_hat

if args.load_pretrain_NF:
    NF.load_state_dict(torch.load(args.pretrain_NF_dir))
    print ("Loaded NF from: {}".format(args.pretrain_NF_dir))

if args.pretrain_NF:
    optim_to_use_NF_pretrain = optimizer_dict[args.optimizer_NF_pretrain]
    optimizer_NF_pretrain = optim_to_use_NF_pretrain(NF.parameters(), lr=args.lr_NF, weight_decay=args.l2_reg_NF)
    if args.scheduler_NF_pretrain == 'adaptive':
        scheduler_NF_pretrain = optim.lr_scheduler.ReduceLROnPlateau(optimizer_NF_pretrain, mode='min', factor=0.5, verbose=True, patience=args.patience_NF)
    elif args.scheduler_NF_pretrain == 'cyclic':
        scheduler_NF_pretrain = optim.lr_scheduler.CosineAnnealingLR(optimizer_NF_pretrain, args.num_training_steps, 0)
    elif args.scheduler_NF_pretrain == 'multi_step':
        scheduler_NF_pretrain = optim.lr_scheduler.MultiStepLR(optimizer_NF_pretrain, milestones=args.multi_lr_milestones, gamma=args.multi_lr_gamma)

    loss_hist = []
    for step in range(args.num_pretrain_NF_steps):
        optimizer_NF_pretrain.zero_grad()
        # training data: a batch of trajectories. Shape: B x K x d
        if args.debug_tensor_type == 'double':
            x = next(train_generator).to(device).double()
        else:
            x = next(train_generator).to(device)

        loss, _ = compute_l(NF, x)
        loss.backward()
        if args.pretrain_NF_grad_clip:
            clip_grad_norm_(NF.parameters(), args.grad_clip_value_NF)
        optimizer_NF_pretrain.step()
        loss_hist.append(float(loss))

        if (step + 1) % args.pretrain_monitor_interval == 0:
            avg_loss = np.mean(loss_hist)
            print ("Step: {}, loss: {:.5f}".format(step, avg_loss))
            scheduler_NF_pretrain.step(np.mean(avg_loss))
            loss_hist = []

    # evaluate
    loss_test_hist = []
    with torch.no_grad():
        for x in test_loader:
            if args.debug_tensor_type == 'double':
                x = x.to(device).double()
            else:
                x = x.to(device)
            loss, _ = compute_l(NF, x)
            loss_test_hist.append(float(loss))
    avg_test_loss = np.mean(loss_test_hist)
    std_test_loss = np.std(loss_test_hist)
    print("NF pretraining finished. UL on the test set: {:.5f} +- {:.5f}".format(\
        avg_test_loss, 2 * std_test_loss / np.sqrt(len(loss_test_hist))))
    

    # save pretrained NF
    NF_pretrain_path  = os.path.join(model_dir, 'NF_UL_pretrain.t')
    torch.save(NF.state_dict(), NF_pretrain_path)

# this loads a obstacle that we know to be close to the ground truth
if args.load_pretrain_obs:
    obstacle.load_state_dict(torch.load(args.pretrain_obs_dir))
    print ("Loaded obstacle from: {}".format(args.pretrain_obs_dir))
    for p in obstacle.parameters():
        p.data = p.data + args.pretrain_obs_eps * torch.rand_like(p)


# logging
B_err_min  = np.inf
step_inner = 0
obs_best = None
NF_best  = None
obs_state_prev = None
NF_state_prev  = None
L_gap_hist = []
l_gap_hist = []
obs_best_path = os.path.join(model_dir, 'obs_best.t')
NF_best_path  = os.path.join(model_dir, 'NF_best.t')
obs_last_path = os.path.join(model_dir, 'obs_last.t')
NF_last_path  = os.path.join(model_dir, 'NF_last.t')
# optim_LL_state = None

########  main loop  #######
for step in range(args.num_training_steps):
    if args.scheduler_inner == 'gap':
        NF_state_prev  = copy.deepcopy(NF.state_dict())
        obs_state_prev = copy.deepcopy(obstacle.state_dict())
    obstacle.zero_grad()
    NF.zero_grad()

    l_log_list = []
    L_log_list = []
    i_max = 1
    l_max = -np.inf
    x_list = []
    z_list = []

    # training data: a batch of trajectories. Shape: B x K x d
    if args.debug_tensor_type == 'double':
        x = next(train_generator).to(device).double()
    else:
        x = next(train_generator).to(device)
    with torch.no_grad():
        loss_UL, _ = compute_l(NF, x)
        x_UL = P_1.sample(args.train_batch_size_inner)
        z_UL = P_0.sample(args.train_batch_size_inner)
        L_UL = compute_mfg_obj(NF, obstacle, x_UL, z_UL, args)['loss']

    if args.algo == 'GM':
        # ### debug
        # NF_copy  = copy.deepcopy(NF)
        # optimizer_copy = optim_to_use_inner(NF_copy.parameters(), lr=args.step_size_inner, weight_decay=args.l2_reg_inner)
        # ###
        with higher.innerloop_ctx(NF, optimizer_inner, copy_initial_weights=False) as (NF_inner, NF_inner_opt):
            # # optimizer state persistence for e.g. Adam
            # if optim_LL_state is not None:
            #     NF_inner_opt.state = optim_LL_state
            for i in range(args.num_training_steps_inner):
                # LL objective
                x_i = P_1.sample(args.train_batch_size_inner)
                z_i = P_0.sample(args.train_batch_size_inner)
                # the lower level (MFG) objective
                loss_inner = compute_mfg_obj(NF_inner, obstacle, x_i, z_i, args)['loss']
                # LL gradient step
                if args.grad_clip_LL:
                    NF_inner_opt.step(loss_inner, grad_callback=grad_clip_callback)
                else:
                    NF_inner_opt.step(loss_inner)

                # UL objective: used for PTT
                # FOR SOME REASON, if we don't do a forward before the inverse (in compute_l), the inverse will be wrong. 
                NF_inner(x[:,0,:])
                loss_outer, hist_gen = compute_l(NF_inner, x)

                # ### debug
                # NF_copy.zero_grad()
                # loss_inner_cp = compute_mfg_obj(NF_copy, obstacle, x_i, z_i, args)['loss']
                # # LL gradient step
                # loss_inner_cp.backward()
                # if args.grad_clip_LL:
                #     clip_grad_norm_(NF_copy.parameters(), 5.)
                # optimizer_copy.step()
                # loss_outer_cp, hist_gen_cp = compute_l(NF_copy, x)
                # ###
                
                # # PTT
                # if loss_outer > l_max:
                #     i_max = i+1
                #     l_max = loss_outer

                # logging 
                L_log_list.append(float(loss_inner))
                l_log_list.append(float(loss_outer))

                # routine gradcheck
                if (step + 1) % args.gradcheck_interval == 0:
                    x_list.append(x_i.detach().clone())
                    z_list.append(z_i.detach().clone())

            # compute UL gradient
            g_obs = grad(loss_outer, obstacle.parameters(), retain_graph=True, allow_unused=True)

        # periodic grad checking
        if (step + 1) % args.gradcheck_interval == 0:
            # gradcheck on random entries
            obs_gradcheck_result, obs_grad_diff = grad_FD_randcheck(NF, obstacle, x, x_list, z_list, loss_outer, \
                                                g_obs, model_to_check='obs', higher_opt=args.gradcheck_use_higher_opt)

            print ("Step: {}, Obs gradcheck result: {}, norm diff: {:.3f}".format(step, obs_gradcheck_result, float(obs_grad_diff)))

        # update NF
        if args.NF_keep_params:
            params_inner = list(NF_inner.parameters(time=-1))
            for i,p in enumerate(NF.parameters()):
                p.data = params_inner[i].data

        # update obs.
        if step + 1 > args.obs_warm_up:
            obstacle = set_grad(obstacle, g_obs)
            if args.obs_reg_mass_loss != 'none':
                B_val, B_true_val = eval_obstacles_on_grid(obstacle, B_true, grid, grid_pad)
                if args.obs_reg_mass_loss == 'l1':
                    mass_diff = args.lbd_mass * torch.abs(torch.norm(B_val) - torch.norm(B_true_val))
                else:
                    mass_diff = args.lbd_mass * (torch.norm(B_val) - torch.norm(B_true_val))**2
                mass_diff.backward()
            if args.grad_clip_obs:
                clip_grad_norm_(obstacle.parameters(), args.grad_clip_value_obs)
            optimizer_obs.step()
    
    elif args.algo == 'IAPTT':
        with higher.innerloop_ctx(NF, optimizer_inner, copy_initial_weights=False) as (NF_inner, NF_inner_opt):
            for i in range(args.num_training_steps_inner):
                # LL objective
                x_i = P_1.sample(args.train_batch_size_inner)
                z_i = P_0.sample(args.train_batch_size_inner)
                # the lower level (MFG) objective
                loss_inner = compute_mfg_obj(NF_inner, obstacle, x_i, z_i, args)['loss']
                # LL gradient step
                NF_inner_opt.step(loss_inner)

                # UL objective: used for PTT
                # FOR SOME REASON, if we don't do a forward before the inverse (in compute_l), the inverse will be wrong. 
                NF_inner(x[:,0,:])
                loss_outer, hist_gen = compute_l(NF_inner, x)

                # PTT
                if loss_outer > l_max:
                    i_max = i+1
                    l_max = loss_outer

                # logging 
                L_log_list.append(float(loss_inner))
                l_log_list.append(float(loss_outer))

                # routine gradcheck
                if (step + 1) % args.gradcheck_interval == 0:
                    x_list.append(x_i.detach().clone())
                    z_list.append(z_i.detach().clone())

            # compute UL gradient
            g_NF  = grad(l_max, NF_inner.parameters(time=0), retain_graph=True, allow_unused=True)
            g_obs = grad(l_max, obstacle.parameters(), retain_graph=True, allow_unused=True)

        # periodic grad checking
        if (step + 1) % args.gradcheck_interval == 0:
            # gradcheck on random entries
            NF_gradcheck_result, NF_grad_diff   = grad_FD_randcheck(NF, obstacle, x, x_list, z_list, l_max, \
                                                                    g_NF, model_to_check='NF', higher_opt=True)
            obs_gradcheck_result, obs_grad_diff = grad_FD_randcheck(NF, obstacle, x, x_list, z_list, l_max, \
                                                                    g_obs, model_to_check='obs', higher_opt=True)

            print ("Step: {}, NF gradcheck result: {}, norm diff: {:.3f}; Obs gradcheck result: {}, norm diff: {:.3f}".format(\
                step, NF_gradcheck_result, float(NF_grad_diff), obs_gradcheck_result, float(obs_grad_diff)))

        # update NF
        NF = set_grad(NF, g_NF)
        optimizer_NF.step()
        # update obs.
        if step + 1 > args.obs_warm_up:
            obstacle = set_grad(obstacle, g_obs)
            optimizer_obs.step()
            
    elif args.algo == 'BDA':
        with higher.innerloop_ctx(NF, optimizer_inner, copy_initial_weights=False) as (NF_inner, NF_inner_opt):
            for i in range(args.num_training_steps_inner):
                # LL objective
                x_i = P_1.sample(args.train_batch_size_inner)
                z_i = P_0.sample(args.train_batch_size_inner)
                # the lower level (MFG) objective
                if args.BDA_adaptive_alpha:
                    loss_inner = (1-args.BDA_alpha/(i+1)) * compute_mfg_obj(NF_inner, obstacle, x_i, z_i, args)['loss'] + \
                                args.BDA_alpha/(i+1) * compute_l(NF_inner, x)[0]
                else:
                    loss_inner = (1-args.BDA_alpha) * compute_mfg_obj(NF_inner, obstacle, x_i, z_i, args)['loss'] + \
                                args.BDA_alpha * compute_l(NF_inner, x)[0]
                # LL gradient step
                NF_inner_opt.step(loss_inner)

                # UL objective: used for PTT
                # FOR SOME REASON, if we don't do a forward before the inverse (in compute_l), the inverse will be wrong. 
                NF_inner(x[:,0,:])
                loss_outer, hist_gen = compute_l(NF_inner, x)

                # logging 
                L_log_list.append(float(loss_inner))
                l_log_list.append(float(loss_outer))

                # routine gradcheck
                if (step + 1) % args.gradcheck_interval == 0:
                    x_list.append(x_i.detach().clone())
                    z_list.append(z_i.detach().clone())

            # compute UL gradient
            g_obs = grad(loss_outer, obstacle.parameters(), retain_graph=True, allow_unused=True)

        # periodic grad checking
        if (step + 1) % args.gradcheck_interval == 0:
            # gradcheck on random entries
            obs_gradcheck_result, obs_grad_diff = grad_FD_randcheck(NF, obstacle, x, x_list, z_list, loss_outer, \
                                                                    g_obs, model_to_check='obs', higher_opt=True)

            print ("Step: {}, Obs gradcheck result: {}, norm diff: {:.3f}".format(step, obs_gradcheck_result, float(obs_grad_diff)))

        # update NF
        if args.NF_keep_params:
            params_inner = list(NF_inner.parameters(time=-1))
            for i,p in enumerate(NF.parameters()):
                p.data = params_inner[i].data

        # update obs
        if step + 1 > args.obs_warm_up:
            obstacle = set_grad(obstacle, g_obs)
            if args.obs_reg_mass_loss != 'none':
                B_val, B_true_val = eval_obstacles_on_grid(obstacle, B_true, grid, grid_pad)
                mass_diff = args.lbd_mass * (torch.norm(B_val) - torch.norm(B_true_val))**2
                mass_diff.backward()
            optimizer_obs.step()

    elif args.algo == 'GM_true_obs': 
        # # Here, we use the ground truth obstacle and see if the LL can converge to the 
        # # solution of the forward problem
        # args.Q_is_dist = args.Q_true_is_dist # for this case, the parametrized obstacle coincides with the real one. 
        # with higher.innerloop_ctx(NF, optimizer_inner, copy_initial_weights=False) as (NF_inner, NF_inner_opt):
        #     for i in range(args.num_training_steps_inner):
        #         # LL objective
        #         x_i = P_1.sample(args.train_batch_size_inner)
        #         z_i = P_0.sample(args.train_batch_size_inner)
        #         # the lower level (MFG) objective
        #         loss_inner = compute_mfg_obj(NF_inner, B_true, x_i, z_i, args)['loss']
        #         # LL gradient step
        #         NF_inner_opt.step(loss_inner)

        #         # UL objective: used for PTT
        #         # FOR SOME REASON, if we don't do a forward before the inverse (in compute_l), the inverse will be wrong. 
        #         NF_inner(x[:,0,:])
        #         loss_outer, hist_gen = compute_l(NF_inner, x)

        #         # logging 
        #         L_log_list.append(float(loss_inner))
        #         l_log_list.append(float(loss_outer))

        #         # routine gradcheck
        #         if (step + 1) % args.gradcheck_interval == 0:
        #             x_list.append(x_i.detach().clone())
        #             z_list.append(z_i.detach().clone())

        # # periodic grad checking
        # if (step + 1) % args.gradcheck_interval == 0:
        #     # gradcheck on random entries
        #     obs_gradcheck_result, obs_grad_diff = grad_FD_randcheck(NF, obstacle, x, x_list, z_list, loss_outer, \
        #                                                             g_obs, model_to_check='obs', higher_opt=True)

        #     print ("Step: {}, Obs gradcheck result: {}, norm diff: {:.3f}".format(step, obs_gradcheck_result, float(obs_grad_diff)))

        # # update NF
        # if args.NF_keep_params:
        #     params_inner = list(NF_inner.parameters(time=-1))
        #     for i,p in enumerate(NF.parameters()):
        #         p.data = params_inner[i].data


        args.Q_is_dist = args.Q_true_is_dist # for this case, the parametrized obstacle coincides with the real one. 
        for i in range(args.num_training_steps_inner):
            # LL objective
            x_i = P_1.sample(args.train_batch_size_inner)
            z_i = P_0.sample(args.train_batch_size_inner)
            # the lower level (MFG) objective
            loss_inner = compute_mfg_obj(NF, B_true, x_i, z_i, args)['loss']
            # LL gradient step
            loss_inner.backward()
            optimizer_inner.step()

            loss_outer, hist_gen = compute_l(NF, x)

            # logging 
            L_log_list.append(float(loss_inner))
            l_log_list.append(float(loss_outer))

    elif args.algo == 'penalty':
        # approximate solution
        NF_hat = solve_MFG(NF, NF_hat, obstacle, args)
        # clear grads
        optimizer_NF.zero_grad()
        optimizer_obs.zero_grad()

        # compute loss
        x_samples = P_1.sample(args.train_batch_size)
        z_samples = P_0.sample(args.train_batch_size)
        l = compute_l(NF, x)[0]
        L = compute_mfg_obj(NF, obstacle, x_samples, z_samples, args)['loss']
        # TODO: this will accumulate grad on NF_hat too. Disable it for speed. However, shouldn't be any issue
        # since we don't optimizer NF_hat, and its gradient will be cleared in the next iteration.
        H = compute_mfg_obj(NF_hat, obstacle, x_samples, z_samples, args)['loss']
        loss = l + args.lbd_penalty * penalty_fun(L - H)

        # optimize
        loss.backward() # should accumulate gradients on both NF and obstacle
        if args.grad_clip_NF:
            clip_grad_norm_(NF.parameters(), args.grad_clip_value_NF)
        optimizer_NF.step()

    elif args.algo == 'penalty_fixNF':
        # approximate solution
        NF_hat = solve_MFG(NF, NF_hat, obstacle, args)
        # clear grads
        optimizer_obs.zero_grad()

        # compute loss
        x_samples = P_1.sample(args.train_batch_size)
        z_samples = P_0.sample(args.train_batch_size)
        L = compute_mfg_obj(NF, obstacle, x_samples, z_samples, args)['loss']
        H = compute_mfg_obj(NF_hat, obstacle, x_samples, z_samples, args)['loss']
        loss = penalty_fun(L - H)

        # optimize
        loss.backward()
    else:
        raise NotImplementedError()

    # optimize obs
    B_val, B_true_val = eval_obstacles_on_grid(obstacle, B_true, grid, grid_pad) # both are n^2 x 1
    if step + 1 > args.obs_warm_up:
        if args.obs_reg_mass_loss != 'none':
            # B_val, B_true_val = eval_obstacles_on_grid(obstacle, B_true, grid, grid_pad)
            if args.obs_reg_mass_loss == 'l1':
                mass_diff = args.lbd_mass * torch.abs(torch.norm(B_val) - torch.norm(B_true_val))
            else:
                mass_diff = args.lbd_mass * (torch.norm(B_val) - torch.norm(B_true_val))**2
            mass_diff.backward()
        if args.grad_clip_obs:
            clip_grad_norm_(obstacle.parameters(), args.grad_clip_value_obs)
        optimizer_obs.step()

    # compute obstacle error
    err = 1/2*torch.norm(B_val.reshape(-1) - B_true_val)**2 * dx**2
    err_inf = torch.max(torch.abs(B_val.reshape(-1) - B_true_val))
    err_l2  = torch.norm(B_val.reshape(-1) - B_true_val) * dx**2
    err_rel = torch.norm(B_val.reshape(-1) - B_true_val) / torch.norm(B_true_val)
    err_inf_rel = err_inf / torch.max(torch.abs(B_true_val))
    if err_rel < B_err_min:
        B_err_min = err_rel
        obs_best = copy.deepcopy(obstacle)
        NF_best  = copy.deepcopy(NF)

    # monitor and log useful info
    if args.algo in ['penalty', 'penalty_fixNF']:
        L_H_gap = float(L-H)
    else:
        l_LL_mean = float(np.mean(l_log_list))
        L_LL_mean = float(np.mean(L_log_list))
        l_gap = float(l_LL_mean - loss_UL)
        L_gap = float(L_LL_mean - L_UL)
        L_gap_hist.append(L_gap)
        l_gap_hist.append(l_gap)

    # obstacle scheduler
    if args.scheduler_obs == 'adaptive':
        scheduler_obs.step(err)
    elif args.scheduler_obs == 'gap':
        # decay lr when blow ups are detected
        if (step + 1) > args.scheduler_gap_warmup and check_bilevel_blowup(args, L_gap_hist, L_gap, l_gap_hist, l_gap):
            print ("Blow up detected, decay obs lr by {}".format(args.scheduler_gap_gamma))
            for g in optimizer_obs.param_groups:
                g['lr'] = g['lr'] * args.scheduler_gap_gamma
            # restore the NF and obstacles to the state prior to this update
            NF.load_state_dict(NF_state_prev)
            obstacle.load_state_dict(obs_state_prev)
    else:
        scheduler_obs.step()
    
    # LL scheduler - penalty based methods work differently
    if args.algo not in ['penalty', 'penalty_fixNF']:
        if args.scheduler_inner not in ['none', 'gap']:
            scheduler_inner.step()
        elif args.scheduler_inner == 'gap':
            if (step + 1) > args.scheduler_gap_warmup and check_bilevel_blowup(args, L_gap_hist, L_gap, l_gap_hist, l_gap):
                print ("Blow up detected, decay LL lr by {}".format(args.scheduler_gap_gamma))
                for g in optimizer_inner.param_groups:
                    g['lr'] = g['lr'] * args.scheduler_gap_gamma
                # restore the NF and obstacles to the state prior to this update
                NF.load_state_dict(NF_state_prev)
                obstacle.load_state_dict(obs_state_prev)

    # logging
    if args.algo in ['penalty', 'penalty_fixNF']:
        print ("Step: {}, loss: {:.4f}, rel L2 err: {:.4f}, L_inf err: {:.4f}, rel L_inf err: {:.4f}, L_H_gap: {:.4f}, l UL: {:.4f}, L UL: {:.4f}, Obs lr: {}, LL lr: {}".format(\
                step, loss, err_rel, err_inf, err_inf_rel, L_H_gap, loss_UL, L_UL, optimizer_obs.param_groups[0]['lr'], \
                optimizer_inner.param_groups[0]['lr']))
        log_names = ['steps', 'loss', 'lower_UL', 'upper_UL', 'L_H_gap', \
                    'B_err', 'B_err_l2', 'B_err_inf', 'B_err_rel', \
                    'B_err_inf_rel']
        log_data  = [step, float(loss), float(L_UL), float(loss_UL), L_H_gap, \
                    float(err), float(err_l2), float(err_inf), float(err_rel), \
                    float(err_inf_rel)]
    else:
        print ("Step: {}, obs L2 err: {:.4f}, rel L2 err: {:.4f}, L_inf err: {:.4f}, rel L_inf err: {:.4f}, l UL: {:.4f}, avg l LL: {:.4f}, l gap: {:.4f}, i_max: {}, L UL: {:.4f}, avg L LL: {:.4f}, L gap: {:.4f}, Obs lr: {}, LL lr: {}".format(\
                step, err, err_rel, err_inf, err_inf_rel, loss_UL, l_LL_mean, l_gap, i_max, L_UL, L_LL_mean, L_gap, optimizer_obs.param_groups[0]['lr'], \
                optimizer_inner.param_groups[0]['lr']))

        log_names = ['steps', 'lower_UL', 'lower_LL_mean', 'upper_UL', 'upper_LL_mean', \
                    'lower_gap', 'upper_gap', 'B_err', 'B_err_l2', 'B_err_inf', 'B_err_rel', \
                    'B_err_inf_rel']
        log_data  = [step, float(L_UL), l_LL_mean, float(loss_UL), L_LL_mean, \
                    L_gap, l_gap, float(err), float(err_l2), float(err_inf), \
                    float(err_rel), float(err_inf_rel)]
        
    plot_logger.log_multiple(log_names, log_data)

    if args.verbose_logging:
        if args.algo in ['penalty', 'penalty_fixNF']:
            grad_norm_NF  = np.sum([float(torch.norm(p.grad.data)) for p in NF.parameters() if p.grad is not None])
            grad_norm_obs = np.sum([float(torch.norm(p.grad.data)) for p in obstacle.parameters() if p.grad is not None])
            print ("Gradient norm on NF: {:.2f}, averaged: {}".format(grad_norm_NF, grad_norm_NF / num_params_NF))
            print ("Gradient norm on obs: {:.2f}, averaged: {}".format(grad_norm_obs, grad_norm_obs / num_params_obs))
            print ("Obs norm: {:.3f}, True obs norm: {:.3f}".format(torch.norm(B_val), torch.norm(B_true_val)))
        else:
            # grad_norm_NF  = np.sum([float(torch.norm(p)) for p in g_NF if p is not None])
            grad_norm_obs = np.sum([float(torch.norm(p)) for p in g_obs if p is not None])
            # print ("Gradient norm on NF: {:.2f}, averaged: {}".format(grad_norm_NF, grad_norm_NF / num_params_NF))
            print ("Gradient norm on obs: {:.2f}, averaged: {}".format(grad_norm_obs, grad_norm_obs / num_params_obs))
            print ("Obs norm: {:.3f}, True obs norm: {:.3f}".format(torch.norm(B_val), torch.norm(B_true_val)))

    if tbx_logging:
        if args.algo in ['penalty', 'penalty_fixNF']:
            writer.add_scalar(tag='lower_mean', scalar_value=float(L_UL), global_step=step)
            writer.add_scalar(tag='upper_mean', scalar_value=float(loss_UL), global_step=step)
            writer.add_scalar(tag='L_H_gap',    scalar_value=L_H_gap, global_step=step)
            writer.add_scalar(tag='loss',       scalar_value=float(loss), global_step=step)
        else:
            writer.add_scalar(tag='lower_mean', scalar_value=float(np.mean(L_log_list)), global_step=step)
            writer.add_scalar(tag='upper_mean', scalar_value=float(np.mean(l_log_list)), global_step=step)
        writer.add_scalar(tag='B_err', scalar_value=float(err), global_step=step)
        writer.add_scalar(tag='B_err_rel', scalar_value=float(err_rel), global_step=step)
        writer.add_scalar(tag='i_max', scalar_value=i_max, global_step=step)

    if (step + 1) % args.monitor_interval == 0:
        l_test_list = []
        # get test error
        for x_test in test_loader:
            if args.debug_tensor_type == 'double':
                x_test = x_test.to(device).double()
            else:
                x_test = x_test.to(device)
            l_test_list.append(float(compute_l(NF, x_test)[0]))
        print ("Step: {}, avg l on the test set: {:.4f} += {:.4f}".format(step, np.mean(l_test_list), \
            2*np.std(l_test_list)/np.sqrt(len(l_test_list)) ))

    if (step + 1) % args.plot_interval == 0:
        fig_name = 'obs_{}'.format(step+1)
        plot_obstacles(B_val, B_true_val, grid_x, grid_y, plot_dir, writer, tbx_logging, fig_name)

    if (step + 1) % args.save_interval == 0:
        # save best models
        torch.save(obs_best.state_dict(), obs_best_path)
        torch.save(NF_best.state_dict(), NF_best_path)
        torch.save(obstacle.state_dict(), obs_last_path)
        torch.save(NF.state_dict(), NF_last_path)
        # save plot for best obs
        B_best_val, B_true_val = eval_obstacles_on_grid(obs_best, B_true, grid, grid_pad)
        fig_name = 'obs_best'
        plot_obstacles(B_best_val, B_true_val, grid_x, grid_y, plot_dir, writer, tbx_logging, fig_name)

# logging
print ("Exp name: {}".format(args.exp_name))
print ("Training finished, best obstacle relative L2 error: {:.6f}".format(B_err_min))

# save best models
torch.save(obs_best.state_dict(), obs_best_path)
torch.save(NF_best.state_dict(), NF_best_path)
torch.save(obstacle.state_dict(), obs_last_path)
torch.save(NF.state_dict(), NF_last_path)
# save plot for best obs
B_best_val, B_true_val = eval_obstacles_on_grid(obs_best, B_true, grid, grid_pad)
fig_name = 'obs_best'
plot_obstacles(B_best_val, B_true_val, grid_x, grid_y, plot_dir, writer, tbx_logging, fig_name)

# =================================================================================== #
#                                     Plotting                                        #
# =================================================================================== #

traj_train = []
traj_pred  = []
traj_test      = []
traj_test_pred = []

for i in range(args.num_batch_to_plot):
    if args.debug_tensor_type == 'double':
        x = next(train_generator).to(device).double()
    else:
        x = next(train_generator).to(device)
    _, _, _, hist, _ = NF.inverse(x[:,0,:])
    # hist = partition_hist(hist, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part) # B x K x d

    traj_train.append(x.detach().cpu())
    traj_pred.append(hist.detach().cpu())

for i in range(args.num_batch_to_plot):
    if args.debug_tensor_type == 'double':
        x = next(test_generator).to(device).double()
    else:
        x = next(test_generator).to(device)
    _, _, _, hist, _ = NF.inverse(x[:,0,:])
    # hist = partition_hist(hist, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part) # B x K x d

    traj_test.append(x.detach().cpu())
    traj_test_pred.append(hist.detach().cpu())

traj_train = torch.cat(traj_train)
traj_pred  = torch.cat(traj_pred)
traj_test = torch.cat(traj_test)
traj_test_pred  = torch.cat(traj_test_pred)

# save data for plotting
grid_save_path       = os.path.join(plot_data_dir, 'grid.mat')
grid_x_save_path     = os.path.join(plot_data_dir, 'grid_x.mat')
grid_y_save_path     = os.path.join(plot_data_dir, 'grid_y.mat')
B_save_path          = os.path.join(plot_data_dir, 'B.mat')
B_best_save_path     = os.path.join(plot_data_dir, 'B_best.mat')
B_true_save_path     = os.path.join(plot_data_dir, 'B_true.mat')
traj_train_save_path = os.path.join(plot_data_dir, 'traj_train.mat')
traj_pred_save_path  = os.path.join(plot_data_dir, 'traj_pred.mat')
traj_test_save_path      = os.path.join(plot_data_dir, 'traj_test.mat')
traj_test_pred_save_path = os.path.join(plot_data_dir, 'traj_test_pred.mat')

scipy.io.savemat(grid_save_path,  dict(data=grid.detach().cpu().numpy()))
scipy.io.savemat(grid_x_save_path,  dict(data=grid_x))
scipy.io.savemat(grid_y_save_path,  dict(data=grid_y))
scipy.io.savemat(B_save_path,  dict(data=B_val.reshape(grid_x.shape).detach().cpu().numpy()))
scipy.io.savemat(B_best_save_path,  dict(data=B_best_val.reshape(grid_x.shape).detach().cpu().numpy()))
scipy.io.savemat(B_true_save_path, dict(data=B_true_val.reshape(grid_x.shape).detach().cpu().numpy()))
scipy.io.savemat(traj_train_save_path,  dict(data=traj_train.numpy()))
scipy.io.savemat(traj_pred_save_path,  dict(data=traj_pred.numpy()))
scipy.io.savemat(traj_test_save_path,  dict(data=traj_test.numpy()))
scipy.io.savemat(traj_test_pred_save_path,  dict(data=traj_test_pred.numpy()))


print ("All done.")