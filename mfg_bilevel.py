import argparse
import json
from symbol import parameters
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

# def parse_arguments():
#     parser = argparse.ArgumentParser()

#     # data
#     parser.add_argument('--exp_name', type=str, default='1')
#     parser.add_argument('--dataset_name', type=str, default='crowd_motion_gaussian_bilevel',
#                         choices=['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong'],
#                         help='Name of dataset to use.')
#     # parser.add_argument('--train_batch_size', type=int, default=64,
#     #                     help='Size of batch used for training.')
#     parser.add_argument('--val_frac', type=float, default=1.,
#                         help='Fraction of validation set to use.')
#     # parser.add_argument('--val_batch_size', type=int, default=512,
#     #                     help='Size of batch used for validation.')

#     # flow details
#     parser.add_argument('--base_transform_type', type=str, default='rq-autoregressive',
#                         choices=['affine-coupling', 'quadratic-coupling', 'rq-coupling',
#                                 'affine-autoregressive', 'quadratic-autoregressive',
#                                 'rq-autoregressive'],
#                         help='Type of transform to use between linear layers.')
#     parser.add_argument('--linear_transform_type', type=str, default='lu',
#                         choices=['permutation', 'lu', 'svd', 'lu_no_perm'],
#                         help='Type of linear transform to use.')
#     parser.add_argument('--num_flow_steps', type=int, default=10,
#                         help='Number of blocks to use in flow.')
#     parser.add_argument('--hidden_features', type=int, default=256,
#                         help='Number of hidden features to use in coupling/autoregressive nets.')
#     parser.add_argument('--tail_bound', type=float, default=3,
#                         help='Box is on [-bound, bound]^2')
#     parser.add_argument('--num_bins', type=int, default=8,
#                         help='Number of bins to use for piecewise transforms.')
#     parser.add_argument('--num_transform_blocks', type=int, default=2,
#                         help='Number of blocks to use in coupling/autoregressive nets.')
#     parser.add_argument('--use_batch_norm', type=int, default=0,
#                         choices=[0, 1],
#                         help='Whether to use batch norm in coupling/autoregressive nets.')
#     parser.add_argument('--dropout_probability', type=float, default=0.25,
#                         help='Dropout probability for coupling/autoregressive nets.')
#     parser.add_argument('--apply_unconditional_transform', type=int, default=1,
#                         choices=[0, 1],
#                         help='Whether to unconditionally transform \'identity\' '
#                             'features in coupling layer.')
#     parser.add_argument('--base_net_act', type=str, default='relu',
#                         choices=['relu', 'tanh'])

#     # logging and checkpoints
#     parser.add_argument('--monitor_interval', type=int, default=1,
#                         help='Interval in steps at which to report training stats.')
#     parser.add_argument('--plot_interval', type=int, default=50,
#                         help='Interval in steps at which to report training stats.')
#     parser.add_argument('--gradcheck_interval', type=int, default=50,
#                         help='Interval in steps at which to report training stats.')

#     # reproducibility
#     parser.add_argument('--seed', type=int, default=1638128,
#                         help='Random seed for PyTorch and NumPy.')

#     # MFG
#     parser.add_argument('--gaussian_multi_dim',     type=int, default=2)
#     parser.add_argument('--gaussian_multi_a',       type=float, default=10.)
#     parser.add_argument('--num_train_data',         type=int, default=10000)
#     parser.add_argument('--num_val_data',           type=int, default=10000)
#     parser.add_argument('--num_test_data',          type=int, default=10000)
#     parser.add_argument('--train_batch_size',       type=int, default=32)
#     parser.add_argument('--val_batch_size',         type=int, default=512)
#     parser.add_argument('--test_batch_size',        type=int, default=512)
#     parser.add_argument('--train_batch_size_inner', type=int, default=32)
#     parser.add_argument('--val_batch_size_inner',   type=int, default=512)
#     parser.add_argument('--test_batch_size_inner',  type=int, default=512)
#     parser.add_argument('--lbd_OT',                 type=float, default=0)
#     parser.add_argument('--lbd_F',                  type=float, default=0)
#     parser.add_argument('--lbd_F_E',                type=float, default=0.01)
#     parser.add_argument('--lbd_F_P',                type=float, default=1)
#     parser.add_argument('--reg_OT_dir',             type=str, default='gen', choices=['gen', 'norm'])
#     parser.add_argument('--OT_comp',                type=str, default='trajectory', choices=['trajectory', 'monge'])
#     parser.add_argument('--OT_part',                type=str, default='module', choices=['block', 'block_CL_no_perm', 'module'])
#     parser.add_argument('--interaction',            type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--LU_last',                type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--NF_loss',                type=str, default='KL_sampling', choices=[
#                                                     'KL_sampling', 'KL_density', 'jeffery'])
#     parser.add_argument('--val_score',              type=str, default='loss', choices=[
#                                                     'loss', 'L', 'G', 'F'])
#     parser.add_argument('--mixture_base',           type=str, default='gaussian', choices=[
#                                                     'gaussian', 'gaussian_mixture'])
#     parser.add_argument('--mixture_weight',         type=str, default='identical', choices=[
#                                                     'identical', 'undersample_one'])  
#     parser.add_argument('--F_ld_weight',            type=str, default='identical', choices=['identical'])                                                                          
#     parser.add_argument('--disc_scheme',            type=str, default='forward', choices=[
#                                                     'forward', 'centered', 'forward_2nd',
#                                                     'FD4_simp', 'FD1_simp', 'FD4_simp_symmetric'])
#     parser.add_argument('--NF_model',               type=str, default='default', choices=[
#                                                     'default', 'single_flow'])
#     parser.add_argument('--interp_hist',        type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--n_interp',           type=int, default=5, 
#                         help='Number of interpolated points inserted between flow points to better approximate MFG costs')

#     # optimization
#     parser.add_argument('--num_training_steps', type=int, default=500, help='Number of total training steps in the outer loop.')
#     parser.add_argument('--l2_reg_NF', type=float, default=0)
#     parser.add_argument('--l2_reg_inner', type=float, default=0)
#     parser.add_argument('--l2_reg_obs', type=float, default=0)
#     parser.add_argument('--optimizer_NF', type=str, default='adam', choices=['adam', 'sgd'])
#     parser.add_argument('--optimizer_obs', type=str, default='adam', choices=['adam', 'sgd'])
#     parser.add_argument('--optimizer_inner', type=str, default='sgd', choices=['adam', 'sgd'])
#     parser.add_argument('--scheduler_obs', type=str, default='adaptive', choices=['step', 'cyclic', 'adaptive'])
#     parser.add_argument('--scheduler_NF', type=str, default='adaptive', choices=['cyclic', 'adaptive'])
#     parser.add_argument('--scheduler_inner', type=str, default='none', choices=['none', 'cyclic'])
#     parser.add_argument('--step_lr_size', type=int, default=50)
#     parser.add_argument('--step_lr_gamma', type=float, default=0.8)
#     parser.add_argument('--adam_beta1_obs', type=float, default=0.9)
#     parser.add_argument('--adam_beta2_obs', type=float, default=0.999)
#     parser.add_argument('--lr_LL_min', type=float, default=1e-7)
    
#     # Bilevel MFG
#     # parser.add_argument('--bilevel_training_data_dir', type=str, default='./results/crowd_motion_gaussian/NSF_CL_crowd_motion_2D_Jeff_OT_1e-1_F_2e-1_B=2048_lr=1e-3_N=50k_FD4Simp/train_traj_crowd_motion_gaussian.pt')
#     # parser.add_argument('--bilevel_training_data_dir', type=str, default='./results/crowd_motion_gaussian/NSF_CL_2D_N=1M_OT_1e-1_F_1e0/train_traj_crowd_motion_gaussian.pt')                            
#     parser.add_argument('--h_obs',                     type=int, default=64, help='hidden dimension in the NN parametrizing the obstacle.')
#     parser.add_argument('--l_obs',                     type=int, default=1, help='number of hidden layers in the NN parametrizing the obstacle.')
#     parser.add_argument('--act_obs',                   type=str, default='relu', choices=
#                                                         ['relu', 'softplus', 'tanh', 'elu', 'leaky_relu'])
#     parser.add_argument('--act_obs_out',               type=str, default='none', choices=
#                                                         ['none', 'exp', 'relu', 'sqr'])
#     parser.add_argument('--softplus_beta',             type=float, default=1.)
#     parser.add_argument('--res_link_obs',              type=lambda x: (str(x).lower() == 'true'), default=True)
#     parser.add_argument('--lr_NF',                     type=float, default=3e-4, help='Learning rate for NF.')
#     parser.add_argument('--lr_obs',                    type=float, default=1e-2, help='Learning rate for the obstacle.')
#     parser.add_argument('--num_training_steps_inner',  type=int, default=3, help='Number of total training steps in the inner loop (lower problem)')
#     parser.add_argument('--step_size_inner',           type=float, default=1e-5, help='beta for the inner loop')
#     parser.add_argument('--step_decay_inner',          type=str, default='none', choices=
#                                                         ['none', 'sqrt'], help='Decay mode for the lower problem step size, range: [0, inf], 0 = no decay')
#     parser.add_argument('--grad_clip_NF',              type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--grad_clip_obs',             type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--grad_clip_value_NF',        type=float, default=5., help='Value by which to clip norm of gradients.')
#     parser.add_argument('--grad_clip_value_obs',       type=float, default=5., help='Value by which to clip norm of gradients.')
#     parser.add_argument('--patience_obs',              type=int, default=10)
#     parser.add_argument('--patience_NF',               type=int, default=100)
#     parser.add_argument('--verbose_logging',           type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--NF_update_interval',        type=int, default=1, help='Update NF params every x outer iterates')

#     # misc.
#     parser.add_argument('--num_batch_to_plot',  type=int, default=1)
#     parser.add_argument('--load_pretrain_NF',   type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--load_pretrain_obs',  type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--compute_lip_bound',  type=lambda x: (str(x).lower() == 'true'), default=False)
#     parser.add_argument('--syn_noise',          type=float, default=0.1)
#     parser.add_argument('--marker_size',        type=float, default=5)
#     parser.add_argument('--color',              type=str, default='order', choices=['order', 'radius'])
#     parser.add_argument('--tabular_subset',     type=lambda x: (str(x).lower() == 'true'), default=False)

#     # debugging
#     parser.add_argument('--debug_obs_init', type=str, default='true', choices=['true', 'rand', 'close'])
#     parser.add_argument('--debug_opt',      type=str, default='default', choices=
#                                                 ['default', 'no_truncate'])
#     parser.add_argument('--debug_obs',      type=str, default='NN', choices=
#                                             ['NN', 'gaussian'])
#     parser.add_argument('--debug_step_NF',  type=lambda x: (str(x).lower() == 'true'), default=True)
#     parser.add_argument('--debug_tensor_type',  type=str, default='float', choices=['float', 'double'])

#     args = parser.parse_args()
#     args = sanitize_args(args)

#     return args

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


if args.dataset_name == 'crowd_motion_gaussian_bilevel':
    args.pretrain_obs_dir = './results/crowd_motion_gaussian_bilevel/pretrain_obs.t'
    args.pretrain_NF_dir = './results/crowd_motion_gaussian/2D_iterative_rerun/crowd_motion_gaussian-best-val-2D_iterative_rerun.t'
    args.bilevel_training_data_dir = './results/crowd_motion_gaussian/NSF_CL_crowd_motion_2D_Jeff_OT_1e-1_F_2e-1_B=2048_lr=1e-3_N=50k_FD4Simp/train_traj_crowd_motion_gaussian.pt'
    X_train, _, _, train_loader, val_loader, test_loader, P_1 = load_crowd_motion_gaussian_bilevel_data(\
        args.gaussian_multi_dim, args.bilevel_training_data_dir, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    train_generator = data_.batch_generator(train_loader)
    test_generator  = data_.batch_generator(test_loader)
    test_batch      = next(iter(train_loader)).to(device)
    features        = args.gaussian_multi_dim
elif args.dataset_name == 'crowd_motion_gaussian_bilevel_strong':
    args.pretrain_obs_dir = './results/crowd_motion_gaussian_bilevel/pretrain_obs.t'
    args.pretrain_NF_dir = './results/crowd_motion_gaussian/NSF_CL_2D_N=1M_OT_1e-1_F_1e0/crowd_motion_gaussian-best-val-NSF_CL_2D_N=1M_OT_1e-1_F_1e0.t'
    args.bilevel_training_data_dir = './results/crowd_motion_gaussian/NSF_CL_2D_N=1M_OT_1e-1_F_1e0/train_traj_crowd_motion_gaussian.pt'
    X_train, _, _, train_loader, val_loader, test_loader, P_1 = load_crowd_motion_gaussian_bilevel_data(\
        args.gaussian_multi_dim, args.bilevel_training_data_dir, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    train_generator = data_.batch_generator(train_loader)
    test_generator  = data_.batch_generator(test_loader)
    test_batch      = next(iter(train_loader)).to(device)
    features        = args.gaussian_multi_dim
else:
    raise NotImplementedError()


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
if args.dataset_name in ['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong']:
    # width = 4
    width = 2
    n_pts = 100
else:
    raise NotImplementedError()
grid, grid_pad, grid_x, grid_y = create_grid(width, n_pts, features)
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

# base dist (P_0)
if args.dataset_name in ['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong']:
    e_2    = torch.zeros(args.gaussian_multi_dim).to(device)
    e_2[1] = 1.
    mean   = 3*e_2
    cov    = 0.3 * torch.eye(args.gaussian_multi_dim).to(device)
    P_0    = distributions.MultivarNormal((features,), mean=mean, cov=cov)
elif args.dataset_name == 'gaussian':
    cov  = 0.01 * torch.eye(args.gaussian_multi_dim).to(device)
    P_0  = distributions.MultivarNormal((features,), cov=cov)
else:
    P_0 = distributions.StandardNormal((features,))

# flows
transform, K = create_transform_iterative()
NF = NF_iterative_flatVar(transform, P_0, K).to(device)
num_params_NF = utils.get_num_parameters(NF)
print('There are {} trainable parameters in the flow model.'.format(num_params_NF))

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
optim_to_use_NF    = optimizer_dict[args.optimizer_NF]
optim_to_use_obs   = optimizer_dict[args.optimizer_obs]
optim_to_use_inner = optimizer_dict[args.optimizer_inner]

if args.optimizer_obs == 'adam':
    optimizer_obs = optim_to_use_obs(obstacle.parameters(), lr=args.lr_obs, weight_decay=args.l2_reg_obs, \
                                    betas=(args.adam_beta1_obs, args.adam_beta2_obs))
else:
    optimizer_obs = optim_to_use_obs(obstacle.parameters(), lr=args.lr_obs, weight_decay=args.l2_reg_obs)
optimizer_inner = optim_to_use_inner(NF.parameters(), lr=args.step_size_inner, weight_decay=args.l2_reg_inner)
optimizer_NF    = optim_to_use_NF(NF.parameters(), lr=args.lr_NF, weight_decay=args.l2_reg_NF)

# scheduler
if args.scheduler_obs == 'adaptive':
    scheduler_obs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_obs, mode='min', factor=0.5, verbose=True, patience=args.patience_obs)
elif args.scheduler_obs == 'step':
    scheduler_obs = optim.lr_scheduler.StepLR(optimizer_obs, step_size=args.step_lr_size, gamma=args.step_lr_gamma)
elif args.scheduler_obs == 'cyclic':
    scheduler_obs = optim.lr_scheduler.CosineAnnealingLR(optimizer_obs, args.num_training_steps, 0)

if args.scheduler_NF == 'adaptive':
    scheduler_NF = optim.lr_scheduler.ReduceLROnPlateau(optimizer_NF, mode='min', factor=0.5, verbose=True, patience=args.patience_NF)
elif args.scheduler_NF == 'cyclic':
    scheduler_NF = optim.lr_scheduler.CosineAnnealingLR(optimizer_NF, args.num_training_steps, 0)

if args.scheduler_inner == 'cyclic':
    scheduler_inner = optim.lr_scheduler.CosineAnnealingLR(optimizer_NF, args.num_training_steps_inner, args.lr_LL_min)


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

def grad_FD_randcheck(NF, obstacle, x, x_list, z_list, l_base, g_AD_all, d=1e-6, model_to_check='obs', num_entries_checked=3):
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

        p_copy.data = p_copy.data + d * e.reshape(p_copy.shape)

        # AD grad
        I = e.reshape(p_copy.shape) != 0
        g_AD[n_check] = g_AD_all[k][I]

        # FD grad
        l_perturbed_max = 0
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
            l_perturbed, _ = compute_l(NF_copy, x)

            if l_perturbed > l_perturbed_max:
                l_perturbed_max = l_perturbed

        g_FD[n_check] = (l_perturbed_max - l_base) / d


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
    log_density = torch.mean(log_density)

    # backward
    z_K, ld_gen, _, hist_gen, hist_ld_gen = NF.inverse(z)

    # distribution matching / terminal cost (G)
    KL_density  = torch.Tensor([0]).to(device)
    KL_sampling = torch.Tensor([0]).to(device)
    if args.NF_loss == 'KL_sampling':
        G_cost = -log_density
        if P_1 is not None:
            log_prob_1  = torch.mean(P_1.log_prob(x))
            KL_sampling = log_prob_1 - log_density
    elif args.NF_loss == 'KL_density':
        assert P_1 is not None
        log_prob_0   = torch.mean(P_0.log_prob(z))
        log_prob_gen = torch.mean(P_1.log_prob(z_K))
        ld_gen       = torch.mean(ld_gen)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = KL_density
    elif args.NF_loss == 'jeffery':
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
        writer.add_scalar(tag='log_density'+ '_' + mode, scalar_value=log_density.item(), global_step=step_inner)
        writer.add_scalar(tag='loss'+ '_' + mode, scalar_value=loss.item(), global_step=step_inner)
        writer.add_scalar(tag='KL_density'+ '_' + mode, scalar_value=KL_density.item(), global_step=step_inner)
        writer.add_scalar(tag='KL_sampling'+ '_' + mode, scalar_value=KL_sampling.item(), global_step=step_inner)
        if args.interaction:
            writer.add_scalar(tag='F_P'+ '_' + mode, scalar_value=F_P.item(), global_step=step_inner)
            writer.add_scalar(tag='F_E'+ '_' + mode, scalar_value=F_E.item(), global_step=step_inner)
            writer.add_scalar(tag='F_cost'+ '_' + mode, scalar_value=F_cost.item(), global_step=step_inner)

    return {'loss': loss, 'G': G_cost, 'L': L_cost, 'F': F_cost}


# load pretrained model if specified (for continued training)
if args.load_pretrain_NF:
    NF.load_state_dict(torch.load(args.pretrain_NF_dir))
    print ("Loaded NF from: {}".format(args.pretrain_NF_dir))
    # for p in NF.parameters():
    #     p.data = p.data + 0.001 * torch.rand_like(p)

if args.load_pretrain_obs:
    obstacle.load_state_dict(torch.load(args.pretrain_obs_dir))
    print ("Loaded obstacle from: {}".format(args.pretrain_obs_dir))
    # for p in obstacle.parameters():
    #     p.data = p.data + 0.001 * torch.rand_like(p)

# logging
B_err_min  = np.inf
step_inner = 0
obs_best = None
NF_best  = None

# main loop
for step in range(args.num_training_steps):
    obstacle.zero_grad()
    NF.zero_grad()

    l_log_list = []
    L_log_list = []
    i_max = 1
    l_max = -np.inf
    x_list = []
    z_list = []

    # training data: a batch of trajectories. Shape: B x K x d
    # x = next(train_generator).to(device)
    if args.debug_tensor_type == 'double':
        x = next(train_generator).to(device).double()
    else:
        x = next(train_generator).to(device)
    with torch.no_grad():
        loss_UL, _ = compute_l(NF, x)
        x_UL = P_1.sample(args.train_batch_size_inner)
        z_UL = P_0.sample(args.train_batch_size_inner)
        L_UL = compute_mfg_obj(NF, obstacle, x_UL, z_UL, args)['loss']

    with higher.innerloop_ctx(NF, optimizer_inner, copy_initial_weights=False) as (NF_inner, NF_inner_opt):
        for i in range(args.num_training_steps_inner):
            # LL objective
            x_i = P_1.sample(args.train_batch_size_inner)
            z_i = P_0.sample(args.train_batch_size_inner)
            # the lower level (MFG) objective
            loss_inner = compute_mfg_obj(NF_inner, obstacle, x_i, z_i, args)['loss']
            # loss_inner_fun = compute_mfg_obj_functional(NF_fun, obstacle, list(NF_inner.parameters(time=-1)), )
            # LL gradient step
            NF_inner_opt.step(loss_inner)

            # UL objective: used for PTT
            # FOR SOME REASON, if we don't do a forward before the inverse (in compute_l), the inverse will be wrong. 
            NF_inner(x[:,0,:])
            loss_outer, hist_gen = compute_l(NF_inner, x)

            # loss_outer_fun, hist_gen_fun = compute_l_functional(NF_fun, list(NF_inner.parameters(time=-1)), names, x)
            # NF_copy.zero_grad()
            # loss_inner_cp = compute_mfg_obj(NF_copy, obstacle, x_i, z_i, args)['loss']
            # # LL gradient step
            # loss_inner_cp.backward()
            # optimizer_copy.step()
            # # UL objective: used for PTT
            # loss_outer_cp, hist_gen_cp = compute_l(NF_copy, x)
            
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
        if args.debug_opt == 'default':
            g_NF  = grad(l_max, NF_inner.parameters(time=0), retain_graph=True, allow_unused=True)
            g_obs = grad(l_max, obstacle.parameters(), retain_graph=True, allow_unused=True)
        elif args.debug_opt == 'no_truncate':
            g_NF  = grad(loss_outer, NF_inner.parameters(time=0), retain_graph=True, allow_unused=True)
            g_obs = grad(loss_outer, obstacle.parameters(), retain_graph=True, allow_unused=True)
        else:
            raise NotImplementedError()

    # periodic grad checking
    if (step + 1) % args.gradcheck_interval == 0:
        # gradcheck on random entries
        NF_gradcheck_result, NF_grad_diff   = grad_FD_randcheck(NF, obstacle, x, x_list, z_list, l_max, \
                                                                g_NF, model_to_check='NF')
        obs_gradcheck_result, obs_grad_diff = grad_FD_randcheck(NF, obstacle, x, x_list, z_list, l_max, \
                                                                g_obs, model_to_check='obs')

        print ("Step: {}, NF gradcheck result: {}, norm diff: {:.3f}; Obs gradcheck result: {}, norm diff: {:.3f}".format(\
            step, NF_gradcheck_result, float(NF_grad_diff), obs_gradcheck_result, float(obs_grad_diff)))

    # record computed gradients
    obstacle = set_grad(obstacle, g_obs)
    NF = set_grad(NF, g_NF)

    # UL update
    if (step + 1) % args.NF_update_interval == 0:
        optimizer_NF.step()
    optimizer_obs.step()

    # compute obstacle error
    B_val, B_true_val = eval_obstacles_on_grid(obstacle, B_true, grid, grid_pad) # both are n^2 x 1
    err = 1/2*torch.norm(B_val.reshape(-1) - B_true_val)**2 / B_val.shape[0]
    if err < B_err_min:
        B_err_min = err
        obs_best = copy.deepcopy(obstacle)
        NF_best  = copy.deepcopy(NF)

    # step schedulers
    scheduler_NF.step(loss_outer)
    if args.scheduler_obs == 'adaptive':
        scheduler_obs.step(err)
    elif args.scheduler_obs == 'cyclic':
        scheduler_obs.step(step)
    else:
        scheduler_obs.step()
    if args.scheduler_inner == 'cyclic':
        # scheduler_inner.step(step)
        scheduler_inner.step()

    # monitor and log useful info
    if (step + 1) % args.monitor_interval == 0:
        # log
        l_LL_mean = float(np.mean(l_log_list))
        L_LL_mean = float(np.mean(L_log_list))
        l_gap = l_LL_mean - loss_UL
        l_gap = float(l_gap)
        L_gap = L_LL_mean - L_UL
        L_gap = float(L_gap)
        print ("Step: {}, obstacle error: {:.4f}, l UL: {:.4f}, avg l LL: {:.4f}, l gap: {:.4f}, i_max: {}, L UL: {:.4f}, avg L LL: {:.4f}, L gap: {:.4f}, Obs lr: {}, LL lr: {}".format(\
                step, err, loss_UL, l_LL_mean, l_gap, i_max, L_UL, L_LL_mean, L_gap, optimizer_obs.param_groups[0]['lr'], \
                optimizer_inner.param_groups[0]['lr']))
        if args.verbose_logging:
            grad_norm_NF  = np.sum([float(torch.norm(p)) for p in g_NF if p is not None])
            grad_norm_obs = np.sum([float(torch.norm(p)) for p in g_obs if p is not None])
            print ("Gradient norm on NF: {:.2f}, averaged: {}".format(grad_norm_NF, grad_norm_NF / num_params_NF))
            print ("Gradient norm on obs: {:.2f}, averaged: {}".format(grad_norm_obs, grad_norm_obs / num_params_obs))

        log_names = ['steps', 'lower_UL', 'lower_LL_mean', 'upper_UL', 'upper_LL_mean', \
                    'lower_gap', 'upper_gap', 'B_err']
        log_data  = [step, float(L_UL), l_LL_mean, float(loss_UL), L_LL_mean, \
                    L_gap, l_gap, float(err)]
        plot_logger.log_multiple(log_names, log_data)
        if tbx_logging:
            writer.add_scalar(tag='lower_mean', scalar_value=float(np.mean(L_log_list)), global_step=step)
            writer.add_scalar(tag='upper_mean', scalar_value=float(np.mean(l_log_list)), global_step=step)
            writer.add_scalar(tag='B_err', scalar_value=float(err), global_step=step)
            writer.add_scalar(tag='i_max', scalar_value=i_max, global_step=step)

    if (step + 1) % args.plot_interval == 0:
        # B_val_np = B_val.detach().cpu().numpy().reshape(grid_x.shape)
        # B_true_val_np = B_true_val.detach().cpu().numpy().reshape(grid_x.shape)
        # fig = plt.figure(figsize=(10, 4))
        # sub_fig1 = fig.add_subplot(121)
        # vmin = np.min(np.minimum(B_val_np, B_true_val_np))
        # vmax = np.max(np.maximum(B_val_np, B_true_val_np))
        # pylab.pcolor(grid_x, grid_y, B_val_np, shading='auto', vmin=vmin, vmax=vmax)
        # sub_fig2 = fig.add_subplot(122)
        # pylab.pcolor(grid_x, grid_y, B_true_val_np, shading='auto', vmin=vmin, vmax=vmax)

        # sub_fig1.set_title("Parametrized Obstacle")
        # sub_fig1.set_xlabel('x')
        # sub_fig1.set_ylabel('y')
        # sub_fig2.set_title("True Obstacle")
        # sub_fig2.set_xlabel('x')
        # pylab.colorbar(ax=[sub_fig1, sub_fig2])
        # # save
        # fig_name = 'obs_{}'.format(step+1)
        # save_dir = os.path.join(plot_dir, fig_name)
        # fig.savefig(save_dir)
        # if tbx_logging:
        #     writer.add_figure(fig_name, fig)
        fig_name = 'obs_{}'.format(step+1)
        plot_obstacles(B_val, B_true_val, grid_x, grid_y, plot_dir, writer, tbx_logging, fig_name)

# logging
print ("Exp name: {}".format(args.exp_name))
print ("Training finished, best obstacle error: {:.2f}".format(B_err_min))

# save best models
obs_best_path = os.path.join(model_dir, 'obs_best.t')
NF_best_path  = os.path.join(model_dir, 'NF_best.t')
obs_last_path = os.path.join(model_dir, 'obs_last.t')
NF_last_path  = os.path.join(model_dir, 'NF_last.t')
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