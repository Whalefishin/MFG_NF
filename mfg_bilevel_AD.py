import argparse
import json
import numpy as np
import torch
import os, sys, shutil

from tensorboardX import SummaryWriter
from time import sleep
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm
import torch.distributions as D
from torch.autograd.functional import vhp, vjp, jacobian
from torch.autograd import grad

import data as data_
import nn as nn_
import utils
from mfp_utils import *

from experiments import cutils
from nde import distributions, flows, transforms
from mfg_bilevel_models import *
import copy


def parse_arguments():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--exp_name', type=str, default='1')
    parser.add_argument('--dataset_name', type=str, default='crowd_motion_gaussian_bilevel',
                        choices=['crowd_motion_gaussian_bilevel'],
                        help='Name of dataset to use.')
    # parser.add_argument('--train_batch_size', type=int, default=64,
    #                     help='Size of batch used for training.')
    parser.add_argument('--val_frac', type=float, default=1.,
                        help='Fraction of validation set to use.')
    # parser.add_argument('--val_batch_size', type=int, default=512,
    #                     help='Size of batch used for validation.')

    # optimization
    parser.add_argument('--num_training_steps', type=int, default=500,
                        help='Number of total training steps in the outer loop.')
    parser.add_argument('--lr_schedule', type=str, default='cyclic',
                        choices=['none', 'cyclic', 'adaptive'])
    parser.add_argument('--lbd_reg', type=float, default=0)
    parser.add_argument('--optimizer_NF', type=str, default='adam',
                        choices=['adam', 'sgd'])
    parser.add_argument('--optimizer_obs', type=str, default='adam',
                        choices=['adam', 'sgd'])

    # flow details
    parser.add_argument('--base_transform_type', type=str, default='rq-autoregressive',
                        choices=['affine-coupling', 'quadratic-coupling', 'rq-coupling',
                                'affine-autoregressive', 'quadratic-autoregressive',
                                'rq-autoregressive'],
                        help='Type of transform to use between linear layers.')
    parser.add_argument('--linear_transform_type', type=str, default='lu',
                        choices=['permutation', 'lu', 'svd', 'lu_no_perm'],
                        help='Type of linear transform to use.')
    parser.add_argument('--num_flow_steps', type=int, default=10,
                        help='Number of blocks to use in flow.')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='Number of hidden features to use in coupling/autoregressive nets.')
    parser.add_argument('--tail_bound', type=float, default=3,
                        help='Box is on [-bound, bound]^2')
    parser.add_argument('--num_bins', type=int, default=8,
                        help='Number of bins to use for piecewise transforms.')
    parser.add_argument('--num_transform_blocks', type=int, default=2,
                        help='Number of blocks to use in coupling/autoregressive nets.')
    parser.add_argument('--use_batch_norm', type=int, default=0,
                        choices=[0, 1],
                        help='Whether to use batch norm in coupling/autoregressive nets.')
    parser.add_argument('--dropout_probability', type=float, default=0.25,
                        help='Dropout probability for coupling/autoregressive nets.')
    parser.add_argument('--apply_unconditional_transform', type=int, default=1,
                        choices=[0, 1],
                        help='Whether to unconditionally transform \'identity\' '
                            'features in coupling layer.')
    parser.add_argument('--base_net_act', type=str, default='relu',
                        choices=['relu', 'tanh'])

    # logging and checkpoints
    parser.add_argument('--monitor_interval', type=int, default=250,
                        help='Interval in steps at which to report training stats.')

    # reproducibility
    parser.add_argument('--seed', type=int, default=1638128,
                        help='Random seed for PyTorch and NumPy.')

    # MFG
    parser.add_argument('--gaussian_multi_dim',     type=int, default=2)
    parser.add_argument('--gaussian_multi_a',       type=float, default=10.)
    parser.add_argument('--num_train_data',         type=int, default=10000)
    parser.add_argument('--num_val_data',           type=int, default=10000)
    parser.add_argument('--num_test_data',          type=int, default=10000)
    parser.add_argument('--train_batch_size',       type=int, default=32)
    parser.add_argument('--val_batch_size',         type=int, default=512)
    parser.add_argument('--test_batch_size',        type=int, default=512)
    parser.add_argument('--train_batch_size_inner', type=int, default=32)
    parser.add_argument('--val_batch_size_inner',   type=int, default=512)
    parser.add_argument('--test_batch_size_inner',  type=int, default=512)
    parser.add_argument('--lbd_OT',                 type=float, default=0)
    parser.add_argument('--lbd_F',                  type=float, default=0)
    parser.add_argument('--lbd_F_E',                type=float, default=0.01)
    parser.add_argument('--lbd_F_P',                type=float, default=1)
    parser.add_argument('--reg_OT_dir',             type=str, default='gen', choices=['gen', 'norm'])
    parser.add_argument('--OT_comp',                type=str, default='trajectory', choices=['trajectory', 'monge'])
    parser.add_argument('--OT_part',                type=str, default='module', choices=['block', 'block_CL_no_perm', 'module'])
    parser.add_argument('--interaction',            type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--LU_last',                type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--NF_loss',                type=str, default='KL_sampling', choices=[
                                                    'KL_sampling', 'KL_density', 'jeffery'])
    parser.add_argument('--val_score',              type=str, default='loss', choices=[
                                                    'loss', 'L', 'G', 'F'])
    parser.add_argument('--mixture_base',           type=str, default='gaussian', choices=[
                                                    'gaussian', 'gaussian_mixture'])
    parser.add_argument('--mixture_weight',         type=str, default='identical', choices=[
                                                    'identical', 'undersample_one'])  
    parser.add_argument('--F_ld_weight',            type=str, default='identical', choices=['identical'])                                                                          
    parser.add_argument('--disc_scheme',            type=str, default='forward', choices=[
                                                    'forward', 'centered', 'forward_2nd',
                                                    'FD4_simp', 'FD1_simp', 'FD4_simp_symmetric'])
    parser.add_argument('--NF_model',               type=str, default='default', choices=[
                                                    'default', 'single_flow'])

    # Bilevel MFG
    parser.add_argument('--bilevel_training_data_dir', type=str, default='./results/crowd_motion_gaussian/NSF_CL_crowd_motion_2D_Jeff_OT_1e-1_F_2e-1_B=2048_lr=1e-3_N=50k_FD4Simp/train_traj_crowd_motion_gaussian.pt')                            
    parser.add_argument('--h_obs',                     type=int, default=128, help='hidden dimension in the NN parametrizing the obstacle.')
    parser.add_argument('--l_obs',                     type=int, default=0, help='number of hidden layers in the NN parametrizing the obstacle.')
    parser.add_argument('--act_obs',                   type=str, default='relu', choices=
                                                        ['relu', 'softplus', 'tanh', 'elu', 'leaky_relu'])
    parser.add_argument('--res_link_obs',              type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--lr_NF',                     type=float, default=3e-4, help='Learning rate for NF.')
    parser.add_argument('--lr_obs',                    type=float, default=1e-2, help='Learning rate for the obstacle.')
    parser.add_argument('--num_training_steps_inner',  type=int, default=3, help='Number of total training steps in the inner loop (lower problem)')
    parser.add_argument('--step_size_inner',           type=float, default=1e-4, help='beta for the inner loop')
    parser.add_argument('--step_decay_inner',          type=str, default='none', choices=
                                                        ['none', 'sqrt'], help='Decay mode for the lower problem step size, range: [0, inf], 0 = no decay')
    parser.add_argument('--grad_update_inner',         type=str, default='none', choices=
                                                        ['sgd', 'adam'], help='Optimization method used in the lower level problem.')
    parser.add_argument('--clip_grad_norm_NF',         type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--clip_grad_norm_obs',        type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--grad_norm_clip_value_NF',   type=float, default=5., help='Value by which to clip norm of gradients.')
    parser.add_argument('--grad_norm_clip_value_obs',  type=float, default=5., help='Value by which to clip norm of gradients.')
    parser.add_argument('--patience_obs',              type=int, default=10)
    parser.add_argument('--patience_NF',               type=int, default=10)
    parser.add_argument('--verbose_logging',           type=lambda x: (str(x).lower() == 'true'), default=False)

    # misc.
    parser.add_argument('--plotting_subset',    type=int, default=10000)
    parser.add_argument('--load_best_val',      type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--compute_lip_bound',  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--syn_noise',          type=float, default=0.1)
    parser.add_argument('--marker_size',        type=float, default=5)
    parser.add_argument('--color',              type=str, default='order', choices=['order', 'radius'])
    parser.add_argument('--tabular_subset',     type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--obs_debug_init',     type=str, default='true', choices=['true', 'rand'])

    args = parser.parse_args()
    args = sanitize_args(args)

    return args

args = parse_arguments()




# =================================================================================== #
#                                       Meta                                          #
# =================================================================================== #

os.environ['DATAROOT']     = 'experiments/dataset/data/'
os.environ['SLURM_JOB_ID'] = '1'

torch.manual_seed(args.seed)
np.random.seed(args.seed)

assert torch.cuda.is_available()
device = torch.device('cuda')
device_cpu = torch.device('cpu')
# torch.set_default_tensor_type(torch.DoubleTensor)


if args.dataset_name == 'crowd_motion_gaussian_bilevel':
    X_train, _, _, train_loader, val_loader, test_loader, P_1 = load_crowd_motion_gaussian_bilevel_data(\
        args.gaussian_multi_dim, args.bilevel_training_data_dir, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    train_generator = data_.batch_generator(train_loader)
    test_batch      = next(iter(train_loader)).to(device)
    features        = args.gaussian_multi_dim
else:
    raise NotImplementedError()


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

# base dist (P_0)
if args.dataset_name == 'crowd_motion_gaussian_bilevel':
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

# create flows
transform, K = create_transform()
flow = flows.Flow(transform, P_0).to(device)
num_params_NF = utils.get_num_parameters(flow)
print('There are {} trainable parameters in the flow model.'.format(num_params_NF))

# create obstacle
# ground truth obstacle
B_true = Obstacle_true(args, device)
# NN-parametrized obstacle
obstacle = Obstacle(features, args).to(device)
# obstacle = Obstacle_gaussian(init=args.obs_debug_init).to(device)
num_params_obs = utils.get_num_parameters(obstacle)
print('There are {} trainable parameters in the obstacle model.'.format(num_params_obs))

# container for NF + obstacle, used for vhp computations
NF_obstacle = NF_Obstacle(flow, obstacle)

# create optimizer
# optimizer = optim.Adam(NF_obstacle.parameters(), lr=args.learning_rate, weight_decay=args.lbd_reg)

# optimizer = optim.Adam(
#     [
#         {"params": NF_obstacle.NF.parameters(), "lr": args.lr_NF},
#         {"params": NF_obstacle.obstacle.parameters(), "lr": args.lr_obs}
#     ],
#     weight_decay=args.lbd_reg
#     )

# if args.lr_schedule == 'cyclic':
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
# elif args.lr_schedule == 'adaptive':
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
# else:
#     scheduler = None

optim_to_use_NF  = {'adam': optim.Adam, "sgd": optim.SGD}[args.optimizer_NF]
optim_to_use_obs = {'adam': optim.Adam, "sgd": optim.SGD}[args.optimizer_obs]
optimizer_NF  = optim_to_use_NF(NF_obstacle.NF.parameters(), lr=args.lr_NF, weight_decay=args.lbd_reg)
optimizer_obs = optim_to_use_obs(NF_obstacle.obstacle.parameters(), lr=args.lr_obs, weight_decay=args.lbd_reg)
scheduler_obs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_obs, mode='min', factor=0.5, verbose=True, patience=args.patience_obs)
scheduler_NF  = optim.lr_scheduler.ReduceLROnPlateau(optimizer_NF, mode='min', factor=0.5, verbose=True, patience=args.patience_NF)

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
if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)
plot_logger = Plot_logger(root_path=plot_data_dir)

tbar = tqdm(range(args.num_training_steps))
best_val_score = 1e10

# grid to evaluate obstacle on
if args.dataset_name == 'crowd_motion_gaussian_bilevel':
    # width = 4
    width = 1
    n_pts = 50
else:
    raise NotImplementedError()
grid, grid_pad, grid_x, grid_y = create_grid(width, n_pts, features)
grid = grid.to(device)
grid_pad = grid_pad.to(device)

# =================================================================================== #
#                                      Training                                       #
# =================================================================================== #


def compute_l(model, theta, names, x):
    # x: B x K x d

    # load params into the network
    load_weights(model, names, theta, as_params=False, device=x.device)
    # compute upper level objective at the specified parameter values
    B = x.shape[0]
    _, _, _, hist_gen, _ = model.NF._transform.inverse(x[:,0,:])
    hist_gen = partition_hist(hist_gen, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)
    F = hist_gen.reshape(B,-1) # B x K*d
    x_flat = x.reshape(B, -1)
    l_val = 1/2 * torch.mean(torch.sum((F - x_flat)**2, dim=-1))

    return l_val


def compute_mfg_obj(model, params, names, x, z, args, mode='train'):
    # load params into the network
    # _, names = make_functional(model) # TODO: see if this is necessary
    load_weights(model, names, params, as_params=False, device=x.device)

    # forward
    log_density, _, _, hist_norm, _, _, _ = model.NF.log_prob(x)
    log_density = torch.mean(log_density)

    # backward
    # z_K, ld_gen, OT_cost_gen, hist_gen, hist_ld_gen, z_0 = flow.sample(args.train_batch_size)
    z_K, ld_gen, _, hist_gen, hist_ld_gen = model.NF._transform.inverse(z)

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
        # z_K, ld_gen, _, _, z_0 = flow.sample(args.train_batch_size)
        log_prob_0   = torch.mean(P_0.log_prob(z))
        log_prob_gen = torch.mean(P_1.log_prob(z_K))
        ld_gen       = torch.mean(ld_gen)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = KL_density
    elif args.NF_loss == 'jeffery':
        # loss = -log_density
        assert P_1 is not None
        # z_K, ld_gen, _, _, _ = flow.sample(args.train_batch_size)
        log_prob_0   = torch.mean(P_0.log_prob(z))
        log_prob_gen = torch.mean(P_1.log_prob(z_K))
        ld_gen       = torch.mean(ld_gen)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = -log_density + KL_density
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
                                        scheme=args.disc_scheme)
        L_cost = torch.mean(L_cost)

    # interaction (F)
    F_P    = torch.Tensor([0]).to(device)
    F_E    = torch.Tensor([0]).to(device)
    F_cost = torch.Tensor([0]).to(device)
    if args.interaction:
        hist_part    = partition_hist(hist_gen, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last)
        hist_ld_part = partition_hist(hist_ld_gen, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, hist_type='ld')
        log_prob_0   = P_0.log_prob(z)
        F_E, F_P     = compute_F(args, model.obstacle, hist_part, log_prob_0, hist_ld_part, \
                                Q_is_dist=False, scheme=args.F_disc_scheme)
        F_E          = torch.mean(F_E)
        F_P          = torch.mean(F_P)
        F_cost       = args.lbd_F_E * F_E + args.lbd_F_P * F_P

    # Overall loss
    loss = G_cost + args.lbd_OT * L_cost + args.lbd_F * F_cost

    # if tbx_logging:
    #     writer.add_scalar(tag='OT_cost' + '_' + mode, scalar_value=L_cost.item(), global_step=step)
    #     writer.add_scalar(tag='log_density'+ '_' + mode, scalar_value=log_density.item(), global_step=step)
    #     writer.add_scalar(tag='loss'+ '_' + mode, scalar_value=loss.item(), global_step=step)
    #     writer.add_scalar(tag='KL_density'+ '_' + mode, scalar_value=KL_density.item(), global_step=step)
    #     writer.add_scalar(tag='KL_sampling'+ '_' + mode, scalar_value=KL_sampling.item(), global_step=step)
    #     if args.interaction:
    #         writer.add_scalar(tag='F_P'+ '_' + mode, scalar_value=F_P.item(), global_step=step)
    #         writer.add_scalar(tag='F_E'+ '_' + mode, scalar_value=F_E.item(), global_step=step)
    #         writer.add_scalar(tag='F_cost'+ '_' + mode, scalar_value=F_cost.item(), global_step=step)

    # # logging to .txt
    # if mode == 'train':
    #     # only do this for training. Logging for validation data is done in the main loop.
    #     plot_logger.log(mode + '_steps', step)
    #     plot_logger.log(mode + '_NLL', float(-log_density))

    # return loss, G_cost, L_cost, F_cost
    return {'loss': loss, 'G': G_cost, 'L': L_cost, 'F': F_cost}

# load pretrained model if specified (for continued training)
if args.load_best_val:
    path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
    flow.load_state_dict(torch.load(path))
    print ("Loaded model from: {}".format(path))


# x = next(train_generator).to(device)

# main loop
for step in tbar:
    NF_obstacle.train()
    # pure function container for the NN
    NF_obstacle_fun = copy.deepcopy(NF_obstacle)

    # intermediate variables used 
    # x_list = []
    # z_list = []
    # params_list = []
    # step_list = []
    l_list = []
    # for logging
    l_log_list = []
    lower_loss_list = []
    
    x = next(train_generator).to(device) # A batch of trajectories. Shape: B x K x d
    
    # upper level obj value
    params_0, names = make_functional(NF_obstacle_fun, device=device)
    # params_0 = tuple([p.cpu().requires_grad_() for p in NF_obstacle_fun.parameters()])
    theta_0, phi_0 = params_0[:NF_obstacle_fun.param_len_flow], params_0[NF_obstacle_fun.param_len_flow:]

    # optimizer_fun = torch.optim.SGD(theta_0, lr=args.step_size_inner)

    l_val = compute_l(NF_obstacle_fun, theta_0, names[:NF_obstacle_fun.param_len_flow], x)
    l_val_max = l_val
    i_max = 0
    l_list.append(l_val)
    # TODO: currently this gives a copy of the tensors and breaks the AD, 
    # i.e., grad(l_val, theta_0 + phi_0, allow_unused=True, retain_graph=True) returns all None
    # params_list.append(tuple_to_device(theta_0 + phi_0, device_cpu))
    # params_list.append(theta_0 + phi_0)

    theta = theta_0
    # lower level updates
    for i in range(args.num_training_steps_inner):
        if args.step_decay_inner == 'sqrt':
            beta = args.step_size_inner / np.sqrt(i+1)
        else:
            beta = args.step_size_inner
        # sample a batch of training data
        x_i = P_1.sample(args.train_batch_size_inner)
        z_i = P_0.sample(args.train_batch_size_inner)
        # the lower level (MFG) objective
        loss = compute_mfg_obj(NF_obstacle_fun, theta + phi_0, names, x_i, z_i, args)['loss']
        
        g_lower = grad(loss, theta, create_graph=True, allow_unused=True)
        # g_lower = get_update_from_grad(g_lower, method=args.grad_update_inner) 
        theta   = tuple_tensor_update(theta, g_lower, step_size=beta)
        # NOTE: the problem with below is that step() updates weights in-place, and the outerloop grad does not like it
        # loss.backward(create_graph=True)
        # optimizer_fun.step()
        # optimizer_fun.zero_grad()

        # record the param that incurs the largest upper level obj.
        l_val = compute_l(NF_obstacle_fun, theta, names[:NF_obstacle_fun.param_len_flow], x)
        if l_val > l_val_max:
            l_val_max = l_val
            i_max = i+1

        # x_list.append(x_i.detach().clone())
        # z_list.append(z_i.detach().clone())
        # params_list.append(tuple_to_device(theta + phi_0, device_cpu))
        # params_list.append(theta + phi_0)
        # step_list.append(beta)
        lower_loss_list.append(float(loss))
        l_list.append(l_val)
        l_log_list.append(float(l_val))

    # we now compute the upper level gradients, including
    # gradients of l wrt theta that parametrizes the NF/agents; 
    # and gradients of l wrt phi that parametrizes the obstacle

    # l_max = l_list[i_max]
    # NOTE: the above is the same as below
    # l_max = compute_l(NF_obstacle_fun, params_list[i_max], names[:NF_obstacle_fun.param_len_flow], x)
    g = grad(l_val_max, theta_0 + phi_0, allow_unused=True, retain_graph=True)

    NF_obstacle = set_grad(NF_obstacle, g)

    if args.clip_grad_norm_NF:
        clip_grad_norm_(NF_obstacle.NF.parameters(), args.grad_norm_clip_value_NF)
    if args.clip_grad_norm_obs:
        clip_grad_norm_(NF_obstacle.obstacle.parameters(), args.grad_norm_clip_value_obs)

    optimizer_NF.step()
    optimizer_obs.step()
    NF_obstacle.zero_grad()

    # monitor and log useful info
    if (step + 1) % args.monitor_interval == 0:
        # see if the parametrized obstacle is getting closer to the real one after each outer iterate
        B_val, B_true_val = eval_obstacles_on_grid(NF_obstacle.obstacle, B_true, grid, grid_pad) # both are n^2 x 1
        err = 1/2*torch.norm(B_val.reshape(-1) - B_true_val)**2 / B_val.shape[0]
        # err = torch.norm(B_val.reshape(-1) - B_true_val)**2
        
        # log
        # print ("Step: {}, obstacle error: {:.2f}, l: {:.2f}, i_max: {}, lower loss: {}, avg lower loss: {:.2f}, l_list: {}, avg l: {:.2f}".format(\
        #         step, err, l_val, i_max, lower_loss_list, np.mean(lower_loss_list), l_log_list, np.mean(l_log_list)))
        print ("Step: {}, obstacle error: {:.2f}, l: {:.2f}, i_max: {}, avg lower loss: {:.2f}, avg l: {:.2f}".format(\
                step, err, l_val, i_max, np.mean(lower_loss_list), np.mean(l_log_list)))
        if args.verbose_logging:
            # print ("Gradient norm on NF: {:.2f}".format(np.sum([float(torch.norm(p.grad)) for p in NF_obstacle.NF.parameters() if p.grad is not None])))
            # print ("Gradient norm on obs: {:.2f}".format(np.sum([float(torch.norm(p.grad)) for p in NF_obstacle.obstacle.parameters() if p.grad is not None])))
            grad_norm_NF  = np.sum([float(torch.norm(p)) for p in g[:NF_obstacle_fun.param_len_flow] if p is not None])
            grad_norm_obs = np.sum([float(torch.norm(p)) for p in g[NF_obstacle_fun.param_len_flow:] if p is not None])
            print ("Gradient norm on NF: {:.2f}, averaged: {}".format(grad_norm_NF, grad_norm_NF / num_params_NF))
            print ("Gradient norm on obs: {:.2f}, averaged: {}".format(grad_norm_obs, grad_norm_obs / num_params_obs))
            # print ("Mean: {}; Diag: {}".format(NF_obstacle.obstacle.mean.data, NF_obstacle.obstacle.diag.data))
        plot_logger.log('steps', step)
        plot_logger.log('lower_mean', float(np.mean(lower_loss_list)))
        plot_logger.log('upper_mean', float(np.mean(l_log_list)))
        plot_logger.log('B_err', float(err))

        # step schedulers
        scheduler_NF.step(l_val)
        scheduler_obs.step(err)



# =================================================================================== #
#                                     Plotting                                        #
# =================================================================================== #

# save the trained obstacle (evaluated on a regular grid) for plotting
grid_save_path   = os.path.join(plot_data_dir, 'grid.mat')
grid_x_save_path = os.path.join(plot_data_dir, 'grid_x.mat')
grid_y_save_path = os.path.join(plot_data_dir, 'grid_y.mat')
B_save_path      = os.path.join(plot_data_dir, 'B.mat')
B_true_save_path = os.path.join(plot_data_dir, 'B_true.mat')
scipy.io.savemat(grid_save_path,  dict(data=grid.detach().cpu().numpy()))
scipy.io.savemat(grid_x_save_path,  dict(data=grid_x))
scipy.io.savemat(grid_y_save_path,  dict(data=grid_y))
scipy.io.savemat(B_save_path,  dict(data=B_val.reshape(grid_x.shape).detach().cpu().numpy()))
scipy.io.savemat(B_true_save_path, dict(data=B_true_val.reshape(grid_x.shape).detach().cpu().numpy()))




print ("All done.")