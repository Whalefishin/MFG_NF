import argparse
import json
from turtle import width
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

import data as data_
import nn as nn_
import utils
from mfp_utils import *

from experiments import cutils
from nde import distributions, flows, transforms
from multi_mfg_models import NF_list


parser = argparse.ArgumentParser()

# data
parser.add_argument('--exp_name', type=str, default='1')
parser.add_argument('--dataset_name', type=str, default='drones_22',
                    choices=['drones_22', 'drones_23', 'drones_82', 'drones_22_obs'],
                    help='drones_ab = a groups of drones in b dimensions')
# parser.add_argument('--train_batch_size', type=int, default=64,
#                     help='Size of batch used for training.')
parser.add_argument('--val_frac', type=float, default=1.,
                    help='Fraction of validation set to use.')
# parser.add_argument('--val_batch_size', type=int, default=512,
#                     help='Size of batch used for validation.')

# optimization
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate for optimizer.')
parser.add_argument('--num_training_steps', type=int, default=200000,
                    help='Number of total training steps.')
# parser.add_argument('--anneal_learning_rate', type=int, default=1,
#                     choices=[0, 1],
#                     help='Whether to anneal the learning rate.')
parser.add_argument('--lr_schedule', type=str, default='cyclic',
                    choices=['none', 'cyclic', 'adaptive'])
parser.add_argument('--lr_patience', type=float, default=10)
parser.add_argument('--grad_norm_clip_value', type=float, default=5.,
                    help='Value by which to clip norm of gradients.')
parser.add_argument('--lbd_reg', type=float, default=0)

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
parser.add_argument('--monitor_interval', type=int, default=250, help='Interval in steps at which to report training stats.')
parser.add_argument('--num_val_batches', type=int, default=20)
parser.add_argument('--num_test_batches', type=int, default=20)

# reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

# MFG
parser.add_argument('--gaussian_multi_dim', type=int, default=2)
parser.add_argument('--gaussian_multi_a',   type=float, default=10.)
parser.add_argument('--num_train_data',     type=int, default=10000)
parser.add_argument('--num_val_data',       type=int, default=10000)
parser.add_argument('--num_test_data',      type=int, default=10000)
parser.add_argument('--train_batch_size',   type=int, default=32)
parser.add_argument('--val_batch_size',     type=int, default=512)
parser.add_argument('--test_batch_size',    type=int, default=512)
parser.add_argument('--lbd_OT',             type=float, default=0)
parser.add_argument('--lbd_F',              type=float, default=0)
parser.add_argument('--lbd_F_E',            type=float, default=0.01)
parser.add_argument('--lbd_F_P',            type=float, default=1.)
parser.add_argument('--lbd_F_inter',        type=float, default=1.)
parser.add_argument('--lbd_F_obs',          type=float, default=1.)
parser.add_argument('--reg_OT_dir',         type=str, default='gen', choices=['gen', 'norm'])
parser.add_argument('--OT_comp',            type=str, default='trajectory', choices=['trajectory', 'monge'])
parser.add_argument('--OT_part',            type=str, default='module', choices=['block', 'block_CL_no_perm', 'module'])
parser.add_argument('--interaction',        type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--LU_last',            type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--NF_loss',            type=str, default='KL_sampling', choices=[
                                            'KL_sampling', 'KL_density', 'jeffery'])
parser.add_argument('--val_score',          type=str, default='loss', choices=['loss', 'L', 'G', 'F'])                                                                       
parser.add_argument('--disc_scheme',        type=str, default='forward', choices=[
                                            'forward', 'centered', 'forward_2nd',
                                            'FD4_simp', 'FD1_simp', 'FD4_simp_symmetric'])
parser.add_argument('--NF_model',           type=str, default='default', choices=['default', 'single_flow'])                                     
parser.add_argument('--radius_82',          type=float, default=4.0)
parser.add_argument('--var_drones',         type=float, default=0.01)
parser.add_argument('--obs_var_x_22',       type=float, default=0.1)
parser.add_argument('--obs_var_y_22',       type=float, default=0.004)
parser.add_argument('--obs_mean_x_22',      type=float, default=1.5)
parser.add_argument('--obs_mean_y_22',      type=float, default=0.5)
parser.add_argument('--drones_22_obs_mu_y', type=float, default=1)

# misc.
parser.add_argument('--plotting_subset',    type=int, default=10000)
parser.add_argument('--load_best_val',      type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--test_model',         type=str, default='last', choices=['best', 'last'])  
parser.add_argument('--syn_noise',          type=float, default=0.1)
parser.add_argument('--marker_size',        type=float, default=5)
parser.add_argument('--color',              type=str, default='order', choices=['order', 'radius'])
parser.add_argument('--tabular_subset',     type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--plot_traj_thres',    type=float, default=0)

args = parser.parse_args()
args = sanitize_args(args)

# =================================================================================== #
#                                       Meta                                          #
# =================================================================================== #

os.environ['DATAROOT']     = 'experiments/dataset/data/'
os.environ['SLURM_JOB_ID'] = '1'

torch.manual_seed(args.seed)
np.random.seed(args.seed)

assert torch.cuda.is_available()
device = torch.device('cuda')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


# =================================================================================== #
#                                       Dataset                                       #
# =================================================================================== #
target_dist = None
N_p = 1 # number of populations
if args.dataset_name in ['drones_22', 'drones_23', 'drones_82', 'drones_22_obs']:
    N_p      = int(args.dataset_name.split('_')[1][0])
    features = int(args.dataset_name.split('_')[1][1])
    X_train, _, _, train_loader, val_loader, test_loader, target_dist = \
        make_drone_data(args, args.num_train_data, args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size,\
             args.test_batch_size, name=args.dataset_name, radius_82=args.radius_82, var_drones=args.var_drones)
    train_generators = [data_.batch_generator(loader) for loader in train_loader]
    val_generators   = [data_.batch_generator(loader) for loader in val_loader]
    test_generators  = [data_.batch_generator(loader) for loader in test_loader]
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

# base dist
if args.dataset_name == 'drones_22':
    cov          = args.var_drones * torch.eye(features).to(device)
    distribution_1 = distributions.MultivarNormal((features,), mean=torch.Tensor([0,0]).to(device), cov=cov)
    distribution_2 = distributions.MultivarNormal((features,), mean=torch.Tensor([1,0]).to(device), cov=cov)
    dists = [distribution_1, distribution_2]
elif args.dataset_name == 'drones_22_obs':
    cov          = args.var_drones * torch.eye(features).to(device)
    distribution_1 = distributions.MultivarNormal((features,), mean=torch.Tensor([0,0]).to(device), cov=cov)
    distribution_2 = distributions.MultivarNormal((features,), mean=torch.Tensor([1,0]).to(device), cov=cov)
    dists = [distribution_1, distribution_2]
elif args.dataset_name == 'drones_23':
    cov          = args.var_drones * torch.eye(features).to(device)
    distribution_1 = distributions.MultivarNormal((features,), mean=torch.Tensor([0,0,0]).to(device), cov=cov)
    distribution_2 = distributions.MultivarNormal((features,), mean=torch.Tensor([1,0,0]).to(device), cov=cov)
    dists = [distribution_1, distribution_2]
elif args.dataset_name == 'drones_82':
    cov = args.var_drones * torch.eye(features).to(device)
    # + np.pi to make the destination of each cluster to be diametrically across
    angle     = np.pi/4*(torch.arange(8)+1).unsqueeze(1) + np.pi # N_m x 1
    e_1       = torch.zeros(1,features)
    e_1[:,0]  = 1.
    e_2       = torch.zeros(1,features)
    e_2[:,1]  = 1.
    means     = args.radius_82 * (torch.cos(angle)*e_1.repeat(N_p,1) + \
                        torch.sin(angle)*e_2.repeat(N_p,1)).to(device) # N_m x d
    dists = [distributions.MultivarNormal((features,), mean=means[i], cov=cov) for i in range(N_p)]
else:
    raise NotImplementedError()

# create flows
NFs = []
for i in range(N_p):
    transform, K = create_transform()
    NFs.append(flows.Flow(transform, dists[i]).to(device))

n_params = utils.get_num_parameters(NFs[0])
print('There are {} populations, and {} trainable parameters for a single population.'.format(N_p, n_params))
NFs = NF_list(NFs, dists, args)

# create optimizer
optimizer = optim.Adam(NFs.parameters(), lr=args.learning_rate, weight_decay=args.lbd_reg)
if args.lr_schedule == 'cyclic':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
elif args.lr_schedule == 'adaptive':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, \
                                                        verbose=True, patience=args.lr_patience)
else:
    scheduler = None

# obstacle
Q = construct_Q(args, device)


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
            # for filename in glob.glob(log_dir + "/events*"):
            #     os.remove(filename)
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
# best_val_score = -1e10
best_val_score = 1e10

# load pretrained model if specified (for continued training)
if args.load_best_val:
    path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
    NFs.load_state_dict(torch.load(path))
    print ("Loaded model from: {}".format(path))


# =================================================================================== #
#                                      Training                                       #
# =================================================================================== #
def compute_loss(NFs, data, args, mode='train'):
    """

    Args:
        flows (NN): N_p NF models
        data (tensor): N_p x B x d

    """

    # log_density, _, _, hist_norm, _, OT_cost_norm, _ = flows(data)
    log_density, hist_norm = NFs(data) # N_p x B, N_p x B x K x d
    log_density = torch.sum(log_density, dim=0).mean()

    # sample if needed
    if args.sample_in_train:
        z_K, z_0, ld_gen, hist_gen = NFs.sample(args.train_batch_size, device=data.device)

    # distribution matching / terminal cost (G)
    KL_density  = torch.Tensor([0]).to(device)
    KL_sampling = torch.Tensor([0]).to(device)
    if args.NF_loss == 'KL_sampling':
        G_cost = -log_density
        if target_dist is not None:
            log_prob_1  = torch.stack([target_dist[i].log_prob(data[i]) for i in range(hist_norm.shape[0])]) # N_p x B
            log_prob_1  = torch.sum(log_prob_1, dim=0).mean()
            KL_sampling = log_prob_1 - log_density
    elif args.NF_loss == 'KL_density':
        # assert target_dist is not None
        # # z_K, ld_gen, _, _, z_0 = flow.sample(args.train_batch_size)
        # log_prob_0   = torch.mean(distribution.log_prob(z_0))
        # log_prob_gen = torch.mean(target_dist.log_prob(z_K))
        # ld_gen       = torch.mean(ld_gen)
        # KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        # G_cost       = KL_density
        # TODO
        pass
    elif args.NF_loss == 'jeffery':
        # assert target_dist is not None
        # # z_K, ld_gen, _, _, _ = flow.sample(args.train_batch_size)
        # log_prob_0   = torch.mean(distribution.log_prob(z_0))
        # log_prob_gen = torch.mean(target_dist.log_prob(z_K))
        # ld_gen       = torch.mean(ld_gen)
        # KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        # G_cost       = -log_density + KL_density

        log_prob_0   = torch.stack([dists[i].log_prob(z_0[i]) for i in range(hist_norm.shape[0])]) # N_p x B
        log_prob_0   = torch.sum(log_prob_0, dim=0).mean()
        log_prob_gen = torch.stack([target_dist[i].log_prob(z_K[i]) for i in range(hist_norm.shape[0])])
        log_prob_gen = torch.sum(log_prob_gen, dim=0).mean()
        ld_gen       = torch.sum(ld_gen, dim=0).mean()

        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = -log_density + KL_density
    else:
        raise NotImplementedError()

    # OT regularization (L)
    L_cost = torch.Tensor([0]).to(device)
    if args.lbd_OT != 0:
        if args.reg_OT_dir == 'gen':
            # sample a batch from the base to compute the OT cost
            # L_cost = OT_cost_gen
            hist   = hist_gen
        else:
            # L_cost = OT_cost_norm
            hist   = hist_norm # N_p x B x K x d
        
        L_cost, _ = compute_OT_cost_multi(hist, args, mode=args.reg_OT_dir, partition_mode=args.OT_part, LU_last=args.LU_last,
                                            scheme=args.disc_scheme) # N_p x B
        L_cost = torch.sum(L_cost, dim=0).mean() # 1

    # interaction (F)
    # F_P    = torch.Tensor([0])
    # F_E    = torch.Tensor([0])
    F_cost = torch.Tensor([0]).to(device)
    if args.interaction:
        hist_part    = partition_hist(hist_gen, args, mode='gen', \
                        partition_mode=args.OT_part, LU_last=args.LU_last, multi_pop=True) # N_p x B x K x d
        # hist_ld_part = partition_hist(hist_ld_gen, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, hist_type='ld')
        # log_prob_0   = distribution.log_prob(z_0)
        # F_E, F_P     = compute_F(args, Q, hist_part, log_prob_0, hist_ld_part, dataset=args.dataset_name, \
        #                 W=F_weight, scheme=args.F_disc_scheme)
        # F_E          = torch.mean(F_E)
        # F_P          = torch.mean(F_P)
        # F_cost       = args.lbd_F_E * F_E + args.lbd_F_P * F_P
        F_inter, F_obs = compute_F_drones(args, hist_part, Q=Q, dataset=args.dataset_name, scheme=args.F_disc_scheme)
        F_cost = args.lbd_F_inter * F_inter + args.lbd_F_obs * F_obs

    # Overall loss
    loss = G_cost + args.lbd_OT * L_cost + args.lbd_F * F_cost

    # logging with tensorboard
    # TODO: currently overwrites stuff for the validation set
    if tbx_logging:
        writer.add_scalar(tag='OT_cost' + '_' + mode, scalar_value=L_cost.item(), global_step=step)
        writer.add_scalar(tag='log_density'+ '_' + mode, scalar_value=log_density.item(), global_step=step)
        writer.add_scalar(tag='loss'+ '_' + mode, scalar_value=loss.item(), global_step=step)
        writer.add_scalar(tag='KL_density'+ '_' + mode, scalar_value=KL_density.item(), global_step=step)
        writer.add_scalar(tag='KL_sampling'+ '_' + mode, scalar_value=KL_sampling.item(), global_step=step)
        if args.interaction:
            writer.add_scalar(tag='F_inter'+ '_' + mode, scalar_value=F_inter.item(), global_step=step)
            writer.add_scalar(tag='F_obs'+ '_' + mode, scalar_value=F_obs.item(), global_step=step)
            writer.add_scalar(tag='F_cost'+ '_' + mode, scalar_value=F_cost.item(), global_step=step)

    # logging to .txt
    if mode == 'train':
        # only do this for training. Logging for validation data is done in the main loop.
        plot_logger.log(mode + '_steps', step)
        plot_logger.log(mode + '_NLL', float(-log_density))

    # return loss, G_cost, L_cost, F_cost
    return {'loss': loss, 'G': G_cost, 'L': L_cost, 'F': F_cost}


# main loop
for step in tbar:
    NFs.train()
    if args.lr_schedule == 'cyclic':
        scheduler.step(step)
    optimizer.zero_grad()

    # grab data
    batch = torch.stack([next(train_generators[i]).to(device) for i in range(N_p)])

    # loss  = compute_loss(flow, batch, args, mode='train')
    loss_dict = compute_loss(NFs, batch, args, mode='train')
    loss      = loss_dict['loss']

    loss.backward()

    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(NFs.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    # logging
    if (step + 1) % args.monitor_interval == 0:
        NFs.eval()
        with torch.no_grad():
            # compute validation score
            running_val_score = 0
            running_val_LL    = 0
            for _ in range(args.num_val_batches):
                val_batch = torch.stack([next(val_generators[i]).to(device) for i in range(N_p)])
                loss_dict_val  = compute_loss(NFs, val_batch.to(device).detach(), args, mode='val')  
                mean_score_val = torch.mean(loss_dict_val[args.val_score]).detach()
                running_val_score += mean_score_val
                running_val_LL    += torch.mean(loss_dict_val['G']).detach()
            running_val_score /= args.num_val_batches
            running_val_LL    /= args.num_val_batches

            plot_logger.log('test_steps', step)
            plot_logger.log('test_NLL', float(running_val_LL))

        if running_val_score < best_val_score:
            best_val_score = running_val_score
            path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
            print ("Saved model")
            torch.save(NFs.state_dict(), path)
        
        # scheduler
        if args.lr_schedule == 'adaptive':
            scheduler.step(running_val_score)

        summaries = {
            'best_loss_val': best_val_score.item(),
        }
        if tbx_logging:
            for summary, value in summaries.items():
                writer.add_scalar(tag=summary, scalar_value=value, global_step=step)


# save the last model
if args.num_training_steps != 1: # setting training_steps = 1 means we're doing evaluation.
    path = os.path.join(log_dir, '{}-last-val-{}.t'.format(args.dataset_name, timestamp))
    print ("Saved model")
    torch.save(NFs.state_dict(), path)

# =================================================================================== #
#                                   Evaluation                                        #
# =================================================================================== #

# load best val model
if args.test_model == 'best':
    path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
elif args.test_model == 'last':
    path = os.path.join(log_dir, '{}-last-val-{}.t'.format(args.dataset_name, timestamp))
NFs.load_state_dict(torch.load(path))
NFs.eval()

# data containers
X_sample   = torch.Tensor([]).to(device)
data_names = ['log_density_test', 'KL_density_test', 'KL_sampling_test', \
                'OT_cost_gen_test', 'OT_cost_norm_test', 'F_cost_test', 'F_inter_test', 'F_obs_test']
data_dict  = {}
for n in data_names:
    data_dict[n] = torch.Tensor([]).to(device)

# Lipschitz const
D_NF = torch.Tensor([]).cpu()
lipschitz_NF = 0

# evaluate the trained model's performance on the test set
with torch.no_grad():
    for steps in tqdm(range(args.num_test_batches)):
        batch = torch.stack([next(test_generators[i]).to(device) for i in range(N_p)])
        # log_density, _, _, hist_norm, _, OT_cost_norm, _ = NFs(batch)
        log_density, hist_norm = NFs(batch)
        if args.OT_part != 'module':
            OT_cost_norm, _ = compute_OT_cost_multi(hist_norm, args, mode='norm', partition_mode=args.OT_part,\
                                 LU_last=args.LU_last, scheme=args.disc_scheme)

        # sum up the group-wise L and G costs
        log_density = torch.sum(log_density, dim=0)
        OT_cost_norm = torch.sum(OT_cost_norm, dim=0)

        data_dict['log_density_test'] = torch.cat([
            data_dict['log_density_test'],
            log_density
        ])
        data_dict['OT_cost_norm_test'] = torch.cat([
            data_dict['OT_cost_norm_test'],
            OT_cost_norm
        ])

        # generative direction
        # z_K, ld_gen, OT_cost_gen, hist, hist_ld_gen, z_0 = NFs.sample(args.test_batch_size)
        z_K, z_0, ld_gen, hist = NFs.sample(args.test_batch_size)

        if args.OT_part != 'module':
            OT_cost_gen, hist_part  = compute_OT_cost_multi(hist, args, mode='gen', partition_mode=args.OT_part,\
                                            LU_last=args.LU_last, scheme=args.disc_scheme)
        else: 
            hist_part = partition_hist(hist, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)

        OT_cost_gen = torch.sum(OT_cost_gen, dim=0)

        X_sample  = torch.cat([
            X_sample,
            hist_part.transpose(0,1) # B x N_p x K x d
        ])
        data_dict['OT_cost_gen_test'] = torch.cat([
            data_dict['OT_cost_gen_test'],
            OT_cost_gen
        ])

        log_prob_0   = torch.stack([dists[i].log_prob(z_0[i]) for i in range(hist_norm.shape[0])]) # N_p x B
        log_prob_gen = torch.stack([target_dist[i].log_prob(z_K[i]) for i in range(hist_norm.shape[0])]) # N_p x B
        log_prob_0   = torch.sum(log_prob_0, dim=0) # add up likelihoods from each population
        log_prob_gen = torch.sum(log_prob_gen, dim=0)
        ld_gen       = torch.sum(ld_gen, dim=0)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)

        log_prob_1  = torch.stack([target_dist[i].log_prob(batch[i]) for i in range(hist_norm.shape[0])]) # N_p x B
        log_prob_1  = torch.sum(log_prob_1, dim=0)
        KL_sampling = log_prob_1 - log_density

        data_dict['KL_sampling_test'] = torch.cat([
                data_dict['KL_sampling_test'],
                KL_sampling
            ])

        data_dict['KL_density_test'] = torch.cat([
                data_dict['KL_density_test'],
                KL_density
            ])

        # hist_ld_part = partition_hist(hist_ld_gen, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, hist_type='ld')
        F_inter, F_obs = compute_F_drones(args, hist_part, dataset=args.dataset_name, scheme=args.F_disc_scheme, Q=Q)
        F_cost = args.lbd_F_inter * F_inter + args.lbd_F_obs * F_obs

        data_dict['F_cost_test'] = torch.cat([
            data_dict['F_cost_test'],
            # TODO: this averages batch-wise first. 
            F_cost.unsqueeze(0)
        ])

        data_dict['F_inter_test'] = torch.cat([
            data_dict['F_inter_test'],
            # TODO: this averages batch-wise first. 
            F_inter.unsqueeze(0)
        ])

        data_dict['F_obs_test'] = torch.cat([
            data_dict['F_obs_test'],
            # TODO: this averages batch-wise first. 
            F_obs.unsqueeze(0)
        ])

# =================================================================================== #
#                                    Plotting                                         #
# =================================================================================== #

# filter point trajectories that has a starting point that's too low density
# improves plotting quality, not needed most of the time
if args.plot_traj_thres != 0:
    if args.dataset_name == 'drones_22_obs':
        mu_1 = torch.Tensor([0,0]).to(device)
        mu_2 = torch.Tensor([1,0]).to(device)
    else:
        raise NotImplementedError()
    # X_sample: B x N_p x K x d
    I_1 = torch.norm(X_sample[:, 0, 0, :] - mu_1, dim=-1) < args.plot_traj_thres * np.sqrt(args.var_drones)
    I_2 = torch.norm(X_sample[:, 1, 0, :] - mu_2, dim=-1) < args.plot_traj_thres * np.sqrt(args.var_drones)
    # take the intersection of the acceptable points from both population
    I = I_1 * I_2
    X_sample = X_sample[I]


if args.dataset_name in ['drones_22', 'drones_23', 'drones_22_obs']:
    x_min = -1
    x_max = 2
    y_min = -1
    y_max = 2
elif args.dataset_name == 'drones_82':
    x_min = -args.radius_82-1
    x_max = -x_min
    y_min = -args.radius_82-1
    y_max = -y_min

plot_evolution_multi(args, X_sample.transpose(0,1).cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, marker_size=args.marker_size, mat_save_name='last_epoch',\
                    TB=writer, subset=args.plotting_subset, X_train=X_train.cpu().detach().numpy())

# track evolution of landmarks
X_landmarks = generate_landmarks_multi(args.var_drones, radius=args.radius_82, dataset=args.dataset_name) # N_p x N x d
hist        = NFs.inverse(X_landmarks)
landmarks_traj   = partition_hist(hist, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, multi_pop=True)
plot_evolution_multi(args, landmarks_traj.cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, marker_size=args.marker_size, mat_save_name='landmarks',\
                    TB=writer, subset=args.plotting_subset)

if args.dataset_name == 'drones_23': # plot 3D trajectory in addition
    z_min = -1
    z_max = 2
    plot_evolution_multi_3D(args, X_sample.transpose(0,1).cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, z_min, z_max, marker_size=args.marker_size, mat_save_name='last_epoch',\
                    TB=writer, subset=args.plotting_subset, X_train=X_train.cpu().detach().numpy())
elif args.dataset_name == 'drones_22_obs': # save the obstacle for plotting
    X = Q.sample((20000,))
    obs_save_path = os.path.join(log_dir, 'obstacle.mat')
    obs_den_save_path = os.path.join(log_dir, 'obstacle_density.mat')
    scipy.io.savemat(obs_save_path, dict(obstacle=X.cpu().detach().numpy()))
    # save density values on a grid, used for contour plots
    width = 2
    n_pts = 200
    grid, grid_pad, grid_x, grid_y = create_grid(width, n_pts, features)
    grid = grid.to(device)
    grid_pad = grid_pad.to(device)
    Z = torch.exp(Q.log_prob(grid)).reshape(grid_x.shape)
    scipy.io.savemat(obs_den_save_path, dict(X=grid_x, Y=grid_y, Z=Z.cpu().detach().numpy()))

# =================================================================================== #
#                                       Logging                                       #
# =================================================================================== #

print ('Experiment name: {}'.format(args.exp_name))

N = args.num_test_batches * args.test_batch_size
test_result_str = ''

for n in data_dict:
    data_tensor = data_dict[n]
    path = os.path.join(log_dir, '{}-{}-{}.npy'.format(
        args.dataset_name,
        args.base_transform_type,
        n
    ))
    # save npy mtx
    np.save(path, utils.tensor2numpy(data_tensor))
    data_mean = data_tensor.mean()
    data_std  = data_tensor.std()

    # TB
    if tbx_logging:
        writer.add_scalar(tag=n, scalar_value=data_mean.item())

    # cout
    s   = 'Final {} for {}: {:.4f} +- {:.4f}'.format(
        n,
        args.dataset_name.capitalize(),
        data_mean.item(),
        2 * data_std.item() / np.sqrt(N)
    )
    # markdown format uses two spaces before \n for line breaks
    test_result_str = test_result_str + s + '  \n'
    print(s)
    filename = os.path.join(log_dir, 'test-results.txt')
    with open(filename, 'a') as file:
        file.write(s)
        file.write('\n')


filename = os.path.join(log_dir, 'test-results.txt')
with open(filename, 'a') as file:
    file.write(s)
    file.write('\n')
test_result_str = test_result_str + s + '  \n'

if tbx_logging:
    writer.add_text(tag='test_results', text_string=test_result_str)

