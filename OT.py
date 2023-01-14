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

import data as data_
import nn as nn_
import utils
from mfp_utils import *

from experiments import cutils
from nde import distributions, flows, transforms
from twoNFs import DoubleNormalizingFlow
from vae_mnist_model import VAE
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser()

# data
parser.add_argument('--exp_name', type=str, default='1')
parser.add_argument('--dataset_name', type=str, default='gaussian_mixture',
                    choices=['3gaussians', 'S_swiss', 'moons_spiral',
                            'mnist_mnist', 'mnist_mnist_e2e'],
                    help='Name of dataset to use.')
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
parser.add_argument('--monitor_interval', type=int, default=250,
                    help='Interval in steps at which to report training stats.')

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
parser.add_argument('--lbd_F_P',            type=float, default=1)
parser.add_argument('--lbd_AE',             type=float, default=1)
parser.add_argument('--reg_OT_dir',         type=str, default='gen', choices=['gen', 'norm'])
parser.add_argument('--OT_part',            type=str, default='module', choices=['block', 'block_CL_no_perm', 'module'])
parser.add_argument('--OT_samples',         type=str, default='P1', choices=['P1', 'H_P0'])
parser.add_argument('--interaction',        type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--LU_last',            type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--load_AE',            type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--NF_loss',            type=str, default='KL_sampling', choices=[
                                            'KL_sampling', 'KL_density', 'jeffery'])
parser.add_argument('--val_score',          type=str, default='loss', choices=[
                                            'loss', 'L', 'G', 'F'])
parser.add_argument('--mixture_base',       type=str, default='gaussian', choices=[
                                            'gaussian', 'gaussian_mixture'])
parser.add_argument('--F_ld_weight',        type=str, default='identical', choices=['identical'])                                                                          
parser.add_argument('--disc_scheme',        type=str, default='forward', choices=[
                                            'forward', 'centered', 'forward_2nd',
                                            'FD4_simp', 'FD1_simp', 'FD4_simp_symmetric'])
parser.add_argument('--NF_model',           type=str, default='default', choices=[
                                            'default', 'single_flow'])                                     

# misc.
parser.add_argument('--warm_up_steps',      type=int, default=0, 
                                    help='Number of steps in the beginning to only train for H_*P0 = P1')
parser.add_argument('--plotting_subset',    type=int, default=10000)
parser.add_argument('--load_best_val',      type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--syn_noise',          type=float, default=0.1)
parser.add_argument('--marker_size',        type=float, default=5)
parser.add_argument('--color',              type=str, default='order', choices=[
                                            'order', 'radius'])
parser.add_argument('--tabular_subset',     type=lambda x: (str(x).lower() == 'true'), default=False)

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
torch.set_default_tensor_type('torch.cuda.FloatTensor')


# =================================================================================== #
#                                       Dataset                                       #
# =================================================================================== #
target_dist = None
if args.dataset_name == '3gaussians':
    X_train_1, _, _, _, _, _, train_loader_1, val_loader_1, test_loader_1, train_loader_2, val_loader_2, test_loader_2 \
        = make_3gaussian_data(args.gaussian_multi_a, args.gaussian_multi_dim, args.num_train_data, \
        args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    # train_generator = data_.batch_generator(train_loader)
    train_generator_1 = data_.batch_generator(train_loader_1)
    train_generator_2 = data_.batch_generator(train_loader_2)
    val_generator_2   = data_.batch_generator(val_loader_2)
    test_generator_2  = data_.batch_generator(test_loader_2)
    features        = args.gaussian_multi_dim
elif args.dataset_name in ['S_swiss', 'moons_spiral']:
    X_train_1, _, _, _, _, _, train_loader_1, val_loader_1, test_loader_1, train_loader_2, val_loader_2, test_loader_2 \
         = make_syn_data_twoNFs(args.num_train_data, \
            args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size, args.test_batch_size,\
            noise=args.syn_noise, name=args.dataset_name)
    # train_generator = data_.batch_generator(train_loader)
    train_generator_1 = data_.batch_generator(train_loader_1)
    train_generator_2 = data_.batch_generator(train_loader_2)
    val_generator_2   = data_.batch_generator(val_loader_2)
    test_generator_2  = data_.batch_generator(test_loader_2)
    features        = 2
elif args.dataset_name in ['mnist_mnist', 'mnist_mnist_e2e']:
    X_train_1, X_val_1, X_test_1, X_train_2, X_val_2, X_test_2, train_loader_1, \
        val_loader_1, test_loader_1, train_loader_2, val_loader_2, test_loader_2 \
        = make_mnist_mnist_data(args.train_batch_size, args.val_batch_size, args.test_batch_size, \
        latent= 'e2e' not in args.dataset_name)
    train_generator_1 = data_.batch_generator(train_loader_1)
    train_generator_2 = data_.batch_generator(train_loader_2)
    val_generator_2   = data_.batch_generator(val_loader_2)
    test_generator_2  = data_.batch_generator(test_loader_2)
    features        = 16
else:
    train_dataset = data_.load_dataset(args.dataset_name, split='train')
    if args.tabular_subset:
        train_dataset.data = train_dataset.data[:args.num_train_data]
        train_dataset.n    = args.num_train_data
    X_train       = torch.Tensor(train_dataset.data)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True
    )
    train_generator = data_.batch_generator(train_loader)
    test_batch = next(iter(train_loader)).to(device)

    # validation set
    val_dataset = data_.load_dataset(args.dataset_name, split='val', frac=args.val_frac)
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        drop_last=True
    )

    # test set
    test_dataset = data_.load_dataset(args.dataset_name, split='test')
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False
    )

    features = train_dataset.dim


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
        flows01 = [
            transforms.CompositeTransform([
                create_linear_transform(),
                create_base_transform(i),
                create_base_transform(i+1)
            ]) for i in range(0, 2*args.num_flow_steps, 2)
        ]
        flows12 = [
            transforms.CompositeTransform([
                create_linear_transform(),
                create_base_transform(i),
                create_base_transform(i+1)
            ]) for i in range(0, 2*args.num_flow_steps, 2)
        ]
    else:
        # flows = [
        #     transforms.CompositeTransform([
        #         create_linear_transform(),
        #         create_base_transform(i)
        #     ]) for i in range(args.num_flow_steps)
        # ]
        raise NotImplementedError()
        
    K = args.num_flow_steps
    if args.LU_last:
        flows01 += [create_linear_transform()]
        flows12 += [create_linear_transform()]
        K     += 1

    transform01   = transforms.CompositeTransform(flows01)
    transform12   = transforms.CompositeTransform(flows12)

    return transform01, transform12, K

# base dist
if args.dataset_name == 'gaussian_mixture':
    if args.mixture_base == 'gaussian':
        cov          = 0.3 * torch.eye(features)
        mean         = torch.zeros(features)
        distribution = distributions.MultivarNormal((features,), mean=mean, cov=cov)
    else:
        num_mixtures          = 8
        weight                = D.Categorical(torch.ones(num_mixtures,))
        angle                 = np.pi/4*(torch.arange(8)+1).unsqueeze(1) # N_m x 1
        e_1                   = torch.zeros(1,features)
        e_1[:,0]              = 1.
        e_2                   = torch.zeros(1,features)
        e_2[:,1]              = 1.

        mean                  = 4 * (torch.cos(angle)*e_1.repeat(num_mixtures,1) + torch.sin(angle)*e_2.repeat(num_mixtures,1)) # N_m x d
        cov                   = 0.3 * torch.eye(features).unsqueeze(0).repeat(num_mixtures,1,1)
        dist                  = D.MultivariateNormal(mean, cov)
        mixture               = D.MixtureSameFamily(weight, dist)
        distribution          = distributions.Mixture((features,), mixture)
elif args.dataset_name == 'crowd_motion_gaussian':
    e_2           = torch.zeros(args.gaussian_multi_dim)
    e_2[1]        = 1.
    mean          = 3*e_2
    cov           = 0.3 * torch.eye(args.gaussian_multi_dim)
    distribution  = distributions.MultivarNormal((features,), mean=mean, cov=cov)
else:
    distribution = distributions.StandardNormal((features,))

# create model
transform01, transform12, K = create_transform()
flow01   = flows.Flow(transform01, distribution).to(device)
flow12   = flows.Flow(transform12, distribution).to(device)
if args.dataset_name == 'mnist_mnist_e2e':
    input_dim    = 784
    hidden_dim_1 = 512
    hidden_dim_2 = 256
    latent_dim   = 16
    vae = VAE(x_dim=input_dim, h_dim1= hidden_dim_1, h_dim2=hidden_dim_2, z_dim=latent_dim)
    if args.load_AE:
        load_dir   = './results/vae_mnist'
        assert os.path.exists(load_dir)
        save_path = os.path.join(load_dir, 'vae_mnist.t')
        vae.load_state_dict(torch.load(save_path))
else:
    vae = None
NF_model = DoubleNormalizingFlow(flow01, flow12, args, AE=vae)
n_params = utils.get_num_parameters(NF_model)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(NF_model.parameters(), lr=args.learning_rate, weight_decay=args.lbd_reg)
if args.lr_schedule == 'cyclic':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
elif args.lr_schedule == 'adaptive':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
else:
    scheduler = None

# crowd motion
Q = construct_Q(args.gaussian_multi_dim, device)
F_weight = torch.ones(K,1).to(device)


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


tbar = tqdm(range(args.num_training_steps))
# best_val_score = -1e10
best_val_score = 1e10

# load pretrained model if specified (for continued training)
if args.load_best_val:
    path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
    NF_model.load_state_dict(torch.load(path))
    print ("Loaded model from: {}".format(path))

# =================================================================================== #
#                                      Training                                       #
# =================================================================================== #
def compute_loss(NF_model, x_1, x_2, args, mode='train'):
    # forward
    log_density_1, log_density_2, hist_norm_01, hist_norm_12, z_1, z_2, \
        OT_cost_norm_12, ae_loss_1, ae_loss_2 = NF_model(x_1, x_2)

    # sample if needed
    if args.sample_in_train:
        # if we sample, it'd be because we want to compute transport/interaction costs
        # and the relevant one is the flow between P_1 and P_2.
        # Note: we sample from P_1 to do this.
        if args.OT_samples == 'P1':
            x_2, ld_gen_12, OT_cost_gen_12, hist_gen_12, hist_ld_gen_12 = NF_model.map_12(x_1)
        else:
            x_2, ld_gen_12, OT_cost_gen_12, hist_gen_12, hist_ld_gen_12 = NF_model.map_12(z_1)

    # distribution matching / terminal cost (G)
    KL_density, KL_sampling = torch.Tensor([0]), torch.Tensor([0])
    if args.NF_loss == 'KL_sampling':
        # The other matching costs requires us to evaluate density on P_1/P_2, 
        # if we could do that, we should've just used one flow (mfg.py)
        D_01    = - torch.mean(log_density_1)
        D_01_12 = - torch.mean(log_density_2)
        if mode == 'warm_up':
            # train H_*P0 = P1 to warm up
            G_cost = D_01
        else:
            G_cost = D_01 + D_01_12

        # if target_dist is not None:
        #     log_prob_1  = torch.mean(target_dist.log_prob(data))
        #     KL_sampling = log_prob_1 - log_density
    # elif args.NF_loss == 'KL_density':
    #     assert target_dist is not None
    #     # z_K, ld_gen, _, _, z_0 = flow.sample(args.train_batch_size)
    #     log_prob_0   = torch.mean(distribution.log_prob(z_0))
    #     log_prob_gen = torch.mean(target_dist.log_prob(z_K))
    #     ld_gen       = torch.mean(ld_gen)
    #     KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
    #     G_cost       = KL_density
    # elif args.NF_loss == 'jeffery':
    #     # loss = -log_density
    #     assert target_dist is not None
    #     # z_K, ld_gen, _, _, _ = flow.sample(args.train_batch_size)
    #     log_prob_0   = torch.mean(distribution.log_prob(z_0))
    #     log_prob_gen = torch.mean(target_dist.log_prob(z_K))
    #     ld_gen       = torch.mean(ld_gen)
    #     KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
    #     G_cost       = -log_density + KL_density
    else:
        raise NotImplementedError()

    # OT regularization (L)
    L_cost = torch.Tensor([0])
    if args.lbd_OT != 0:
        if args.reg_OT_dir == 'gen':
            L_cost = OT_cost_gen_12
            hist   = hist_gen_12
        else:
            L_cost = OT_cost_norm_12
            hist   = hist_norm_12
        # the OT cost returned by forward/sample is the module-wise cost
        # other forms of OT cost can be computed from the history.
        if args.OT_part != 'module':
            L_cost, _ = compute_OT_cost(hist, args, mode=args.reg_OT_dir, partition_mode=args.OT_part, LU_last=args.LU_last,
                                            scheme=args.disc_scheme)
        L_cost = torch.mean(L_cost)

    # interaction (F)
    F_P, F_E, F_cost = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])

    if args.interaction:
        hist_part    = partition_hist(hist_gen_12, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last)
        hist_ld_part = partition_hist(hist_ld_gen_12, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, hist_type='ld')
        # TODO: double check if this works as intended.
        log_prob_0   = log_density_1
        F_E, F_P     = compute_F(args, Q, hist_part, log_prob_0, hist_ld_part, dataset=args.dataset_name, \
                        W=F_weight, scheme=args.F_disc_scheme)
        F_E          = torch.mean(F_E)
        F_P          = torch.mean(F_P)
        F_cost       = args.lbd_F_E * F_E + args.lbd_F_P * F_P

    # Overall loss
    if mode == 'warm_up':
        # only train for H_*P0 = P1
        loss = G_cost
    else:
        ae_loss = ae_loss_1.mean() + ae_loss_2.mean()
        loss    = G_cost + args.lbd_OT * L_cost + args.lbd_F * F_cost + args.lbd_AE * ae_loss

    # logging with tensorboard
    if tbx_logging:
        writer.add_scalar(tag='OT_cost' + '_' + mode, scalar_value=L_cost.item(), global_step=step)
        writer.add_scalar(tag='D_01'+ '_' + mode, scalar_value=D_01.item(), global_step=step)
        writer.add_scalar(tag='D_01_12'+ '_' + mode, scalar_value=D_01_12.item(), global_step=step)
        writer.add_scalar(tag='loss'+ '_' + mode, scalar_value=loss.item(), global_step=step)
        writer.add_scalar(tag='KL_density'+ '_' + mode, scalar_value=KL_density.item(), global_step=step)
        writer.add_scalar(tag='KL_sampling'+ '_' + mode, scalar_value=KL_sampling.item(), global_step=step)
        if args.interaction:
            writer.add_scalar(tag='F_P'+ '_' + mode, scalar_value=F_P.item(), global_step=step)
            writer.add_scalar(tag='F_E'+ '_' + mode, scalar_value=F_E.item(), global_step=step)
            writer.add_scalar(tag='F_cost'+ '_' + mode, scalar_value=F_cost.item(), global_step=step)

    # return loss, G_cost, L_cost, F_cost
    return {'loss': loss, 'G': G_cost, 'L': L_cost, 'F': F_cost, 'AE': ae_loss}


# main loop
for step in tbar:
    NF_model.train()
    if args.lr_schedule == 'cyclic':
        scheduler.step(step)
    optimizer.zero_grad()

    # grab data
    # x_1, x_2  = next(train_generator)
    x_1 = next(train_generator_1)
    x_2 = next(train_generator_2)
    x_1 = x_1.float().to(device)
    x_2 = x_2.float().to(device)
    if step < args.warm_up_steps:
        loss_mode = 'warm_up'
    else:
        loss_mode = 'train'
    loss_dict = compute_loss(NF_model, x_1, x_2, args, mode=loss_mode)
    loss      = loss_dict['loss']

    loss.backward()

    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(NF_model.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    # logging
    if (step + 1) % args.monitor_interval == 0:
        NF_model.eval()
        with torch.no_grad():
            # compute validation score
            running_val_score = 0
            # for (x_val_1, x_val_2) in val_loader:
            for x_val_1 in val_loader_1:
                x_val_2 = next(val_generator_2)
                x_val_1 = x_val_1.float().to(device).detach()
                x_val_2 = x_val_2.float().to(device).detach()
                loss_dict_val  = compute_loss(NF_model, x_val_1, x_val_2, args, mode='val')  
                mean_score_val = torch.mean(loss_dict_val[args.val_score]).detach()
                running_val_score += mean_score_val
            running_val_score /= len(val_loader_1)

        if running_val_score < best_val_score:
            best_val_score = running_val_score
            path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
            print ("Saved model")
            torch.save(NF_model.state_dict(), path)
        
        # scheduler
        if args.lr_schedule == 'adaptive':
            scheduler.step(running_val_score)

        # logging
        summaries = {
            'best_loss_val': best_val_score.item(),
        }
        if tbx_logging:
            for summary, value in summaries.items():
                writer.add_scalar(tag=summary, scalar_value=value, global_step=step)


# =================================================================================== #
#                                   Evaluation                                        #
# =================================================================================== #

# load best val model
path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
NF_model.load_state_dict(torch.load(path))
NF_model.eval()

# data containers
X_sample_01 = torch.Tensor([])
X_sample_12 = torch.Tensor([])
data_names  = ['log_density_1_test', 'log_density_2_test', 'KL_density_test', 'KL_sampling_test', \
                'OT_cost_gen_test', 'OT_cost_norm_test', 'AE_loss']
if args.interaction:
    data_names += ['F_cost_test', 'F_E_test', 'F_P_test']
data_dict  = {}
for n in data_names:
    data_dict[n] = torch.Tensor([])

# evaluate the trained model's performance on the test set
with torch.no_grad():
    # for (x_1, x_2) in tqdm(test_loader):
    for x_1 in tqdm(test_loader_1):
        x_2 = next(test_generator_2)
        x_1 = x_1.float().to(device)
        x_2 = x_2.float().to(device)
        # normalizing direction
        # log_density, hist_norm, _, OT_cost_norm = NF_model.log_prob(batch)
        log_density_1, log_density_2, hist_norm_01, hist_norm_12, z_1, z_2, \
                                        OT_cost_norm, ae_loss_1, ae_loss_2 = NF_model(x_1, x_2)
        if args.OT_part != 'module':
            OT_cost_norm, _ = compute_OT_cost(hist_norm_12, args, mode='norm', partition_mode=args.OT_part,\
                                 LU_last=args.LU_last, scheme=args.disc_scheme)

        data_dict['log_density_1_test'] = torch.cat([
            data_dict['log_density_1_test'],
            log_density_1
        ])
        data_dict['log_density_2_test'] = torch.cat([
            data_dict['log_density_2_test'],
            log_density_2
        ])
        data_dict['OT_cost_norm_test'] = torch.cat([
            data_dict['OT_cost_norm_test'],
            OT_cost_norm
        ])

        data_dict['AE_loss'] = torch.cat([
            data_dict['AE_loss'],
            ae_loss_1, 
            ae_loss_2
        ])

        # generative direction
        # z_K, ld_gen, OT_cost_gen, hist, hist_ld_gen, z_0 = flow.sample(args.test_batch_size)
        # log_prob_0   = distribution.log_prob(z_0)

        # for flow12. Note that sampling amounts to passing x_1 through the generative direction.
        # x_2, ld_gen_12, OT_cost_gen_12, hist_gen_12, hist_ld_gen_12 = NF_model.flow12._transform.inverse(x_1)
        x_2, ld_gen_12, OT_cost_gen_12, hist_gen_12, hist_ld_gen_12 = NF_model.map_12(x_1)
        # for flow01
        z_K, _, _, hist_gen_01, _, _ = NF_model.flow01.sample(args.test_batch_size)
        hist_part01 = partition_hist(hist_gen_01, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)

        if args.OT_part != 'module':
            OT_cost_gen, hist_part12  = compute_OT_cost(hist_gen_12, args, mode='gen', partition_mode=args.OT_part,\
                                            LU_last=args.LU_last, scheme=args.disc_scheme)
        else: 
            hist_part12 = partition_hist(hist_gen_12, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)

        X_sample_01  = torch.cat([
            X_sample_01,
            hist_part01
        ])
        X_sample_12  = torch.cat([
            X_sample_12,
            hist_part12
        ])
        data_dict['OT_cost_gen_test'] = torch.cat([
            data_dict['OT_cost_gen_test'],
            OT_cost_gen
        ])

        # # compute the actual KL values for fair comparison
        # if args.dataset_name == 'gaussian_mixture' or args.dataset_name == 'crowd_motion_gaussian':
        #     log_prob_gen = target_dist.log_prob(z_K)
        #     KL_density   = - (log_prob_gen + ld_gen - log_prob_0)

        #     log_prob_1   = target_dist.log_prob(batch)
        #     KL_sampling  = log_prob_1 - log_density

        #     data_dict['KL_density_test'] = torch.cat([
        #         data_dict['KL_density_test'],
        #         KL_density
        #     ])
        #     data_dict['KL_sampling_test'] = torch.cat([
        #         data_dict['KL_sampling_test'],
        #         KL_sampling
        #     ])

        if args.interaction:
            hist_ld_part = partition_hist(hist_ld_gen_12, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, hist_type='ld')
            # TODO: I swapped log_prob_0 for log_density_1, double check to see if this is the correct one.
            F_E, F_P = compute_F(args, Q, hist_part12, log_density_1, hist_ld_part, dataset=args.dataset_name, \
                            W=F_weight, scheme=args.F_disc_scheme)
            F_cost   = args.lbd_F_E * F_E + args.lbd_F_P * F_P
            data_dict['F_cost_test'] = torch.cat([
                data_dict['F_cost_test'],
                F_cost
            ])
            data_dict['F_E_test'] = torch.cat([
                data_dict['F_E_test'],
                F_E
            ])
            data_dict['F_P_test'] = torch.cat([
                data_dict['F_P_test'],
                F_P
            ])

# =================================================================================== #
#                                    Plotting                                         #
# =================================================================================== #
if args.dataset_name == 'mnist_mnist':
    plot_size    = 100
    num_rows     = 10
    input_dim    = 784
    hidden_dim_1 = 512
    hidden_dim_2 = 256
    latent_dim   = 16

    # load vae for decoding 
    vae = VAE(x_dim=input_dim, h_dim1= hidden_dim_1, h_dim2=hidden_dim_2, z_dim=latent_dim)
    load_dir   = './results/vae_mnist'
    assert os.path.exists(load_dir)

    save_path = os.path.join(load_dir, 'vae_mnist.t')
    vae.load_state_dict(torch.load(save_path))

    # save some decoded training samples
    decoded_x_1_train = vae.decoder(next(train_generator_1)).view(-1, 1, 28, 28)
    decoded_x_2_train = vae.decoder(next(train_generator_2)).view(-1, 1, 28, 28)

    images_1  = make_grid(decoded_x_1_train.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_1    = 'train_samples_1.png'
    save_image(images_1, os.path.join(log_dir, name_1))

    images_2  = make_grid(decoded_x_2_train.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_2    = 'train_samples_2.png'
    save_image(images_2, os.path.join(log_dir, name_2))

    # 1 is the set of digits {0,1,2,3,4}, 2 is the set of digits {5,6,7,8,9}
    decoded_x_1  = vae.decoder(X_sample_12[:plot_size, 0,:]).view(-1, 1, 28, 28)
    decoded_x_H0 = vae.decoder(X_sample_01[:plot_size, -1,:]).view(-1, 1, 28, 28)
    decoded_x_F1 = vae.decoder(X_sample_12[:plot_size, -1,:]).view(-1, 1, 28, 28)

    # plot the matching digits from set 1 -> set 2
    # images_1  = make_grid(torch.clamp(decoded_x_1.cpu(), min=-args.color_map_range, max=args.color_map_range), \
    #             nrow=num_rows, pad_value=1, normalize=True)
    images_1  = make_grid(decoded_x_1.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_1    = 'samples_P1.png'
    save_image(images_1, os.path.join(log_dir, name_1))

    images_H0  = make_grid(decoded_x_H0.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_H0    = 'H_P0.png'
    save_image(images_H0, os.path.join(log_dir, name_H0))

    images_F1  = make_grid(decoded_x_F1.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_F1    = 'F_P1.png'
    save_image(images_F1, os.path.join(log_dir, name_F1))

    # save data
    train_save_path_1  = os.path.join(log_dir, '_train_P1.t')
    train_save_path_2  = os.path.join(log_dir, '_train_P2.t')
    sample_save_path_1 = os.path.join(log_dir, '_sample_trajectory_P0P1.t')
    sample_save_path_2 = os.path.join(log_dir, '_sample_trajectory_P1P2.t')
    # scipy.io.savemat(train_save_path_1,  dict(data=X_train_1))
    # scipy.io.savemat(train_save_path_2,  dict(data=X_train_2))
    # scipy.io.savemat(sample_save_path_1, dict(data=X_sample_01))
    # scipy.io.savemat(sample_save_path_2, dict(data=X_sample_12))
    torch.save(X_train_1,  train_save_path_1)
    torch.save(X_train_2,  train_save_path_2)
    torch.save(X_sample_01,  sample_save_path_1)
    torch.save(X_sample_12,  sample_save_path_2)


elif args.dataset_name == 'mnist_mnist_e2e':
    plot_size    = 100
    num_rows     = 10
    # save some training samples
    x_1_train = next(train_generator_1).view(-1, 1, 28, 28)
    x_2_train = next(train_generator_2).view(-1, 1, 28, 28)

    images_1  = make_grid(x_1_train.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_1    = 'train_samples_1.png'
    save_image(images_1, os.path.join(log_dir, name_1))

    images_2  = make_grid(x_2_train.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_2    = 'train_samples_2.png'
    save_image(images_2, os.path.join(log_dir, name_2))

    # 1 is the set of digits {0,1,2,3,4}, 2 is the set of digits {5,6,7,8,9}
    decoded_x_1  = NF_model.AE.decoder(X_sample_12[:plot_size, 0,:]).view(-1, 1, 28, 28)
    decoded_x_H0 = NF_model.AE.decoder(X_sample_01[:plot_size, -1,:]).view(-1, 1, 28, 28)
    decoded_x_F1 = NF_model.AE.decoder(X_sample_12[:plot_size, -1,:]).view(-1, 1, 28, 28)

    # plot the matching digits from set 1 -> set 2
    # images_1  = make_grid(torch.clamp(decoded_x_1.cpu(), min=-args.color_map_range, max=args.color_map_range), \
    #             nrow=num_rows, pad_value=1, normalize=True)
    images_1  = make_grid(decoded_x_1.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_1    = 'samples_P1.png'
    save_image(images_1, os.path.join(log_dir, name_1))

    images_H0 = make_grid(decoded_x_H0.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_H0   = 'H_P0.png'
    save_image(images_H0, os.path.join(log_dir, name_H0))

    images_F1  = make_grid(decoded_x_F1.cpu(), nrow=num_rows, pad_value=1, normalize=True)
    name_F1    = 'F_P1.png'
    save_image(images_F1, os.path.join(log_dir, name_F1))
else:
    # plot ranges for syn data
    plot_range = {'3gaussians':[-5,15,-5,15], 'moons_spiral':[-5,5,-5,5], 'S_swiss':[-3,3,-3,3]}

    if args.dataset_name in plot_range:
        x_min, x_max, y_min, y_max = plot_range[args.dataset_name]
    else:
        x_min, x_max, y_min, y_max = -10, 10, -10, 10
    X_1_train = torch.cat([x for x in train_loader_1])
    X_2_train = torch.cat([x for x in train_loader_2])
    plot_evolution_twoNFs(args, X_sample_01.cpu().detach().numpy(), X_sample_12.cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, marker_size=args.marker_size, mat_save_name='last_epoch',\
                    TB=writer, subset=args.plotting_subset, X_1_train=X_1_train.cpu().detach().numpy(), 
                    X_2_train=X_2_train.cpu().detach().numpy())


# =================================================================================== #
#                                       Logging                                       #
# =================================================================================== #

print ('Experiment name: {}'.format(args.exp_name))


if args.dataset_name in ['mnist_mnist']:
    # TODO: not exactly 30k, the two log density should have different N's
    N = 30000
else:
    N = args.num_test_data
    

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

if tbx_logging:
    writer.add_text(tag='test_results', text_string=test_result_str)

