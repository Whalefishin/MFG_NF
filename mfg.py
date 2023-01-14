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

import data as data_
import nn as nn_
import utils
from mfp_utils import *

from experiments import cutils
from nde import distributions, flows, transforms

import klampt
from klampt.plan import cspace, robotplanning
from klampt.plan.robotcspace import RobotCSpace
from klampt.model import collide
from klampt.model.trajectory import RobotTrajectory
from klampt.io import resource

from mfg_models import construct_Q

parser = argparse.ArgumentParser()

# data
parser.add_argument('--exp_name', type=str, default='1')
parser.add_argument('--dataset_name', type=str, default='gaussian_mixture',
                    choices=['gaussian_mixture', 'crowd_motion_gaussian', 'crowd_motion_gaussian_close'
                            'crowd_motion_gaussian_nonsmooth_obs', 'moons', 
                            'gaussian', '2spirals', 'checkerboard',
                            'power', 'gas', 'hepmass', 'miniboone', 'bsds300',
                            'robot_1'],
                    help='Name of dataset to use.')
parser.add_argument('--val_frac', type=float, default=1.,
                    help='Fraction of validation set to use.')

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
parser.add_argument('--reg_OT_dir',         type=str, default='gen', choices=['gen', 'norm'])
parser.add_argument('--OT_comp',            type=str, default='trajectory', choices=['trajectory', 'monge'])
parser.add_argument('--OT_part',            type=str, default='module', choices=['block', 'block_CL_no_perm', 'module'])
parser.add_argument('--interaction',        type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--LU_last',            type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--NF_loss',            type=str, default='KL_sampling', choices=[
                                            'KL_sampling', 'KL_density', 'jeffery'])
parser.add_argument('--val_score',          type=str, default='loss', choices=[
                                            'loss', 'L', 'G', 'F'])
parser.add_argument('--mixture_base',       type=str, default='gaussian', choices=[
                                            'gaussian', 'gaussian_mixture'])
parser.add_argument('--mixture_weight',     type=str, default='identical', choices=[
                                            'identical', 'undersample_one'])  
parser.add_argument('--F_ld_weight',        type=str, default='identical', choices=['identical'])                                                                          
parser.add_argument('--disc_scheme',        type=str, default='forward', choices=[
                                            'forward', 'centered', 'forward_2nd',
                                            'FD4_simp', 'FD1_simp', 'FD4_simp_symmetric'])
parser.add_argument('--NF_model',           type=str, default='default', choices=[
                                            'default', 'single_flow'])                                     
parser.add_argument('--obs_nonsmooth_val',  type=float, default=100.)
parser.add_argument('--interp_hist',        type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--n_interp',           type=int, default=5, 
                    help='Number of interpolated points inserted between flow points to better approximate MFG costs')
## robotics
parser.add_argument('--robot_init_pos',     type=str, default='default', choices=['default', 
                                            'under_table', 'under_table_2', 'under_table_3', 'under_table_4',
                                            'under_table_hard']) 
parser.add_argument('--robot_term_pos',     type=str, default='cup', choices=['cup']) 
parser.add_argument('--robot_var',          type=float, default=1e-5)
parser.add_argument('--robot_obs_val',      type=float, default=1e2)
parser.add_argument('--robot_1_obs',        type=str, default='thick_sigmoid_B=256')
parser.add_argument('--robot_1_obs_l',      type=int, default=3)
parser.add_argument('--robot_1_obs_act',    type=str, default='relu', choices=['relu', 'tanh'])
parser.add_argument('--robot_1_base_dist',  type=str, default='default', choices=['default', 'two_init'])
parser.add_argument('--obs_robot_sig',      type=lambda x: (str(x).lower() == 'true'), default=True) 

# misc.
parser.add_argument('--plotting_subset',    type=int, default=10000)
parser.add_argument('--load_best_val',      type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--compute_lip_bound',  type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--save_train_traj',    type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--syn_noise',          type=float, default=0.1)
parser.add_argument('--marker_size',        type=float, default=5)
parser.add_argument('--color',              type=str, default='order', choices=[
                                            'order', 'radius'])
parser.add_argument('--tabular_subset',     type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--tensor_type',        type=str, default='float', choices=['float', 'double']) 

args = parser.parse_args()
args = sanitize_args(args)

# =================================================================================== #
#                                       Meta                                          #
# =================================================================================== #

os.environ['DATAROOT'] = 'experiments/dataset/data/'
os.environ['SLURM_JOB_ID'] = '1'

torch.manual_seed(args.seed)
np.random.seed(args.seed)

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =================================================================================== #
#                                       Dataset                                       #
# =================================================================================== #
target_dist = None
space = None
if args.dataset_name == 'gaussian_mixture':
    num_mixtures = 8
    if args.mixture_weight == 'identical':
        weight = D.Categorical(torch.ones(num_mixtures,).to(device))
    elif args.mixture_weight == 'undersample_one':
        weight = D.Categorical(torch.Tensor([1.]*7 + [0.01]))
    else:
        raise NotImplementedError()
    X_train, _, _, train_loader, val_loader, test_loader, target_dist = make_gaussian_mixture_data(args.mixture_base, args.gaussian_multi_dim, args.num_train_data, \
        args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size, args.test_batch_size, weight=weight)
    train_generator = data_.batch_generator(train_loader)
    test_batch      = next(iter(train_loader)).to(device)
    features        = args.gaussian_multi_dim
elif args.dataset_name in ['crowd_motion_gaussian', 'crowd_motion_gaussian_nonsmooth_obs', 'crowd_motion_gaussian_close']:
    X_train, _, _, train_loader, val_loader, test_loader, target_dist = make_crowd_motion_gaussian_data(args, args.gaussian_multi_dim, args.num_train_data, \
        args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    train_generator = data_.batch_generator(train_loader)
    test_batch      = next(iter(train_loader)).to(device)
    features        = args.gaussian_multi_dim
elif args.dataset_name in ['robot_1']:
    init_path_dict = {'default': 'default_start.config', 'under_table': 'under_table_start.config',\
                    'under_table_2': 'under_table_start_2.config', 'under_table_3': 'under_table_start_3.config', \
                    'under_table_4': 'under_table_start_4.config', 'under_table_hard': 'under_table_start_hard.config'}
    term_path_dict = {'cup': 'cup_end_2.config'}

    # setting up the scenario and C space in Klampt. Collision checking is done through space.isFeasible()
    world = klampt.WorldModel()
    # world.readFile("./Klampt-examples/data/tx90cuptable.xml")
    world.readFile("./Klampt-examples/data/tx90cuptable_2.xml")
    robot = world.robot(0)
    space = RobotCSpace(robot, collide.WorldCollider(world))

    qstart = resource.get(init_path_dict[args.robot_init_pos])
    qgoal  = resource.get(term_path_dict[args.robot_term_pos])
    qdefault = resource.get('default_start.config')
    qunder   = resource.get('under_table_start_3.config')

    settings = {'type':'rrt',
        'perturbationRadius':0.25,
        'bidirectional':True,
        'shortcut':True,
        'restart':True,
        'restartTermCond':"{foundSolution:1,maxIters:1000}"
    }
    planner = cspace.MotionPlan(space, **settings)
    planner.setEndpoints(qstart, qgoal)

    d = 12
    features = 10

    x_0_raw = torch.Tensor(qstart).to(device)
    x_K_raw = torch.Tensor(qgoal).to(device)
    x_default_start_raw = torch.Tensor(qdefault).to(device)
    x_under_start_raw   = torch.Tensor(qunder).to(device)
    # process to remove dummy dimensions
    x_0 = process_robot_data(x_0_raw, dataset=args.dataset_name)
    x_K = process_robot_data(x_K_raw, dataset=args.dataset_name)
    x_default_start = process_robot_data(x_default_start_raw, dataset=args.dataset_name)
    x_under_start = process_robot_data(x_under_start_raw, dataset=args.dataset_name)

    # TODO
    X_train, _, _, train_loader, val_loader, test_loader, target_dist = make_robot_data(x_K, args.robot_var, features, \
        args.num_train_data, args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size, \
        args.test_batch_size)

    train_generator = data_.batch_generator(train_loader)
    test_batch      = next(iter(train_loader)).to(device)
    
elif args.dataset_name == 'gaussian':
    X_train, _, _, train_loader, val_loader, test_loader, target_dist = make_gaussian_data(args.gaussian_multi_a, args.gaussian_multi_dim, args.num_train_data, \
        args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size, args.test_batch_size)
    train_generator = data_.batch_generator(train_loader)
    test_batch      = next(iter(train_loader)).to(device)
    features        = args.gaussian_multi_dim
elif args.dataset_name in ['moons', 'checkerboard', '2spirals']:
    X_train, _, _, train_loader, val_loader, test_loader = make_syn_data(args.num_train_data, \
        args.num_val_data, args.num_test_data, args.train_batch_size, args.val_batch_size, \
            args.test_batch_size, noise=args.syn_noise, name=args.dataset_name)
    train_generator = data_.batch_generator(train_loader)
    test_batch      = next(iter(train_loader)).to(device)
    features        = 2
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
if args.dataset_name == 'gaussian_mixture':
    if args.mixture_base == 'gaussian':
        cov          = 0.3 * torch.eye(features).to(device)
        mean         = torch.zeros(features).to(device)
        distribution = distributions.MultivarNormal((features,), mean=mean, cov=cov)
    else:
        num_mixtures          = 8
        weight                = D.Categorical(torch.ones(num_mixtures,).to(device))
        angle                 = np.pi/4*(torch.arange(8)+1).unsqueeze(1) # N_m x 1
        e_1                   = torch.zeros(1,features)
        e_1[:,0]              = 1.
        e_2                   = torch.zeros(1,features)
        e_2[:,1]              = 1.

        mean                  = 4 * (torch.cos(angle)*e_1.repeat(num_mixtures,1) + torch.sin(angle)*e_2.repeat(num_mixtures,1)) # N_m x d
        cov                   = 0.3 * torch.eye(features).unsqueeze(0).repeat(num_mixtures,1,1)
        dist                  = D.MultivariateNormal(mean.to(device), cov)
        mixture               = D.MixtureSameFamily(weight.to(device), dist)
        distribution          = distributions.Mixture((features,), mixture)
elif args.dataset_name in ['crowd_motion_gaussian']:
    e_2           = torch.zeros(args.gaussian_multi_dim).to(device)
    e_2[1]        = 1.
    mean          = 3*e_2
    cov           = 0.3 * torch.eye(args.gaussian_multi_dim).to(device)
    distribution  = distributions.MultivarNormal((features,), mean=mean, cov=cov)
elif args.dataset_name == 'gaussian':
    cov = 0.01 * torch.eye(args.gaussian_multi_dim).to(device)
    distribution  = distributions.MultivarNormal((features,), cov=cov)
elif args.dataset_name == 'robot_1':
    if args.robot_1_base_dist == 'two_init':
        num_mixtures = 2
        mean   = torch.stack([x_default_start, x_under_start]).to(device) # 2 x d
        cov    = args.robot_var * torch.eye(features).unsqueeze(0).repeat(num_mixtures,1,1).to(device) # 2 x d x d
        dist   = D.MultivariateNormal(mean, cov)
        weight = D.Categorical(torch.ones(num_mixtures,).to(device))
        mixture      = D.MixtureSameFamily(weight, dist)
        distribution = distributions.Mixture((features,), mixture)
    else:
        cov  = args.robot_var * torch.eye(features).to(device)
        distribution  = distributions.MultivarNormal((features,), mean=x_0, cov=cov)
else:
    mean = torch.zeros(features).to(device)
    cov  = torch.eye(features).to(device)
    distribution  = distributions.MultivarNormal((features,), mean=mean, cov=cov)

# create flows
if args.NF_model == 'single_flow':
    transform, _ = create_transform()
    K    = args.K
    flow = flows.SingleFlow(transform, distribution, K).to(device)
else:
    transform, K = create_transform()
    flow = flows.Flow(transform, distribution).to(device)
n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate, weight_decay=args.lbd_reg)
if args.lr_schedule == 'cyclic':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
elif args.lr_schedule == 'adaptive':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
else:
    scheduler = None

# crowd motion
Q = construct_Q(args, device)
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

plot_data_dir = os.path.join(log_dir, 'plot_data/')
if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)
plot_logger = Plot_logger(root_path=plot_data_dir)

tbar = tqdm(range(args.num_training_steps))
# best_val_score = -1e10
best_val_score = 1e10


# =================================================================================== #
#                                      Training                                       #
# =================================================================================== #
def compute_loss(flow, data, args, mode='train'):
    # forward
    log_density, _, _, hist_norm, hist_ld_norm, OT_cost_norm, _ = flow.log_prob(data)
    log_density = torch.mean(log_density)

    # sample if needed
    if args.sample_in_train:
        z_K, ld_gen, OT_cost_gen, hist_gen, hist_ld_gen, z_0 = flow.sample(args.train_batch_size)

    # distribution matching / terminal cost (G)
    KL_density  = torch.Tensor([0]).to(device)
    KL_sampling = torch.Tensor([0]).to(device)
    if args.NF_loss == 'KL_sampling':
        G_cost = -log_density
        if target_dist is not None:
            log_prob_1  = torch.mean(target_dist.log_prob(data))
            KL_sampling = log_prob_1 - log_density
    elif args.NF_loss == 'KL_density':
        assert target_dist is not None
        log_prob_0   = torch.mean(distribution.log_prob(z_0))
        log_prob_gen = torch.mean(target_dist.log_prob(z_K))
        ld_gen       = torch.mean(ld_gen)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = KL_density
    elif args.NF_loss == 'jeffery':
        # loss = -log_density
        assert target_dist is not None
        log_prob_1  = torch.mean(target_dist.log_prob(data))
        log_prob_0   = torch.mean(distribution.log_prob(z_0))
        log_prob_gen = torch.mean(target_dist.log_prob(z_K))
        ld_gen       = torch.mean(ld_gen)
        KL_density   = - (log_prob_gen + ld_gen - log_prob_0)
        G_cost       = log_prob_1 - log_density + KL_density
    else:
        raise NotImplementedError()

    # OT regularization (L)
    L_cost = torch.Tensor([0]).to(device)
    if args.lbd_OT != 0:
        if args.reg_OT_dir == 'gen':
            # sample a batch from the base to compute the OT cost
            L_cost = OT_cost_gen
            hist   = hist_gen
        else:
            L_cost = OT_cost_norm
            hist   = hist_norm
        # the OT cost returned by forward/sample is the module-wise cost
        if args.OT_part != 'module':
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
        log_prob_0   = distribution.log_prob(z_0)
        F_E, F_P     = compute_F(args, Q, hist_part, log_prob_0, hist_ld_part, dataset=args.dataset_name, \
                        W=F_weight, scheme=args.F_disc_scheme, Q_is_dist=args.Q_is_dist)
        F_E          = torch.mean(F_E)
        F_P          = torch.mean(F_P)
        F_cost       = args.lbd_F_E * F_E + args.lbd_F_P * F_P

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
            writer.add_scalar(tag='F_P'+ '_' + mode, scalar_value=F_P.item(), global_step=step)
            writer.add_scalar(tag='F_E'+ '_' + mode, scalar_value=F_E.item(), global_step=step)
            writer.add_scalar(tag='F_cost'+ '_' + mode, scalar_value=F_cost.item(), global_step=step)

    # logging to .txt
    if mode == 'train':
        # only do this for training. Logging for validation data is done in the main loop.
        plot_logger.log(mode + '_steps', step)
        plot_logger.log(mode + '_NLL', float(-log_density))

    # return loss, G_cost, L_cost, F_cost
    return {'loss': loss, 'G': G_cost, 'L': L_cost, 'F': F_cost}

# load pretrained model if specified (for continued training)
if args.load_best_val:
    path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
    flow.load_state_dict(torch.load(path))
    print ("Loaded model from: {}".format(path))

# main loop
for step in tbar:
    flow.train()
    if args.lr_schedule == 'cyclic':
        scheduler.step(step)
    optimizer.zero_grad()

    # grab data
    batch = next(train_generator).to(device)
    # loss  = compute_loss(flow, batch, args, mode='train')
    loss_dict = compute_loss(flow, batch, args, mode='train')
    loss      = loss_dict['loss']

    loss.backward()

    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    # logging
    if (step + 1) % args.monitor_interval == 0:
        flow.eval()
        with torch.no_grad():
            # compute validation score
            running_val_score = 0
            running_val_LL    = 0
            for val_batch in val_loader:
                loss_dict_val  = compute_loss(flow, val_batch.to(device).detach(), args, mode='val')  
                mean_score_val = torch.mean(loss_dict_val[args.val_score]).detach()
                running_val_score += mean_score_val
                running_val_LL    += torch.mean(loss_dict_val['G']).detach()
            running_val_score /= len(val_loader)
            running_val_LL    /= len(val_loader)

            plot_logger.log('test_steps', step)
            plot_logger.log('test_NLL', float(running_val_LL))

        if running_val_score < best_val_score:
            best_val_score = running_val_score
            path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
            print ("Saved model")
            torch.save(flow.state_dict(), path)
        
        # scheduler
        if args.lr_schedule == 'adaptive':
            scheduler.step(running_val_score)

        # # compute reconstruction
        # with torch.no_grad():
        #     test_batch_noise                  = flow.transform_to_noise(test_batch)
        #     test_batch_reconstructed, _, _, _ = flow._transform.inverse(test_batch_noise)
        # errors = test_batch - test_batch_reconstructed
        # max_abs_relative_error = torch.abs(errors / test_batch).max()
        # average_abs_relative_error = torch.abs(errors / test_batch).mean()
        # if tbx_logging:
        #     writer.add_scalar('max-abs-relative-error',
        #                         max_abs_relative_error, global_step=step)
        #     writer.add_scalar('average-abs-relative-error',
        #                         average_abs_relative_error, global_step=step)

        summaries = {
            'best_loss_val': best_val_score.item(),
            # 'log_density_val': running_val_log_density.item(),
            # 'best_log_density_val': best_val_score.item(),
            # 'loss_val': running_val_loss.item()
            # 'max-abs-relative-error': max_abs_relative_error.item(),
            # 'average-abs-relative-error': average_abs_relative_error.item()
        }
        if tbx_logging:
            for summary, value in summaries.items():
                writer.add_scalar(tag=summary, scalar_value=value, global_step=step)


# =================================================================================== #
#                                   Evaluation                                        #
# =================================================================================== #

# load best val model
path = os.path.join(log_dir, '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
flow.load_state_dict(torch.load(path))
flow.eval()

# data containers
X_sample   = torch.Tensor([]).to(device)
data_names = ['log_density_test', 'KL_density_test', 'KL_sampling_test', \
                'OT_cost_gen_test', 'OT_cost_norm_test']
if args.interaction:
    data_names += ['F_cost_test', 'F_E_test', 'F_P_test']
data_dict  = {}
for n in data_names:
    data_dict[n] = torch.Tensor([]).to(device)


# evaluate the trained model's performance on the test set
with torch.no_grad():
    for batch in tqdm(test_loader):
        if args.tensor_type == 'double':
            batch = batch.to(device).double()
        else:
            batch = batch.to(device)
        # normalizing direction
        # loss, log_density, OT_cost_norm, _, _, _ = compute_loss(flow, batch.to(device), args)
        log_density, _, _, hist_norm, _, OT_cost_norm, _ = flow.log_prob(batch)
        if args.OT_part != 'module':
            OT_cost_norm, _ = compute_OT_cost(hist_norm, args, mode='norm', partition_mode=args.OT_part,\
                                 LU_last=args.LU_last, scheme=args.disc_scheme)

        data_dict['log_density_test'] = torch.cat([
            data_dict['log_density_test'],
            log_density
        ])
        data_dict['OT_cost_norm_test'] = torch.cat([
            data_dict['OT_cost_norm_test'],
            OT_cost_norm
        ])

        # generative direction
        z_K, ld_gen, OT_cost_gen, hist, hist_ld_gen, z_0 = flow.sample(args.test_batch_size)

        if args.OT_part != 'module':
            OT_cost_gen, hist_part = compute_OT_cost(hist, args, mode='gen', partition_mode=args.OT_part,\
                                            LU_last=args.LU_last, scheme=args.disc_scheme)
        # else: 
        #     hist_part = partition_hist(hist, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)
        
        X_sample  = torch.cat([
            X_sample,
            hist_part
        ])
        data_dict['OT_cost_gen_test'] = torch.cat([
            data_dict['OT_cost_gen_test'],
            OT_cost_gen
        ])

        # compute the actual KL values for fair comparison
        if args.dataset_name in ['gaussian_mixture', 'crowd_motion_gaussian', 'crowd_motion_gaussian_nonsmooth_obs',\
                                 'robot_1']:
            log_prob_0   = distribution.log_prob(z_0)
            log_prob_gen = target_dist.log_prob(z_K)
            KL_density   = - (log_prob_gen + ld_gen - log_prob_0)

            log_prob_1   = target_dist.log_prob(batch)
            KL_sampling  = log_prob_1 - log_density

            data_dict['KL_density_test'] = torch.cat([
                data_dict['KL_density_test'],
                KL_density
            ])
            data_dict['KL_sampling_test'] = torch.cat([
                data_dict['KL_sampling_test'],
                KL_sampling
            ])

        if args.interaction:
            hist_ld_part = partition_hist(hist_ld_gen, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last, hist_type='ld')
            hist_part    = partition_hist(hist, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)
            F_E, F_P = compute_F(args, Q, hist_part, log_prob_0, hist_ld_part, dataset=args.dataset_name, \
                            W=F_weight, scheme=args.F_disc_scheme, Q_is_dist=args.Q_is_dist)
            F_cost = args.lbd_F_E * F_E + args.lbd_F_P * F_P

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


# Lipschitz const
D_NF = torch.Tensor([]).cpu()
lipschitz_NF = 0
if args.compute_lip_bound:
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        D     = batch_jacobian(flow.forward_ret_z, batch) # B x d x d
        D_NF  = torch.cat([D_NF, D.cpu()])

    D_NF_norms   = np.linalg.norm(D_NF.numpy(), axis=(1,2), ord=2) # N x K
    lipschitz_NF = torch.from_numpy(np.max(D_NF_norms).reshape(1)) # 1

# =================================================================================== #
#                                    Plotting                                         #
# =================================================================================== #

if args.dataset_name in ['gaussian_mixture', 'crowd_motion_gaussian', 'crowd_motion_gaussian_nonsmooth_obs']:
    x_min = -7
    x_max = 7
    y_min = -7
    y_max = 7
    plot_evolution(args, X_sample.cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, subset=args.plotting_subset, marker_size=args.marker_size, \
                    mat_save_name='last_epoch', TB=writer, X_train=X_train.cpu().detach().numpy())

    X_landmarks      = generate_landmarks(features, dataset=args.dataset_name)
    if args.NF_model == 'single_flow':
        _, _, _, hist, _ = flow.inverse(X_landmarks)
    else:
        _, _, _, hist, _ = flow._transform.inverse(X_landmarks)
    landmarks_traj   = partition_hist(hist, args, mode='gen', partition_mode=args.OT_part, LU_last=args.LU_last)

    plot_evolution(args, landmarks_traj.cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, subset=args.plotting_subset, marker_size=args.marker_size,\
                    mat_save_name='landmarks', TB=writer)
    # plot the evolution of some landmarks for crowd motion
    if args.dataset_name in ['crowd_motion_gaussian', 'crowd_motion_gaussian_nonsmooth_obs']:
        # save samples from the obstacle
        X = Q.sample((20000,))
        obs_save_path  = os.path.join(log_dir, 'obstacle.mat')
        scipy.io.savemat(obs_save_path, dict(obstacle=X.cpu().detach().numpy()))
elif args.dataset_name == 'gaussian':
    x_min = -1
    x_max = 1.5
    y_min = -1
    y_max = 1.5
    plot_evolution(args, X_sample.cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, marker_size=args.marker_size, mat_save_name='last_epoch',\
                    TB=writer, subset=args.plotting_subset, X_train=X_train.cpu().detach().numpy())
elif args.dataset_name == 'moons':
    # x_min = -1.5
    # x_max = 2.5
    # y_min = -1.0
    # y_max = 1.5
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    plot_evolution(args, X_sample.cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, marker_size=args.marker_size, mat_save_name='last_epoch', \
                        TB=writer, subset=args.plotting_subset, X_train=X_train.cpu().detach().numpy())
elif args.dataset_name == 'robot_1':
    # post processing
    # clamp the trajectories to be within bounds
    I_dummy = [0, 7]
    I_keep = [i for i in range(d) if i not in I_dummy]
    a    = torch.tensor(space.bound)
    low  = a[:,0][I_keep]
    high = a[:,1][I_keep]
    X_sample = X_sample.cpu().detach().numpy()
    X_sample = np.clip(X_sample, low, high) # N x K x 10
    # add the dummy dimenions back
    X_sample = process_robot_data(X_sample, dataset=args.dataset_name, mode='postprocess') # N x K x 12
    # get all the feasible trajectories
    I = [traj_is_Feasible(space, traj) for traj in X_sample]
    X_sample = X_sample[I]
    # save the feasible trajectories
    traj_path = os.path.join(plot_data_dir, 'traj_density.npy')
    np.save(traj_path, X_sample)

    # forward and save the single trajectory that starts from the actual robot location
    traj_robot = flow._transform.inverse(x_0.unsqueeze(0))[3]
    traj_robot = partition_hist(traj_robot, args, mode='gen', \
                    partition_mode=args.OT_part, LU_last=args.LU_last).squeeze(0) # K x 10
    traj_robot = process_robot_data(traj_robot, dataset=args.dataset_name, mode='postprocess') # K x 12

    traj_robot_interp = interpolate_hist(traj_robot.unsqueeze(0), args).squeeze(0)
    traj_robot_path = os.path.join(plot_data_dir, 'traj_robot.npy')
    traj_robot_interp_path = os.path.join(plot_data_dir, 'traj_robot_interp.npy')
    np.save(traj_robot_path, traj_robot.detach().cpu().numpy())
    np.save(traj_robot_interp_path, traj_robot_interp.detach().cpu().numpy())
    if args.robot_1_base_dist == 'two_init':
        traj_default_start = flow._transform.inverse(x_default_start.unsqueeze(0))[3]
        traj_default_start = partition_hist(traj_default_start, args, mode='gen', \
                        partition_mode=args.OT_part, LU_last=args.LU_last).squeeze(0) # K x 10
        traj_default_start = process_robot_data(traj_default_start, dataset=args.dataset_name, mode='postprocess') # K x 12
        traj_under_start = flow._transform.inverse(x_under_start.unsqueeze(0))[3]
        traj_under_start = partition_hist(traj_under_start, args, mode='gen', \
                        partition_mode=args.OT_part, LU_last=args.LU_last).squeeze(0) # K x 10
        traj_under_start = process_robot_data(traj_under_start, dataset=args.dataset_name, mode='postprocess') # K x 12
        traj_default_start_path = os.path.join(plot_data_dir, 'traj_default_start.npy')
        traj_under_start_path   = os.path.join(plot_data_dir, 'traj_under_start.npy')
        np.save(traj_default_start_path, traj_default_start.detach().cpu().numpy())
        np.save(traj_under_start_path, traj_under_start.detach().cpu().numpy())

else:
    x_min = -10
    x_max = 10
    y_min = -10
    y_max = 10
    plot_evolution(args, X_sample.cpu().detach().numpy(), log_dir, x_min, x_max, \
                    y_min, y_max, marker_size=args.marker_size, mat_save_name='last_epoch',\
                    TB=writer, subset=args.plotting_subset, X_train=X_train.cpu().detach().numpy())

# =================================================================================== #
#                                       Logging                                       #
# =================================================================================== #

print ('Experiment name: {}'.format(args.exp_name))

# if args.dataset_name == 'gaussian_mixture' or args.dataset_name == 'crowd_motion_gaussian'\
#     or args.dataset_name == 'moons':
#     N = args.num_test_data
# else:
#     N = len(test_dataset)


# save training trajectories if desired
if args.save_train_traj:
    N_save = 100000
    tbar = tqdm(range(N_save // args.train_batch_size))
    hist_array = torch.Tensor([])
    for step in tbar:
        flow.train()
        _, _, _, hist, _, _ = flow.sample(args.train_batch_size)
        hist = partition_hist(hist, args, mode='gen', LU_last=args.LU_last, partition_mode=args.OT_part)
        hist_array = torch.cat([hist_array, hist.detach().cpu()])

    path = os.path.join(log_dir, 'train_traj_{}.pt'.format(args.dataset_name))
    torch.save(hist_array, path)


if args.dataset_name in ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']:
    N = len(test_dataset)
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


# s = 'Bound for the NF Lipschitz constant: {:.4f}'.format(float(lipschitz_NF))
# print (s)
filename = os.path.join(log_dir, 'test-results.txt')
with open(filename, 'a') as file:
    file.write(s)
    file.write('\n')
test_result_str = test_result_str + s + '  \n'

if tbx_logging:
    writer.add_text(tag='test_results', text_string=test_result_str)

