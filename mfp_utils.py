import warnings
import numpy as np
from numpy.lib.npyio import _save_dispatcher
import torch
import torch.distributions as D
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import scipy
from sklearn.decomposition import PCA
from sklearn.manifold import *
from sklearn.datasets import make_moons, make_s_curve, make_swiss_roll
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch import optim
from nde.transforms import coupling
from nde import distributions
from mfg_bilevel_models import Obstacle_indicator, Obstacle_robot, Obstacle, Obstacle_cuboid
import pylab

device = torch.device('cuda')


# =================================================================================== #
#                                  MFG Computations                                   #
# =================================================================================== #



# def construct_Q(args, device):
#     if args.dataset_name == 'crowd_motion_gaussian':
#         mean = torch.zeros(2).cuda(device)
#         diag = torch.Tensor([1., 0.5]).cuda(device)
#         cov  = torch.diag_embed(diag)
#         Q = D.MultivariateNormal(mean, cov)
#     elif args.dataset_name == 'crowd_motion_gaussian_nonsmooth_obs':
#         Q = Obstacle_indicator(args).to(device)
#     elif args.dataset_name in ['crowd_motion_two_bars', 'crowd_motion_two_bars_bilevel',\
#                                'crowd_motion_two_bars_uniform', 'crowd_motion_two_bars_uniform_bilevel']:
#         # coordinates for the corners of the bars
#         points = [(-5, 0.9, 1.15, 1.35), (1.1, 7, 0.65, 0.85)]
#         # points = [(-2, 0.5, 1.1, 1.3), (0.5, 3, 0.7, 0.9)]
#         Q = Obstacle_cuboid(points, args.two_bars_sharpness)
#     elif args.dataset_name in ['crowd_motion_gaussian_two_bars']:
#         num_gaussians = 2
#         weight = D.Categorical(torch.ones(num_gaussians,).to(device))
#         mean = torch.tensor([[-2,1.2], [4,0.8]]).to(device) # N_m x d
#         cov  = torch.tensor([[1,0], [0,1e-2]]).unsqueeze(0).repeat(num_gaussians,1,1).to(device)
#         dist = D.MultivariateNormal(mean.to(device), cov)
#         # mixture = D.MixtureSameFamily(weight, dist)
#         # Q = distributions.Mixture((dim,), mixture)
#         Q = D.MixtureSameFamily(weight, dist)
#     elif args.dataset_name == 'crowd_motion_gaussian_close':
#         e_2    = torch.zeros(2).to(device)
#         e_2[1] = 1.
#         mean = 0.5 * e_2
#         diag = 0.02 * torch.Tensor([1., 0.5]).cuda(device)
#         cov  = torch.diag_embed(diag)
#         Q = D.MultivariateNormal(mean, cov)
#     elif args.dataset_name == 'crowd_motion_gaussian_NN_obs':
#         Q = Obstacle(args.gaussian_multi_dim, args).to(device)
#         pretrain_obs_dir = './results/crowd_motion_gaussian_bilevel/pretrain_obs.t'
#         Q.load_state_dict(torch.load(pretrain_obs_dir))
#         print ("Loaded obstacle from: {}".format(pretrain_obs_dir))
#         # disable training for the obstacle
#         for p in Q.parameters():
#             p.requires_grad = False
#     elif args.dataset_name == 'drones_22_obs':
#         mean = torch.Tensor([args.obs_mean_x_22, args.obs_mean_y_22]).cuda(device)
#         diag = torch.Tensor([args.obs_var_x_22, args.obs_var_y_22]).cuda(device)
#         cov  = torch.diag_embed(diag)
#         Q = D.MultivariateNormal(mean, cov)
#     elif args.dataset_name == 'robot_1':
#         d_in = 10 # removed 2 dummy dimensions from the original 12
#         h = 512
#         Q = Obstacle_robot(d_in, h ,args).to(device)
#         Q.load_state_dict(torch.load(args.obs_dir))
#         # disable training for the obstacle
#         for p in Q.parameters():
#             p.requires_grad = False
#     else:
#         Q = None
    
#     return Q


def compute_F(args, Q, hist, logp0, ld, Q_is_dist=True, W=None, scheme='right_pt', dataset=None, pad_ld=True):
    """Computes the MFG interaction cost

    Args:
        Q (distribution): obstacle
        hist (tensor): B x (K+1) x d
        logp0 (tensor): B x 1
        ld (tensor): B x (K+1) x 1
        dataset (str, optional): name of dataset. Defaults to None.
        W(tensor): for fast computation, no particular meaning.
    Returns:
        F_E, F_P: B x 1
    """

    B = hist.shape[0]
    F_E, F_P = torch.zeros(B,1).to(hist.device), torch.zeros(B,1).to(hist.device)
    if pad_ld:
        # the initial ld is logdet(I) = 0
        ld = torch.cat([torch.zeros_like(ld[:,-1,:].unsqueeze(1)), ld], dim=1) # B x (K+1) x 1
    if args.interp_hist:
        hist = interpolate_hist(hist, args)
        ld   = interpolate_hist(ld, args)
    
    # assume hist is correctly partitioned
    K   = hist.shape[1] - 1
    if scheme == 'right_pt':
        F_E = logp0 - torch.sum(ld, dim=[1,2]) / K
        # evaluate the first two components on Q
        if Q_is_dist:
            F_P = 50 * torch.sum(torch.exp(Q.log_prob(hist[:,:,:args.Q_dim])), dim=-1) / K
        else:
            # the constant 50 is in Q
            F_P = torch.sum(Q(hist[:,:,:args.Q_dim].reshape(-1,args.Q_dim)).reshape(B, -1), dim=-1) / K
    elif scheme == 'simp':
        # if pad_ld:
        #     # the initial ld is logdet(I) = 0
        #     ld_pad = torch.cat([torch.zeros_like(ld[:,-1,:].unsqueeze(1)), ld], dim=1).squeeze(-1)
        # else:
        #     ld_pad = ld.squeeze(-1)
        ld = ld.squeeze(-1) # B x (K+1)
        W_simp = torch.zeros(hist.shape[0], K+1).to(hist.device) # B x (K+1)
        W_simp[:, torch.arange(K // 2+1)*2] = 2.0 
        W_simp[:, torch.arange(K // 2)*2+1] = 4.0
        W_simp[:,0] = W_simp[:,-1] = 1.0
        F_E = logp0 - torch.sum(W_simp*ld, dim=-1) / (3*K)
        if Q_is_dist:
            F_P = 50 * torch.sum(W_simp * torch.exp(Q.log_prob(hist[:,:,:args.Q_dim])), dim=-1) / (3*K)
        else:
            h = hist[:,:,:args.Q_dim].reshape(-1, args.Q_dim) # B*K x d_Q
            # the constant 50 is in Q
            F_P = torch.sum(W_simp * Q(h).reshape(B, -1), dim=-1) / (3*K)
    else:
        raise NotImplementedError()

    return F_E, F_P


def compute_F_drones(args, hist, Q=None, dataset=None, scheme='right_pt'):
    # hist: N_p x B x K+1 x d
    N_p, B, K, d = hist.shape[0], hist.shape[1], hist.shape[2] - 1, hist.shape[3]
    # assume hist is correctly partitioned
    H = hist.unsqueeze(0) - hist.unsqueeze(1) # N_p x N_p x B x K x d
    H = torch.exp(-1/2*torch.norm(H, dim=-1)**2)  # N_p x N_p x B x K

    # collision cost between groups
    if scheme == 'right_pt':
        F_inter  = torch.sum(H, dim=-1) / K # N_p x N_p x B
    elif scheme == 'simp':
        # W   = torch.zeros(H_masked.shape[0], H_masked.shape[1], K+1) # N_p*(N_p+1)/2-N_p x B x (K+1)
        W   = torch.zeros_like(H) # N_p x N_p x B x K
        W[:,:,:, torch.arange(K // 2+1)*2] = 2.0 
        W[:,:,:, torch.arange(K // 2)*2+1] = 4.0
        W[:,:,:,0] = W[:,:,:,-1] = 1.0
        F_inter  = torch.sum(W * H, dim=-1) / (3*K) # N_p x N_p x B
    else:
        raise NotImplementedError()

    mask    = torch.triu(torch.ones(N_p, N_p), diagonal=1).reshape(N_p,N_p,1).repeat(1,1,B).to(hist.device)
    F_inter = torch.masked_select(F_inter, mask.bool()).sum() / B
    
    # collision cost with obstacle 
    F_obs = torch.tensor(0.).to(hist.device)
    if Q is not None:
        if scheme == 'right_pt':
            F_obs = torch.sum(torch.exp(Q.log_prob(hist[:,:,:,:2])), dim=-1) / K # N_p x B
        elif scheme == 'simp':
            W = torch.zeros_like(H) # N_p x B x K
            W[:,:, torch.arange(K // 2+1)*2] = 2.0 
            W[:,:, torch.arange(K // 2)*2+1] = 4.0
            W[:,:,0] = W[:,:,:,-1] = 1.0
            F_obs = 50 * torch.sum(W * torch.exp(Q.log_prob(hist[:,:,:,:2])), dim=-1) / (3*K) # N_p x B
        else:
            raise NotImplementedError()
        # group dimension is summed over, batch dimension is averged over.
        F_obs = torch.sum(F_obs, dim=0).mean()

    return F_inter, F_obs


def partition_hist(hist, args, mode='norm', partition_mode='block', LU_last=True, hist_type='traj', multi_pop=False):
    assert len(hist) != 0
    if multi_pop:
        # hist is already processed to be N_p x B x K x d
        part = hist
        K    = part.shape[2]
    else:
        part = torch.cat(hist).reshape(len(hist), hist[0].shape[0], hist[0].shape[-1]).permute(1,0,2) # B x K x d
        K    = part.shape[1]

    if partition_mode =='block':
        NSF_block_length  = 5
        L_block_length    = 3
    elif partition_mode == 'block_CL_no_perm':
        if args.NF_model == 'single_flow':
            NSF_block_length  = 5
            L_block_length    = 1
        else:
            NSF_block_length  = 4
            L_block_length    = 1
    elif partition_mode == 'block_AR_no_perm':
        NSF_block_length  = 3
        L_block_length    = 1
    else: # just return the whole history
        return part

    if args.NF_model == 'single_flow':
        if hist_type == 'traj':
            if mode == 'norm':
                I    = torch.arange(-1, K, step=NSF_block_length)
                I[0] = 0
                if LU_last:
                    raise NotImplementedError()
            elif mode == 'gen':
                if LU_last:
                    raise NotImplementedError()
                else:
                    I    = torch.arange(-1, K, step=NSF_block_length)
                    I[0] = 0
            else:
                raise NotImplementedError()
            
            assert I[-1] == K-1
            part = part[:,I,:]

        elif hist_type == 'ld':
            if mode == 'norm':
                raise NotImplementedError()
            elif mode == 'gen':
                if LU_last:
                    raise NotImplementedError()
                else:
                    I   = torch.arange(2, K, step=NSF_block_length)
                    Ip1 = torch.arange(3, K, step=NSF_block_length)
            else:
                raise NotImplementedError()
            part = part[:,I,:] + part[:,Ip1,:] # The starting ld is not included, because it'd be all 0's
        else:
            raise NotImplementedError()
    else:
        if hist_type == 'traj':
            if mode == 'norm':
                I = torch.arange(0, K, step=NSF_block_length)
                if LU_last:
                    I = torch.cat((I, torch.tensor(I[-1]+L_block_length).unsqueeze(0)))
            elif mode == 'gen':
                if LU_last:
                    I = torch.arange(L_block_length, K, step=NSF_block_length)
                    I = torch.cat((torch.tensor(0).unsqueeze(0), I))
                else:
                    I = torch.arange(0, K, step=NSF_block_length)
            else:
                raise NotImplementedError()
            
            assert I[-1] == K-1
            if multi_pop:
                part = part[:,:,I,:]
            else:
                part = part[:,I,:]

        elif hist_type == 'ld':
            if mode == 'norm':
                raise NotImplementedError()
            elif mode == 'gen':
                if LU_last:
                    raise NotImplementedError()
                else:
                    I   = torch.arange(2, K, step=NSF_block_length)
                    Ip1 = torch.arange(3, K, step=NSF_block_length)
            else:
                raise NotImplementedError()
            part = part[:,I,:] + part[:,Ip1,:] # The starting ld is not included, because it'd be all 0's
        else:
            raise NotImplementedError()

    return part


def compute_OT_cost(hist, args, mode='norm', partition_mode='block', LU_last=True, scheme='forward', part_hist=True):
    if part_hist:
        hist_part = partition_hist(hist, args, mode=mode, partition_mode=partition_mode, LU_last=LU_last) # B x (K+1) x d
    else:
        hist_part = hist

    if args.interp_hist:
        hist_part = interpolate_hist(hist_part, args)
    K = hist_part.shape[1]-1

    # for MFG with **no interaction costs**, the solution trajectory is straight, so we can 
    # directly compute the transport cost by taking E_P0 [|T(x)-x|^2]
    if args.OT_comp == 'monge':
        L_cost = torch.norm(hist_part[:,-1,:] - hist_part[:,0,:], dim=-1)**2 # B
    else:
        if scheme == 'forward':
            I = torch.arange(K)
            # don't aggregate over the batch dimension
            # equivalent to taking a sqr norm in dim 2 and sum over dim 1
            L_cost = K * torch.norm(hist_part[:,I,:] - hist_part[:,I+1,:], dim=[1,2])**2 # B 
        elif scheme == 'centered':
            # pad with ghost pt at the start
            I         = torch.arange(0, K)
            hist_pad  = torch.cat([hist_part[:,0,:].unsqueeze(1), hist_part], dim=1)
            L_cost    = K/4 * torch.norm(hist_pad[:,I-1,:] - hist_pad[:,I+1,:], dim=[1,2])**2
        elif scheme == 'forward_2nd':
            # pad with ghost pt at the end
            hist_pad  = torch.cat([hist_part, hist_part[:,-1,:].unsqueeze(1)], dim=1)
            I         = torch.arange(0, K)
            L_cost    = K * torch.norm(-3/2*hist_pad[:,I,:] + 2*hist_pad[:,I+1,:] - 1/2*hist_pad[:,I+2,:], dim=[1,2])**2
        elif scheme == 'FD4_simp':
            hist_pad  = torch.cat([hist_part, hist_part[:,-1,:].unsqueeze(1).repeat(1,4,1)], dim=1)
            I         = torch.arange(0, K+1)
            L_i       = K**2 * torch.norm(-25/12*hist_pad[:,I,:] + 4*hist_pad[:,I+1,:] - 3*hist_pad[:,I+2,:] \
                            + 4/3*hist_pad[:,I+3,:] - 1/4*hist_pad[:,I+4,:], dim=-1)**2 # B x (K+1)
            W = torch.zeros(hist_pad.shape[0], K+1).to(device) # B x (K+1)
            W[:, torch.arange(K // 2+1)*2] = 2.0 
            W[:, torch.arange(K // 2)*2+1] = 4.0
            W[:,0] = W[:,-1] = 1.0
            L_i    = W * L_i 
            L_cost = 1/(3*K) * torch.sum(L_i, dim=1)
        elif scheme == 'FD4_simp_symmetric':
            # pad in both directions
            hist_pad  = torch.cat([hist_part[:,0,:].unsqueeze(1).repeat(1,4,1), \
                            hist_part, hist_part[:,-1,:].unsqueeze(1).repeat(1,4,1)], dim=1)
            I         = torch.arange(0, K+1+4)
            L_i       = K**2 * torch.norm(-25/12*hist_pad[:,I,:] + 4*hist_pad[:,I+1,:] - 3*hist_pad[:,I+2,:] \
                            + 4/3*hist_pad[:,I+3,:] - 1/4*hist_pad[:,I+4,:], dim=-1)**2 # B x (K+1)
            W = torch.zeros(hist_pad.shape[0], K+1+4) # B x (K+1)
            W[:, torch.arange((K+4) // 2+1)*2] = 2.0 
            W[:, torch.arange((K+4) // 2)*2+1] = 4.0
            W[:,0] = W[:,-1] = 1.0
            L_i    = W * L_i 
            L_cost = 1/(3*K) * torch.sum(L_i, dim=1)
        elif scheme == 'FD1_simp':
            hist_pad  = torch.cat([hist_part, hist_part[:,-1,:].unsqueeze(1).repeat(1,1,1)], dim=1)
            I         = torch.arange(0, K+1)
            # FD1
            L_i       = K**2 * torch.norm(hist_pad[:,I,:] - hist_pad[:,I+1,:], dim=-1)**2 # B x (K+1)
            # simp
            W = torch.zeros(hist_pad.shape[0], K+1) # B x (K+1)
            W[:, torch.arange(K // 2+1)*2] = 2.0 
            W[:, torch.arange(K // 2)*2+1] = 4.0
            W[:,0] = W[:,-1] = 1.0
            L_i    = W * L_i 
            L_cost = 1/(3*K) * torch.sum(L_i, dim=1)
        else:
            raise NotImplementedError()

    return L_cost, hist_part


def compute_OT_cost_multi(hist, args, mode='norm', partition_mode='block', LU_last=True, scheme='forward'):
    hist_part = partition_hist(hist, args, mode=mode, \
                    partition_mode=partition_mode, LU_last=LU_last, multi_pop=True) # N_p x B x (K+1) x d
    K = hist_part.shape[2]-1

    if scheme == 'forward':
        I         = torch.arange(K)
        # # don't aggregate over the batch dimension
        # # equivalent to taking a sqr norm in dim 2 and sum over dim 1
        L_cost    = K * torch.norm(hist_part[:,:,I,:] - hist_part[:,:,I+1,:], dim=[2,3])**2 # N_p x B
    elif scheme == 'FD4_simp':
        hist_pad  = torch.cat([hist_part, hist_part[:,:,-1,:].unsqueeze(2).repeat(1,1,4,1)], dim=2)
        I         = torch.arange(0, K+1)
        L_i       = K**2 * torch.norm(-25/12*hist_pad[:,:,I,:] + 4*hist_pad[:,:,I+1,:] - 3*hist_pad[:,:,I+2,:] \
                        + 4/3*hist_pad[:,:,I+3,:] - 1/4*hist_pad[:,:,I+4,:], dim=-1)**2 # N_p x B x (K+1)
        W = torch.zeros(hist_pad.shape[0], hist_pad.shape[1], K+1).to(device) # N_p x B x (K+1)
        W[:,:, torch.arange(K // 2+1)*2] = 2.0 
        W[:,:, torch.arange(K // 2)*2+1] = 4.0
        W[:,:,0] = W[:,:,-1] = 1.0
        L_i    = W * L_i 
        L_cost = 1/(3*K) * torch.sum(L_i, dim=2) # N_p x B
    else:
        raise NotImplementedError()

    return L_cost, hist_part


def interpolate_hist(hist_orig, args):
    # hist: N x (K+1) x d
    # hist_interp: B x K*(n_interp-1)+1 x d

    B = hist_orig.shape[0]
    K = hist_orig.shape[1]-1
    d = hist_orig.shape[2]
    I = torch.arange(K)

    hist = hist_orig.permute(0,2,1) # B x d x (K+1)
    hist_gap = hist[:,:,I+1] - hist[:,:,I] # B x d x K
    interp_vec = torch.linspace(0,1,args.n_interp).reshape(1,1,1,-1).to(hist_orig.device) # B x d x K x n_interp
    hist_gap = hist_gap.unsqueeze(-1) * interp_vec # B x d x K x n_interp
    hist_last = hist[:, :, -1].unsqueeze(-1) # B x d x 1
    hist = hist[:,:,:-1].unsqueeze(-1) # B x d x K x 1

    hist_interp = hist + hist_gap # B x d x K x n_interp
    # remove the last interpolated point in each segment
    # since it coincides with the first point in the next segment
    hist_interp = hist_interp[:, :, :, :-1] # B x d x K x (n_interp-1)
    hist_interp = torch.cat((hist_interp.reshape(B, d, -1), hist_last), dim=-1) # B x d x K*(n_interp-1)+1
    hist_interp = hist_interp.permute(0,2,1) # B x K*(n_interp-1)+1 x d

    return hist_interp

# =================================================================================== #
#                                      Datasets                                       #
# =================================================================================== #
def make_spr(n_samples):
    z = torch.randn(n_samples, 2)
    n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * np.pi) / 360
    d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
    d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
    x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
    return x + 0.1*z

def make_gaussian_mixture_data(base_dist, dim, num_train_data, num_val_data, num_test_data, \
                                    train_batch_size, val_batch_size, test_batch_size, weight=None):
    
    # build the gaussian mixture
    if base_dist == 'gaussian':
        num_mixtures          = 8
        if weight is None:
            weight = D.Categorical(torch.ones(num_mixtures,).to(device))
        angle                 = np.pi/4*(torch.arange(8)+1).unsqueeze(1) # N_m x 1
        e_1                   = torch.zeros(1,dim)
        e_1[:,0]              = 1.
        e_2                   = torch.zeros(1,dim)
        e_2[:,1]              = 1.

        mean                  = 4 * (torch.cos(angle)*e_1.repeat(num_mixtures,1) + torch.sin(angle)*e_2.repeat(num_mixtures,1)) # N_m x d
        cov                   = 0.3 * torch.eye(dim).unsqueeze(0).repeat(num_mixtures,1,1)
        dist                  = D.MultivariateNormal(mean.to(device), cov.to(device))
        gaussian_target       = D.MixtureSameFamily(weight, dist)
    else:
        cov             = 0.3 * torch.eye(dim).to(device)
        mean            = torch.zeros(dim).to(device)
        gaussian_target = D.MultivariateNormal(mean, cov)

    # get samples from the dist as training data
    X_train             = torch.zeros(num_train_data, dim)
    X_val               = torch.zeros(num_val_data, dim)
    X_test              = torch.zeros(num_test_data, dim)
    sample_batch_size   = 100
    N_train             = int (num_train_data // sample_batch_size)
    N_val               = int (num_val_data // sample_batch_size)
    N_test              = int (num_test_data // sample_batch_size)
    

    for i in range(N_train):
        x = gaussian_target.sample((sample_batch_size,))
        X_train[i*sample_batch_size:(i+1)*sample_batch_size, :] = x
    for i in range(N_val):
        x = gaussian_target.sample((sample_batch_size,))
        X_val[i*sample_batch_size:(i+1)*sample_batch_size, :]   = x
    for i in range(N_test):
        x = gaussian_target.sample((sample_batch_size,))
        X_test[i*sample_batch_size:(i+1)*sample_batch_size, :]  = x
    
    # # make them into pytorch dataloaders
    train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    val_data_loader   = DataLoader(X_val, batch_size=val_batch_size, shuffle=True)
    test_data_loader  = DataLoader(X_test, batch_size=test_batch_size, shuffle=True)

    # train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    # val_data_loader   = DataLoader(X_val, batch_size=val_batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    # test_data_loader  = DataLoader(X_test, batch_size=test_batch_size, shuffle=True, generator=torch.Generator(device='cuda'))


    return X_train, X_val, X_test, train_data_loader, val_data_loader, test_data_loader, gaussian_target


def make_crowd_motion_gaussian_data(args, dim, num_train_data, num_val_data, num_test_data, \
                                    train_batch_size, val_batch_size, test_batch_size):

    # if args.dataset_name == 'crowd_motion_gaussian_close':
    #     mean   = torch.zeros(dim).to(device)
    #     cov    = 0.01 * torch.eye(dim).to(device)
    # elif args.dataset_name in ['crowd_motion_two_bars', 'crowd_motion_two_bars_uniform']:
    #     mean   = torch.tensor([1.3, 0.25]).to(device)
    #     # mean   = torch.tensor([1., 1.]).to(device)
    #     cov    = 0.01 * torch.eye(dim).to(device)
    # elif args.dataset_name in ['crowd_motion_gaussian_two_bars']:    
    #     mean   = torch.tensor([1.3, 0.35]).to(device)
    #     cov    = 0.01 * torch.eye(dim).to(device)
    # elif args.dataset_name in ['crowd_motion_gaussian_two_bars_uniform']:
    #     mean   = torch.tensor([1., 1.]).to(device)
    #     cov    = 0.01 * torch.eye(dim).to(device)
    # elif args.dataset_name in ['crowd_motion_gaussian_one_bar_uniform']:
    #     mean   = torch.tensor([1.3, 0.25]).to(device)
    #     cov    = 0.01 * torch.eye(dim).to(device)
    # else:
    #     e_2    = torch.zeros(dim)
    #     e_2[1] = 1.
    #     mean   = -3*e_2.to(device)
    #     cov    = 0.3 * torch.eye(dim).to(device)
    # gaussian_target = D.MultivariateNormal(mean, cov)

    gaussian_target = create_target_dist(args, dim)

    # get samples from the dist as training data
    X_train             = torch.zeros(num_train_data, dim)
    X_val               = torch.zeros(num_val_data, dim)
    X_test              = torch.zeros(num_test_data, dim)
    sample_batch_size   = 100
    N_train             = int (num_train_data // sample_batch_size)
    N_val               = int (num_val_data // sample_batch_size)
    N_test              = int (num_test_data // sample_batch_size)
    

    for i in range(N_train):
        # x = gaussian_target.sample((sample_batch_size,))
        x = gaussian_target.sample(sample_batch_size)
        X_train[i*sample_batch_size:(i+1)*sample_batch_size, :] = x
    for i in range(N_val):
        # x = gaussian_target.sample((sample_batch_size,))
        x = gaussian_target.sample(sample_batch_size)
        X_val[i*sample_batch_size:(i+1)*sample_batch_size, :]   = x
    for i in range(N_test):
        # x = gaussian_target.sample((sample_batch_size,))
        x = gaussian_target.sample(sample_batch_size)
        X_test[i*sample_batch_size:(i+1)*sample_batch_size, :]  = x
    
    # make them into pytorch dataloaders
    train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    val_data_loader   = DataLoader(X_val, batch_size=val_batch_size, shuffle=True)
    test_data_loader  = DataLoader(X_test, batch_size=test_batch_size, shuffle=True)


    return X_train, X_val, X_test, train_data_loader, val_data_loader, test_data_loader, gaussian_target


def make_syn_data(num_train_data, num_val_data, num_test_data, \
                    train_batch_size, val_batch_size, test_batch_size, noise=0.1, name='moons'):

    if name == 'moons':
        X_train = torch.Tensor(make_moons(num_train_data, shuffle=True, noise=noise)[0]).float()
        X_val   = torch.Tensor(make_moons(num_val_data, shuffle=True, noise=noise)[0]).float() 
        X_test  = torch.Tensor(make_moons(num_test_data, shuffle=True, noise=noise)[0]).float()     

        # # make them into pytorch dataloaders
        # train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
        # val_data_loader   = DataLoader(X_val, batch_size=val_batch_size, shuffle=True)
        # test_data_loader  = DataLoader(X_test, batch_size=test_batch_size, shuffle=True)

    elif name == 'checkerboard':
        def make_cbd(n_samples):
            x1 = torch.rand(n_samples) * 4 - 2
            x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
            x2 = x2_ + x1.floor() % 2
            return torch.stack([x1, x2], dim=1) * 2

        X_train = torch.Tensor(make_cbd(num_train_data)).float()
        X_val   = torch.Tensor(make_cbd(num_val_data)).float() 
        X_test  = torch.Tensor(make_cbd(num_test_data)).float() 

    elif name == '2spirals':
        # def make_spr(n_samples):
        #     z = torch.randn(n_samples, 2)
        #     n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * np.pi) / 360
        #     d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        #     d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        #     x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
        #                 torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        #     return x + 0.1*z

        X_train = torch.Tensor(make_spr(num_train_data)).float()
        X_val   = torch.Tensor(make_spr(num_val_data)).float() 
        X_test  = torch.Tensor(make_spr(num_test_data)).float()  

    # make them into pytorch dataloaders
    train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    val_data_loader   = DataLoader(X_val, batch_size=val_batch_size, shuffle=True)
    test_data_loader  = DataLoader(X_test, batch_size=test_batch_size, shuffle=True)


    return X_train, X_val, X_test, train_data_loader, val_data_loader, test_data_loader


def make_gaussian_data(a, dim, num_train_data, num_val_data, num_test_data, \
                                    train_batch_size, val_batch_size, test_batch_size):

    target_dist = D.MultivariateNormal(torch.tensor([a] * dim).to(device), 0.01*torch.eye(dim).to(device))
    # sample incrementally
    X_train             = torch.zeros(num_train_data, dim)
    X_val               = torch.zeros(num_val_data, dim)
    X_test              = torch.zeros(num_test_data, dim)
    sample_batch_size   = 100
    N_train             = int (num_train_data // sample_batch_size)
    N_val               = int (num_val_data // sample_batch_size)
    N_test              = int (num_test_data // sample_batch_size)
    for i in range(N_train):
        x = target_dist.sample((sample_batch_size,))
        X_train[i*sample_batch_size:(i+1)*sample_batch_size, :] = x
    for i in range(N_val):
        x = target_dist.sample((sample_batch_size,))
        X_val[i*sample_batch_size:(i+1)*sample_batch_size, :] = x
    for i in range(N_test):
        x = target_dist.sample((sample_batch_size,))
        X_test[i*sample_batch_size:(i+1)*sample_batch_size, :] = x
    
    train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    val_data_loader   = DataLoader(X_val,   batch_size=val_batch_size, shuffle=True)
    test_data_loader  = DataLoader(X_test,  batch_size=test_batch_size, shuffle=True)


    return X_train, X_val, X_test, train_data_loader, val_data_loader, test_data_loader, target_dist


def make_3gaussian_data(a, dim, num_train_data, num_val_data, num_test_data, \
                                    train_batch_size, val_batch_size, test_batch_size):

    target_dist_2 = D.MultivariateNormal(torch.tensor([a] * dim), torch.eye(dim))
    target_dist_1 = D.MultivariateNormal(torch.tensor([0.] * (dim-1) + [a]), torch.eye(dim))
    # sample incrementally
    sample_batch_size = 1
    X_train = []
    X_val   = []
    X_test  = []
    N_train = int (num_train_data // sample_batch_size)
    N_val   = int (num_val_data // sample_batch_size)
    N_test  = int (num_test_data // sample_batch_size)
    # for i in range(N_train):
    #     x_1 = target_dist_1.sample((sample_batch_size,)).squeeze(0)
    #     x_2 = target_dist_2.sample((sample_batch_size,)).squeeze(0)
    #     X_train.append([x_1,x_2])
    # for i in range(N_val):
    #     x_1 = target_dist_1.sample((sample_batch_size,)).squeeze(0)
    #     x_2 = target_dist_2.sample((sample_batch_size,)).squeeze(0)
    #     X_val.append([x_1,x_2])
    # for i in range(N_test):
    #     x_1 = target_dist_1.sample((sample_batch_size,)).squeeze(0)
    #     x_2 = target_dist_2.sample((sample_batch_size,)).squeeze(0)
    #     X_test.append([x_1,x_2])
    
    # train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    # val_data_loader   = DataLoader(X_val,   batch_size=val_batch_size, shuffle=True)
    # test_data_loader  = DataLoader(X_test,  batch_size=test_batch_size, shuffle=True)

    X_train_1 = torch.cat([target_dist_1.sample((sample_batch_size,)).squeeze(0) for i in range(N_train)])
    X_train_2 = torch.cat([target_dist_2.sample((sample_batch_size,)).squeeze(0) for i in range(N_train)])
    X_val_1   = torch.cat([target_dist_1.sample((sample_batch_size,)).squeeze(0) for i in range(N_val)])
    X_val_2   = torch.cat([target_dist_2.sample((sample_batch_size,)).squeeze(0) for i in range(N_val)])
    X_test_1  = torch.cat([target_dist_1.sample((sample_batch_size,)).squeeze(0) for i in range(N_test)])
    X_test_2  = torch.cat([target_dist_2.sample((sample_batch_size,)).squeeze(0) for i in range(N_test)])

    train_data_loader_1 = DataLoader(X_train_1, batch_size=train_batch_size, shuffle=True)
    val_data_loader_1   = DataLoader(X_val_1,   batch_size=val_batch_size, shuffle=True)
    test_data_loader_1  = DataLoader(X_test_1,  batch_size=test_batch_size, shuffle=True)

    train_data_loader_2 = DataLoader(X_train_2, batch_size=train_batch_size, shuffle=True)
    val_data_loader_2   = DataLoader(X_val_2,   batch_size=val_batch_size, shuffle=True)
    test_data_loader_2  = DataLoader(X_test_2,  batch_size=test_batch_size, shuffle=True)


    return X_train_1, X_val_1, X_test_1, X_train_2, X_val_2, X_test_2, \
        train_data_loader_1, val_data_loader_1, test_data_loader_1, train_data_loader_2, val_data_loader_2, test_data_loader_2


def make_syn_data_twoNFs(num_train_data, num_val_data, num_test_data, \
                    train_batch_size, val_batch_size, test_batch_size, noise=0.1, name='moons'):

    if name == 'S_swiss':
        X_train_1, _ = make_s_curve(num_train_data, noise=noise)
        X_val_1, _   = make_s_curve(num_val_data, noise=noise)
        X_test_1, _  = make_s_curve(num_test_data, noise=noise)

        X_train_2, _ = make_swiss_roll(num_train_data, noise=noise)
        X_val_2, _   = make_swiss_roll(num_val_data, noise=noise)
        X_test_2, _  = make_swiss_roll(num_test_data, noise=noise)

        # only take the 2D part
        X_train_1 = np.stack((X_train_1[:,0], X_train_1[:,2]), axis=-1)
        X_val_1   = np.stack((X_val_1[:,0], X_val_1[:,2]), axis=-1)
        X_test_1  = np.stack((X_test_1[:,0], X_test_1[:,2]), axis=-1)
        X_train_2 = np.stack((X_train_2[:,0], X_train_2[:,2]), axis=-1) / 5
        X_val_2   = np.stack((X_val_2[:,0], X_val_2[:,2]), axis=-1) / 5
        X_test_2  = np.stack((X_test_2[:,0], X_test_2[:,2]), axis=-1) / 5
    elif name == 'moons_spiral':
        # moons
        X_train_1 = make_moons(num_train_data, shuffle=True, noise=noise)[0]
        X_val_1   = make_moons(num_val_data, shuffle=True, noise=noise)[0]
        X_test_1  = make_moons(num_test_data, shuffle=True, noise=noise)[0]

        # spiral
        X_train_2 = make_spr(num_train_data)
        X_val_2   = make_spr(num_val_data)
        X_test_2  = make_spr(num_test_data)
    else:
        raise NotImplementedError()

    # # create pairs of training data
    # X_train = []
    # X_val   = []
    # X_test  = []        
    # for i in range(len(X_train_1)):
    #     X_train.append([X_train_1[i], X_train_2[i]])
    # for i in range(len(X_val_1)):
    #     X_val.append([X_val_1[i], X_val_2[i]])
    # for i in range(len(X_test_1)):
    #     # x_1 = np.array([X_test_1[i][0], X_test_1[i][2]])
    #     # x_2 = np.array([X_test_2[i][0], X_test_2[i][2]]) / 5
    #     X_test.append([X_test_1[i], X_test_2[i]])

    # train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    # val_data_loader   = DataLoader(X_val,   batch_size=val_batch_size, shuffle=True)
    # test_data_loader  = DataLoader(X_test,  batch_size=test_batch_size, shuffle=True)

    train_data_loader_1 = DataLoader(X_train_1, batch_size=train_batch_size, shuffle=True)
    val_data_loader_1   = DataLoader(X_val_1,   batch_size=val_batch_size, shuffle=True)
    test_data_loader_1  = DataLoader(X_test_1,  batch_size=test_batch_size, shuffle=True)

    train_data_loader_2 = DataLoader(X_train_2, batch_size=train_batch_size, shuffle=True)
    val_data_loader_2   = DataLoader(X_val_2,   batch_size=val_batch_size, shuffle=True)
    test_data_loader_2  = DataLoader(X_test_2,  batch_size=test_batch_size, shuffle=True)

    return X_train_1, X_val_1, X_test_1, X_train_2, X_val_2, X_test_2, \
        train_data_loader_1, val_data_loader_1, test_data_loader_1, train_data_loader_2, val_data_loader_2, test_data_loader_2


def make_mnist_mnist_data(train_batch_size, val_batch_size, test_batch_size, latent=True):
    log_dir   = './results/vae_mnist'
    assert os.path.exists(log_dir)
    if latent:
        X_train_1 = np.load(os.path.join(log_dir, 'latent_01234_train.npy'))
        X_train_2 = np.load(os.path.join(log_dir, 'latent_56789_train.npy'))

        # we re-used the training data here. 
        # Don't look at the statistics associated with validation data - we don't need them.
        X_val_1 = np.load(os.path.join(log_dir, 'latent_01234_train.npy'))
        X_val_2 = np.load(os.path.join(log_dir, 'latent_56789_train.npy'))

        X_test_1 = np.load(os.path.join(log_dir, 'latent_01234_test.npy'))
        X_test_2 = np.load(os.path.join(log_dir, 'latent_56789_test.npy'))
    else:
        X_train_1 = np.load(os.path.join(log_dir, 'img_01234_train.npy'))
        X_train_2 = np.load(os.path.join(log_dir, 'img_56789_train.npy'))

        # we re-used the training data here. 
        # Don't look at the statistics associated with validation data - we don't need them.
        X_val_1 = np.load(os.path.join(log_dir, 'img_01234_train.npy'))
        X_val_2 = np.load(os.path.join(log_dir, 'img_56789_train.npy'))

        X_test_1 = np.load(os.path.join(log_dir, 'img_01234_test.npy'))
        X_test_2 = np.load(os.path.join(log_dir, 'img_56789_test.npy'))

    # X_train = []
    # X_val   = []
    # X_test  = []        
    # for i in range(len(X_train_1)):
    #     X_train.append([X_train_1[i], X_train_2[i]])
    # for i in range(len(X_val_1)):
    #     X_val.append([X_val_1[i], X_val_2[i]])
    # for i in range(len(X_test_1)):
    #     X_test.append([X_test_1[i], X_test_2[i]])

    train_data_loader_1 = DataLoader(X_train_1, batch_size=train_batch_size, shuffle=True)
    val_data_loader_1   = DataLoader(X_val_1,   batch_size=val_batch_size, shuffle=True)
    test_data_loader_1  = DataLoader(X_test_1,  batch_size=test_batch_size, shuffle=True)

    train_data_loader_2 = DataLoader(X_train_2, batch_size=train_batch_size, shuffle=True)
    val_data_loader_2   = DataLoader(X_val_2,   batch_size=val_batch_size, shuffle=True)
    test_data_loader_2  = DataLoader(X_test_2,  batch_size=test_batch_size, shuffle=True)

    return X_train_1, X_val_1, X_test_1, X_train_2, X_val_2, X_test_2, \
        train_data_loader_1, val_data_loader_1, test_data_loader_1, train_data_loader_2, val_data_loader_2, test_data_loader_2


def make_drone_data(args, num_train_data, num_val_data, num_test_data, \
                        train_batch_size, val_batch_size, test_batch_size, var_drones=0.01, radius_82=4, name='drones_22'):

    if name == 'drones_22':
        dim = 2
        N_p = 2
        cov = var_drones * torch.eye(dim)
        target_dist_1 = D.MultivariateNormal(torch.Tensor([1,1]).to(device), cov.to(device))
        target_dist_2 = D.MultivariateNormal(torch.Tensor([0,1]).to(device), cov.to(device))
        target_dist = [target_dist_1, target_dist_2]
    elif name == 'drones_23':
        dim = 3
        N_p = 2
        cov = var_drones * torch.eye(dim)
        target_dist_1 = D.MultivariateNormal(torch.Tensor([1,1,1]).to(device), cov.to(device))
        target_dist_2 = D.MultivariateNormal(torch.Tensor([0,1,1]).to(device), cov.to(device))
        target_dist = [target_dist_1, target_dist_2]
    elif name == 'drones_82':
        dim = 2
        N_p = 8
        cov = var_drones * torch.eye(dim).to(device)
        angle     = np.pi/4*(torch.arange(8)+1).unsqueeze(1) # N_m x 1
        e_1       = torch.zeros(1,dim)
        e_1[:,0]  = 1.
        e_2       = torch.zeros(1,dim)
        e_2[:,1]  = 1.
        mean      = radius_82 * (torch.cos(angle)*e_1.repeat(N_p,1) + \
                            torch.sin(angle)*e_2.repeat(N_p,1)).to(device) # N_m x d
        target_dist = [D.MultivariateNormal(mean[i], cov) for i in range(N_p)]
        # target_dist_1 = D.MultivariateNormal(torch.Tensor([1,1,1]), cov)
        # target_dist_2 = D.MultivariateNormal(torch.Tensor([0,1,1]), cov)
        # target_dist = [target_dist_1, target_dist_2]
    elif name == 'drones_22_obs':
        dim = 2
        N_p = 2
        cov = var_drones * torch.eye(dim)
        target_dist_1 = D.MultivariateNormal(torch.Tensor([0, args.drones_22_obs_mu_y]).to(device), cov.to(device))
        target_dist_2 = D.MultivariateNormal(torch.Tensor([1, args.drones_22_obs_mu_y]).to(device), cov.to(device))
        target_dist = [target_dist_1, target_dist_2]
    else:
        raise NotImplementedError()

    # get samples from the dist as training data
    sample_batch_size   = 100
    N_train             = int (num_train_data // sample_batch_size)
    N_val               = int (num_val_data // sample_batch_size)
    N_test              = int (num_test_data // sample_batch_size)
    
    X_train = torch.stack(
        [torch.cat(
                [target_dist[j].sample((sample_batch_size,)).squeeze(0) for i in range(N_train)]
            ) for j in range(N_p)]
        ) # N_p x N x d
    
    X_val = torch.stack(
        [torch.cat(
                [target_dist[j].sample((sample_batch_size,)).squeeze(0) for i in range(N_val)]
            ) for j in range(N_p)]
        ) # N_p x N x d

    X_test = torch.stack(
        [torch.cat(
                [target_dist[j].sample((sample_batch_size,)).squeeze(0) for i in range(N_test)]
            ) for j in range(N_p)]
        ) # N_p x N x d

    # for i in range(N_train):
    #     x = gaussian_target.sample((sample_batch_size,))
    #     X_train[i*sample_batch_size:(i+1)*sample_batch_size, :] = x
    # for i in range(N_val):
    #     x = gaussian_target.sample((sample_batch_size,))
    #     X_val[i*sample_batch_size:(i+1)*sample_batch_size, :]   = x
    # for i in range(N_test):
    #     x = gaussian_target.sample((sample_batch_size,))
    #     X_test[i*sample_batch_size:(i+1)*sample_batch_size, :]  = x
    
    # make them into pytorch dataloaders
    train_data_loader = [DataLoader(X_train[i], batch_size=train_batch_size, shuffle=True) for i in range(N_p)]
    val_data_loader   = [DataLoader(X_val[i], batch_size=val_batch_size, shuffle=True) for i in range(N_p)]
    test_data_loader  = [DataLoader(X_test[i], batch_size=test_batch_size, shuffle=True) for i in range(N_p)]


    return X_train, X_val, X_test, train_data_loader, val_data_loader, test_data_loader, target_dist


def load_bilevel_data(args, n_train, n_test, dim, data_dir, train_batch_size, val_batch_size, test_batch_size):
    # if name in ['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong']:
    #     e_2    = torch.zeros(dim)
    #     e_2[1] = 1.
    #     mean   = -3*e_2.to(device)
    #     cov    = 0.3 * torch.eye(dim).to(device)
    # elif name in ['crowd_motion_two_bars_bilevel', 'crowd_motion_two_bars_uniform_bilevel']:
    #     mean   = torch.tensor([1.3, 0.25]).to(device)
    #     cov    = 0.01 * torch.eye(dim).to(device)
    # elif name in ['crowd_motion_gaussian_two_bars_uniform_bilevel']:
    #     mean   = torch.tensor([1., 1.]).to(device)
    #     cov    = 0.01 * torch.eye(dim).to(device)
    # else:
    #     raise NotImplementedError()
    # gaussian_target = distributions.MultivarNormal((dim,), mean=mean, cov=cov)

    P_1 = create_target_dist(args, dim)

    X = torch.load(data_dir)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    X_train, X_test = train_test_split(X_train, test_size=0.2/0.8, random_state=42)
    X_train = X_train[:n_train]
    X_test  = X_test[:n_test]
    
    train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    val_data_loader   = DataLoader(X_val,   batch_size=val_batch_size, shuffle=True)
    test_data_loader  = DataLoader(X_test,  batch_size=test_batch_size, shuffle=True)

    return X_train, X_val, X_test, train_data_loader, val_data_loader, test_data_loader, P_1    


def make_robot_data(x_K, var, dim, num_train_data, num_val_data, num_test_data, \
                                    train_batch_size, val_batch_size, test_batch_size):
    
    # target dist
    cov             = var * torch.eye(dim).to(device)
    mean            = x_K
    gaussian_target = D.MultivariateNormal(mean, cov)

    # get samples from the dist as training data
    X_train             = torch.zeros(num_train_data, dim)
    X_val               = torch.zeros(num_val_data, dim)
    X_test              = torch.zeros(num_test_data, dim)
    sample_batch_size   = 100
    N_train             = int (num_train_data // sample_batch_size)
    N_val               = int (num_val_data // sample_batch_size)
    N_test              = int (num_test_data // sample_batch_size)
    
    for i in range(N_train):
        x = gaussian_target.sample((sample_batch_size,))
        X_train[i*sample_batch_size:(i+1)*sample_batch_size, :] = x
    for i in range(N_val):
        x = gaussian_target.sample((sample_batch_size,))
        X_val[i*sample_batch_size:(i+1)*sample_batch_size, :]   = x
    for i in range(N_test):
        x = gaussian_target.sample((sample_batch_size,))
        X_test[i*sample_batch_size:(i+1)*sample_batch_size, :]  = x
    
    # # make them into pytorch dataloaders
    train_data_loader = DataLoader(X_train, batch_size=train_batch_size, shuffle=True)
    val_data_loader   = DataLoader(X_val, batch_size=val_batch_size, shuffle=True)
    test_data_loader  = DataLoader(X_test, batch_size=test_batch_size, shuffle=True)

    return X_train, X_val, X_test, train_data_loader, val_data_loader, test_data_loader, gaussian_target



def create_target_dist(args, dim):
    if args.dataset_name == 'crowd_motion_gaussian_close':
        mean   = torch.zeros(dim).to(device)
        cov    = 0.01 * torch.eye(dim).to(device)
    elif args.dataset_name in ['crowd_motion_two_bars', 'crowd_motion_two_bars_uniform', 'crowd_motion_two_bars_bilevel', 'crowd_motion_two_bars_uniform_bilevel']:
        mean   = torch.tensor([1.3, 0.25]).to(device)
        # mean   = torch.tensor([1., 1.]).to(device)
        cov    = 0.01 * torch.eye(dim).to(device)
    elif args.dataset_name in ['crowd_motion_gaussian_two_bars']:    
        mean   = torch.tensor([1.3, 0.35]).to(device)
        cov    = 0.01 * torch.eye(dim).to(device)
    elif args.dataset_name in ['crowd_motion_gaussian_two_bars_uniform', 'crowd_motion_gaussian_two_bars_uniform_bilevel',\
                                'crowd_motion_gaussian_two_bars_gaussian', 'crowd_motion_gaussian_two_bars_gaussian_bilevel', \
                                'crowd_motion_two_bars_gaussian']:
        mean   = torch.tensor([1., 1.]).to(device)
        cov    = 0.01 * torch.eye(dim).to(device)
    elif args.dataset_name in ['crowd_motion_gaussian_one_bar_uniform', 'crowd_motion_gaussian_one_bar_uniform_bilevel']:
        mean   = torch.tensor([0.25, 1.75]).to(device)
        cov    = 0.01 * torch.eye(dim).to(device)
    elif args.dataset_name in ['crowd_motion_gaussian', 'crowd_motion_gaussian_bilevel', 
            'crowd_motion_gaussian_bilevel_strong', 'crowd_motion_flower']:
        e_2    = torch.zeros(dim)
        e_2[1] = 1.
        mean   = -3*e_2.to(device)
        cov    = 0.3 * torch.eye(dim).to(device)
    else:
        raise NotImplementedError()

    # P_1 = D.MultivariateNormal(mean, cov)
    P_1 = distributions.MultivarNormal((dim,), mean=mean, cov=cov)

    return P_1


def create_base_dist(args, features):
    if args.dataset_name == 'gaussian_mixture':
        if args.mixture_base == 'gaussian':
            cov          = 0.3 * torch.eye(features).to(device)
            mean         = torch.zeros(features).to(device)
            P_0 = distributions.MultivarNormal((features,), mean=mean, cov=cov)
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
            P_0          = distributions.Mixture((features,), mixture)
    elif args.dataset_name in ['crowd_motion_gaussian', 'crowd_motion_gaussian_nonsmooth_obs', \
                                'crowd_motion_gaussian_NN_obs', 'crowd_motion_gaussian_bilevel', \
                                'crowd_motion_gaussian_bilevel_strong', 'crowd_motion_flower']:
        e_2           = torch.zeros(args.gaussian_multi_dim).to(device)
        e_2[1]        = 1.
        mean          = 3*e_2
        cov           = 0.3 * torch.eye(args.gaussian_multi_dim).to(device)
        P_0  = distributions.MultivarNormal((features,), mean=mean, cov=cov)
    elif args.dataset_name in ['crowd_motion_two_bars', 'crowd_motion_two_bars_bilevel']:
        # mean = torch.tensor([0.35, 1.65]).to(device)
        # mean = torch.tensor([0.25, 1.65]).to(device)
        mean = torch.tensor([0.7, 1.75]).to(device)
        cov  = 0.01 * torch.eye(args.gaussian_multi_dim).to(device)
        P_0  = distributions.MultivarNormal((features,), mean=mean, cov=cov)
    elif args.dataset_name in ['crowd_motion_two_bars_uniform', 'crowd_motion_two_bars_uniform_bilevel',
                              'crowd_motion_gaussian_two_bars_uniform', 'crowd_motion_gaussian_two_bars_uniform_bilevel',
                              'crowd_motion_gaussian_one_bar_uniform']:
        P_0  = distributions.Uniform(torch.tensor([0.,0.]).to(device),torch.tensor([2.,2.]).to(device))
    elif args.dataset_name in ['crowd_motion_gaussian_two_bars_gaussian', 'crowd_motion_gaussian_two_bars_gaussian_bilevel',
                            'crowd_motion_two_bars_gaussian']:
        mean = torch.tensor([1., 1.]).to(device)
        cov  = 2. * torch.eye(args.gaussian_multi_dim).to(device)
        P_0  = distributions.MultivarNormal((features,), mean=mean, cov=cov)
    elif args.dataset_name == 'crowd_motion_gaussian_two_bars':
        mean = torch.tensor([0.7, 1.65]).to(device)
        cov  = 0.01 * torch.eye(args.gaussian_multi_dim).to(device)
        P_0  = distributions.MultivarNormal((features,), mean=mean, cov=cov)
    elif args.dataset_name == 'gaussian':
        cov  = 0.01 * torch.eye(args.gaussian_multi_dim).to(device)
        P_0  = distributions.MultivarNormal((features,), cov=cov)
    else:
        P_0 = distributions.StandardNormal((features,))

    return P_0

# =================================================================================== #
#                                      Plotting                                       #
# =================================================================================== #

def embed_data(X_sample, X_train=None, method=None, n_components=2):
    # new containers
    N = X_sample.shape[0]
    K = X_sample.shape[1]
    X_sample_emb = np.zeros((N,K,n_components))
    X_train_emb  = np.zeros((N,n_components))

    method    = eval(method)
    try:
        embedding = method(n_components=n_components, n_jobs=-1)
    except:
        embedding = method(n_components=n_components)
    
    # use the training data to fit the embeddings.
    if X_train is not None:
        embedding.fit(X_train)
        X_train_emb = embedding.transform(X_train)
    else:
        embedding.fit(X_sample[:,0,:])

    for i in range(X_sample.shape[1]):
        X_sample_emb[:,i,:] = embedding.transform(X_sample[:,i,:])


    return X_sample_emb, X_train_emb


def generate_landmarks(d, dataset='crowd_motion_gaussian'):
    assert d >= 2
    if dataset in ['crowd_motion_gaussian', 'crowd_motion_gaussian_nonsmooth_obs', 'crowd_motion_gaussian_NN_obs',
                    'crowd_motion_flower']: 
        # The initial gaussian is supported roughly on B([0,3], 1.5)
        # four points on the same y-level as mean
        x_0 = torch.Tensor([[-0.75,3], [-0.25,3], [0.25,3], [0.75,3]])
        # two points in the back
        x_1 = torch.Tensor([[-0.5,3.5], [0.5,3.5]])
        # two points in the front
        x_2 = torch.Tensor([[-0.35,2.5], [0.35,2.5]])
        X   = torch.cat([x_0, x_1, x_2]) # N x 2
    elif dataset == 'crowd_motion_gaussian_close':
        # four points on the same y-level as mean
        x_0 = torch.Tensor([[-0.2,1], [-0.1,1], [0.1,1], [0.2,1]])
        # two points in the back
        x_1 = torch.Tensor([[-0.1,1.1], [0.1,1.1]])
        # two points in the front
        x_2 = torch.Tensor([[-0.08, 0.9], [0.08, 0.9]])
        X   = torch.cat([x_0, x_1, x_2]) # N x 2
    elif dataset in ['crowd_motion_two_bars']:
        # # four points on the same y-level as mean
        # x_0 = torch.Tensor([[0.2,1.65], [0.3,1.65], [0.4,1.65], [0.5,1.65]])
        # # two points in the back
        # x_1 = torch.Tensor([[0.25,1.75], [0.45,1.75]])
        # # two points in the front
        # x_2 = torch.Tensor([[0.25, 1.55], [0.45, 1.55]])
        # four points on the same y-level as mean
        x_0 = torch.Tensor([[0.6,1.75], [0.65,1.75], [0.75,1.75], [0.8,1.75]])
        # two points in the back
        x_1 = torch.Tensor([[0.625,1.85], [0.775,1.85]])
        # two points in the front
        x_2 = torch.Tensor([[0.625, 1.65], [0.775, 1.65]])
        X   = torch.cat([x_0, x_1, x_2]) # N x 2
    elif dataset == 'crowd_motion_gaussian_two_bars':
        # four points on the same y-level as mean
        x_0 = torch.Tensor([[0.6,1.65], [0.65,1.65], [0.75,1.65], [0.8,1.65]])
        # two points in the back
        x_1 = torch.Tensor([[0.625,1.75], [0.775,1.75]])
        # two points in the front
        x_2 = torch.Tensor([[0.625, 1.55], [0.775, 1.55]])
        X   = torch.cat([x_0, x_1, x_2]) # N x 2
    elif dataset in ['crowd_motion_two_bars_uniform', 'crowd_motion_gaussian_two_bars_uniform', \
                    'crowd_motion_gaussian_one_bar_uniform', 'crowd_motion_gaussian_two_bars_gaussian',
                    'crowd_motion_two_bars_gaussian']:
        x_0 = torch.Tensor([[0.25,0.25], [0.25,1.75], [1.75,0.25], [1.75,1.75]])
        x_1 = torch.Tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5], [0.25, 0.75], [1.75, 1.5]])
        x_2 = torch.Tensor([[0.25, 1.5], [1.75, 0.5], [0.25, 1.0], [1.75, 1.0], [1., 0.5], [1., 1.5]])
        X   = torch.cat([x_0, x_1, x_2]) # N x 2
    elif dataset == 'gaussian_mixture':
        # 16 points equally spaced in theta
        N       = 16
        r       = 0.5
        theta_0 = 0.01
        theta   = torch.linspace(theta_0, theta_0 + 2*np.pi,N+1)
        x,y     = torch.cos(theta), torch.sin(theta)
        X       = r * torch.stack([x,y]).transpose(0,1) # N x 2
    else:
        raise NotImplementedError()

    # pad with zeros for higher dim
    d_pad = d - X.shape[1]
    X     = torch.cat([X, torch.zeros(X.shape[0], d_pad)], dim=-1)
    
    return X.to(device)


def plot_evolution(args, X_sample, plot_dir, x_min, x_max, y_min, y_max, subset=0, save_data=True, X_train=None, \
                    marker_size=5, mat_save_name='last_epoch', dim_reduction=False, reduction_method=None, TB=None):
    # X_sample: N x K x d
    subset = int(subset)
    if subset != 0:
        X_sample = X_sample[:subset]
        if X_train is not None:
            X_train  = X_train[:subset]

    if dim_reduction:
        assert (reduction_method is not None)
        X_sample, X_train = embed_data(X_sample, X_train, method=reduction_method)
    
    K = X_sample.shape[1]

    # coloring
    if args.color == 'order':
        t = np.arange(X_sample.shape[0])
    elif args.color == 'radius':
        plt.set_cmap('jet')
        # we base all trajectory's color on the initial radius to track the evolution of points
        t = np.linalg.norm(X_sample[:,0,:], axis=-1)
    else:
        raise NotImplementedError()

    # plots
    for i in range(K):
        f = plt.figure(figsize=(5,5))
        plt_whole     = f.add_subplot(111)
        plt_whole.scatter(X_sample[:,i,0], X_sample[:,i,1], marker='.', c=t, s=marker_size)
        # last flow, plot ground truth to compare
        if i == K-1 and X_train is not None: 
            plt_whole.scatter(X_train[:,0], X_train[:,1], marker='.', color='lightgrey', s=marker_size, alpha=0.5)
        plt_whole.set_xlabel(r'$x$')
        plt_whole.set_ylabel(r'$y$')
        plt_whole.set_title('Sampled Data, all')
        if dim_reduction:
            fig_name = 'all_{}_{}.png'.format(reduction_method, i)
        else:
            fig_name = 'all_{}.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

    # plots on a fixed range
    for i in range(K):
        f = plt.figure(figsize=(5,5))
        plt_sameScale = f.add_subplot(111)

        plt_sameScale.scatter(X_sample[:,i,0], X_sample[:,i,1], marker='.', c=t, s=marker_size)
        # if i == K-1: # last flow
        #     plt_sameScale.scatter(X_train[:,0], X_train[:,1], color='lightgrey', s=marker_size)
        plt_sameScale.set_xlabel(r'$x$')
        plt_sameScale.set_ylabel(r'$y$')
        plt_sameScale.set_xlim([x_min, x_max])
        plt_sameScale.set_ylim([y_min, y_max])
        plt_sameScale.set_title('Sampled Data, Same Scale')

        if dim_reduction:
            fig_name = '{}_{}.png'.format(reduction_method, i)
        else:
            fig_name = '{}.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

    # sample sample trajectories and dataset
    if save_data:
        train_save_path  = os.path.join(plot_dir, mat_save_name + '_train.mat')
        sample_save_path = os.path.join(plot_dir, mat_save_name + '_sample_trajectory.mat')
        if X_train is not None:
            scipy.io.savemat(train_save_path,  dict(data=X_train))
        scipy.io.savemat(sample_save_path, dict(data=X_sample))


def plot_evolution_twoNFs(args, X_sample_01, X_sample_12, plot_dir, x_min, x_max, y_min, y_max, subset=0, \
                            save_data=True, X_1_train=None, X_2_train=None, marker_size=5, mat_save_name='last_epoch', \
                            dim_reduction=False, reduction_method=None, TB=None):
    # X_sample: N x K x d
    subset = int(subset)
    if subset != 0:
        X_sample_01 = X_sample_01[:subset]
        X_sample_12 = X_sample_12[:subset]
        if X_1_train is not None:
            X_1_train  = X_1_train[:subset]
        if X_2_train is not None:
            X_2_train  = X_2_train[:subset]

    if dim_reduction:
        assert (reduction_method is not None)
        X_sample_01, X_train = embed_data(X_sample_01, X_1_train, method=reduction_method)
        X_sample_12, X_train = embed_data(X_sample_12, X_2_train, method=reduction_method)
    
    K = X_sample_12.shape[1]

    # coloring
    if args.color == 'order':
        t = np.arange(X_sample_12.shape[0])
    elif args.color == 'radius':
        plt.set_cmap('jet')
        # we base all trajectory's color on the initial radius to track the evolution of points
        t = np.linalg.norm(X_sample_12[:,0,:], axis=-1)
    else:
        raise NotImplementedError()

    # plots
    for i in range(K):
        # P0 to P1
        f = plt.figure(figsize=(5,5))
        plt_whole     = f.add_subplot(111)
        plt_whole.scatter(X_sample_01[:,i,0], X_sample_01[:,i,1], marker='.', c=t, s=marker_size)
        # last flow, plot ground truth to compare
        if i == K-1 and X_1_train is not None: 
            plt_whole.scatter(X_1_train[:,0], X_1_train[:,1], marker='.', color='lightgrey', s=marker_size, alpha=0.5)
        plt_whole.set_xlabel(r'$x$')
        plt_whole.set_ylabel(r'$y$')
        plt_whole.set_title('Sampled Data P0->P1, all')
        if dim_reduction:
            fig_name = 'all_{}_{}_P0P1.png'.format(reduction_method, i)
        else:
            fig_name = 'all_{}_P0P1.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_P0P1_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

        # P1 to P2
        f = plt.figure(figsize=(5,5))
        plt_whole     = f.add_subplot(111)
        plt_whole.scatter(X_sample_12[:,i,0], X_sample_12[:,i,1], marker='.', c=t, s=marker_size)
        # last flow, plot ground truth to compare
        if i == K-1 and X_2_train is not None: 
            plt_whole.scatter(X_2_train[:,0], X_2_train[:,1], marker='.', color='lightgrey', s=marker_size, alpha=0.5)
        plt_whole.set_xlabel(r'$x$')
        plt_whole.set_ylabel(r'$y$')
        plt_whole.set_title('Sampled Data P1->P2, all')
        if dim_reduction:
            fig_name = 'all_{}_{}_P1P2.png'.format(reduction_method, i)
        else:
            fig_name = 'all_{}_P1P2.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_P1P2_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

    # plots on a fixed range
    for i in range(K):
        f = plt.figure(figsize=(5,5))
        plt_sameScale = f.add_subplot(111)

        plt_sameScale.scatter(X_sample_01[:,i,0], X_sample_01[:,i,1], marker='.', c=t, s=marker_size)
        # if i == K-1: # last flow
        #     plt_sameScale.scatter(X_train[:,0], X_train[:,1], color='lightgrey', s=marker_size)
        plt_sameScale.set_xlabel(r'$x$')
        plt_sameScale.set_ylabel(r'$y$')
        plt_sameScale.set_xlim([x_min, x_max])
        plt_sameScale.set_ylim([y_min, y_max])
        plt_sameScale.set_title('Sampled Data P0->P1, Same Scale')

        if dim_reduction:
            fig_name = '{}_{}_P0P1.png'.format(reduction_method, i)
        else:
            fig_name = '{}_P0P1.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_P0P1_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

        # P1 to P2
        f = plt.figure(figsize=(5,5))
        plt_sameScale = f.add_subplot(111)

        plt_sameScale.scatter(X_sample_12[:,i,0], X_sample_12[:,i,1], marker='.', c=t, s=marker_size)
        # if i == K-1: # last flow
        #     plt_sameScale.scatter(X_train[:,0], X_train[:,1], color='lightgrey', s=marker_size)
        plt_sameScale.set_xlabel(r'$x$')
        plt_sameScale.set_ylabel(r'$y$')
        plt_sameScale.set_xlim([x_min, x_max])
        plt_sameScale.set_ylim([y_min, y_max])
        plt_sameScale.set_title('Sampled Data P1->P2, Same Scale')

        if dim_reduction:
            fig_name = '{}_{}_P1P2.png'.format(reduction_method, i)
        else:
            fig_name = '{}_P1P2.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_P1P2_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

    # sample sample trajectories and dataset
    if save_data:
        train_save_path_1  = os.path.join(plot_dir, mat_save_name + '_train_P1.mat')
        train_save_path_2  = os.path.join(plot_dir, mat_save_name + '_train_P2.mat')
        sample_save_path_1 = os.path.join(plot_dir, mat_save_name + '_sample_trajectory_P0P1.mat')
        sample_save_path_2 = os.path.join(plot_dir, mat_save_name + '_sample_trajectory_P1P2.mat')
        if X_1_train is not None:
            scipy.io.savemat(train_save_path_1,  dict(data=X_1_train))
        if X_2_train is not None:
            scipy.io.savemat(train_save_path_2,  dict(data=X_2_train))
        scipy.io.savemat(sample_save_path_1, dict(data=X_sample_01))
        scipy.io.savemat(sample_save_path_2, dict(data=X_sample_12))


### multi-population ###
def generate_landmarks_multi(cov, radius=2, dataset='drones_22'):
    if dataset in ['drones_22', 'drones_22_obs']: # these two datasets share the same P_0's
        # 8 points equally spaced in theta
        cov     = 1e-2
        N       = 6
        theta_0 = 0.01
        theta   = torch.linspace(theta_0, theta_0 + 2*np.pi,N+1)
        x,y     = torch.cos(theta), torch.sin(theta)

        mu_1 = torch.Tensor([0,0])
        mu_2 = torch.Tensor([1,0])
        X_1  = cov**(1/2)*torch.stack([x,y]).transpose(0,1) + mu_1 # N x d
        X_2  = cov**(1/2)*torch.stack([x,y]).transpose(0,1) + mu_2 # N x d
        X    = torch.stack([X_1, X_2]) # N_p x N x d 

    elif dataset == 'drones_82':
        N_p = 8
        # + np.pi to make the destination of each cluster to be diametrically across
        angle     = np.pi/4*(torch.arange(8)+1).unsqueeze(1) + np.pi # N_m x 1
        e_1       = torch.zeros(1,2)
        e_1[:,0]  = 1.
        e_2       = torch.zeros(1,2)
        e_2[:,1]  = 1.
        means     = radius * (torch.cos(angle)*e_1.repeat(N_p,1) + \
                            torch.sin(angle)*e_2.repeat(N_p,1)) # N_m x 2

        N       = 3
        theta_0 = 0.01
        # theta   = torch.linspace(theta_0, theta_0 + 2*np.pi,N+1)
        # x,y     = torch.cos(theta), torch.sin(theta)
        # x_y     = cov**(1/2)*torch.stack([x,y]).transpose(0,1) # N x 2

        X = torch.zeros(N_p, N+1, 2)
        for i in range(N_p):
            theta = torch.linspace(theta_0 + float(angle[i]), theta_0 + 2*np.pi + float(angle[i]), N+1)
            x,y   = torch.cos(theta), torch.sin(theta)
            x_y   = cov**(1/2)*torch.stack([x,y]).transpose(0,1) # N x 2
            X[i]  = x_y + means[i].unsqueeze(0)

    elif dataset == 'drones_23':
        # 8 points equally spaced in theta
        cov     = 1e-2
        N       = 6
        theta_0 = 0.01
        theta   = torch.linspace(theta_0, theta_0 + 2*np.pi,N+1)
        x,y     = torch.cos(theta), torch.sin(theta)
        z       = torch.zeros_like(x)

        mu_1 = torch.Tensor([0,0,0])
        mu_2 = torch.Tensor([1,0,0])
        X_1  = cov**(1/2)*torch.stack([x,y,z]).transpose(0,1) + mu_1 # N x d
        X_2  = cov**(1/2)*torch.stack([x,y,z]).transpose(0,1) + mu_2 # N x d
        X    = torch.stack([X_1, X_2]) # N_p x N x d 
    else:
        raise NotImplementedError()

    return X.to(device)


def plot_evolution_multi(args, X_sample, plot_dir, x_min, x_max, y_min, y_max, subset=0, save_data=True, X_train=None, \
                    marker_size=5, mat_save_name='last_epoch', dim_reduction=False, reduction_method=None, TB=None):
    # X_sample: N_p x N x K x d
    subset = int(subset)
    if subset != 0:
        X_sample = X_sample[:subset]
        if X_train is not None:
            X_train  = X_train[:subset]

    N_p, K = X_sample.shape[0], X_sample.shape[2]

    # coloring
    if args.color == 'order':
        t = np.arange(X_sample.shape[0])
    elif args.color == 'radius':
        plt.set_cmap('jet')
        # we base all trajectory's color on the initial radius to track the evolution of points
        t = np.linalg.norm(X_sample[:,0,:], axis=-1)
    else:
        raise NotImplementedError()

    # plots
    for i in range(K):
        f = plt.figure(figsize=(5,5))
        plt_whole = f.add_subplot(111)
        for j in range(N_p):
            # plt_whole.scatter(X_sample[j,:,i,0], X_sample[j,:,i,1], marker='.', c=t, s=marker_size)
            plt_whole.scatter(X_sample[j,:,i,0], X_sample[j,:,i,1], marker='.', s=marker_size)
            # last flow, plot ground truth to compare
            if i == K-1 and X_train is not None: 
                plt_whole.scatter(X_train[j,:,0], X_train[j,:,1], marker='.', color='lightgrey', s=marker_size, alpha=0.5)
        plt_whole.set_xlabel(r'$x$')
        plt_whole.set_ylabel(r'$y$')
        plt_whole.set_title('Sampled Data, all')
        if dim_reduction:
            fig_name = 'all_{}_{}.png'.format(reduction_method, i)
        else:
            fig_name = 'all_{}.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

    # plots on a fixed range
    for i in range(K):
        f = plt.figure(figsize=(5,5))
        plt_sameScale = f.add_subplot(111)
        for j in range(N_p):
            # plt_sameScale.scatter(X_sample[j,:,i,0], X_sample[j,:,i,1], marker='.', c=t, s=marker_size)
            plt_sameScale.scatter(X_sample[j,:,i,0], X_sample[j,:,i,1], marker='.', s=marker_size)
        # if i == K-1: # last flow
        #     plt_sameScale.scatter(X_train[:,0], X_train[:,1], color='lightgrey', s=marker_size)
        plt_sameScale.set_xlabel(r'$x$')
        plt_sameScale.set_ylabel(r'$y$')
        plt_sameScale.set_xlim([x_min, x_max])
        plt_sameScale.set_ylim([y_min, y_max])
        plt_sameScale.set_title('Sampled Data, Same Scale')

        if dim_reduction:
            fig_name = '{}_{}.png'.format(reduction_method, i)
        else:
            fig_name = '{}.png'.format(i)

        save_dir = os.path.join(plot_dir, 'sampling_trajectory_' + mat_save_name + '_' + fig_name)
        f.savefig(save_dir)
        # TB
        if TB is not None:
            TB.add_figure(fig_name, f)
        plt.close()

    # sample sample trajectories and dataset
    if save_data:
        train_save_path  = os.path.join(plot_dir, mat_save_name + '_train.mat')
        sample_save_path = os.path.join(plot_dir, mat_save_name + '_sample_trajectory.mat')
        if X_train is not None:
            scipy.io.savemat(train_save_path,  dict(data=X_train))
        scipy.io.savemat(sample_save_path, dict(data=X_sample))


def plot_evolution_multi_3D(args, X_sample, plot_dir, x_min, x_max, y_min, y_max, z_min, z_max,\
                                subset=0, save_data=True, X_train=None, marker_size=5, mat_save_name='last_epoch', TB=None):
    # X_sample: N_p x N x K x d
    subset = int(subset)
    if subset != 0:
        X_sample = X_sample[:subset]
        if X_train is not None:
            X_train  = X_train[:subset]

    N_p, K = X_sample.shape[0], X_sample.shape[2]

    # coloring
    if args.color == 'order':
        t = np.arange(X_sample.shape[0])
    elif args.color == 'radius':
        plt.set_cmap('jet')
        # we base all trajectory's color on the initial radius to track the evolution of points
        t = np.linalg.norm(X_sample[:,0,:], axis=-1)
    else:
        raise NotImplementedError()

    # plots
    # vertical, horizontal
    view_angles = [[15, 60], [30, 60], [30, 75]]
    num_angles  = len(view_angles)
    for i in range(K):
        for a in range(num_angles):
            f = plt.figure(figsize=(5,5))
            plt_whole     = f.add_subplot(111, projection='3d')
            for j in range(N_p): 
                plt_whole.scatter3D(X_sample[j,:,i,0], X_sample[j,:,i,1], X_sample[j,:,i,2], marker='.', s=marker_size)
                # last flow, plot ground truth to compare
                if i == K-1 and X_train is not None: 
                    # plt_whole.scatter(X_train[j,:,0], X_train[j,:,1], marker='.', color='lightgrey', s=marker_size, alpha=0.5)
                    plt_whole.scatter3D(X_train[j,:,0], X_train[j,:,1], X_train[j,:,2], \
                        marker='.', color='lightgrey', s=marker_size, alpha=0.5)
            plt_whole.set_xlabel(r'$x$')
            plt_whole.set_ylabel(r'$y$')
            plt_whole.set_zlabel(r'$z$')
            plt_whole.set_title('Sampled Data, all')
            plt_whole.view_init(*view_angles[a])
            fig_name = 'angle_{}_all_{}.png'.format(a, i)

            save_dir = os.path.join(plot_dir, 'sampling_trajectory_' + mat_save_name + '_' + fig_name)
            f.savefig(save_dir)
            # TB
            if TB is not None:
                TB.add_figure(fig_name, f)
            plt.close()

    # plots on a fixed range
    for i in range(K):
        for a in range(num_angles):
            f = plt.figure(figsize=(5,5))
            plt_sameScale = f.add_subplot(111, projection='3d')
            for j in range(N_p):
                plt_sameScale.scatter3D(X_sample[j,:,i,0], X_sample[j,:,i,1], X_sample[j,:,i,2], marker='.', s=marker_size)
            plt_sameScale.set_xlabel(r'$x$')
            plt_sameScale.set_ylabel(r'$y$')
            plt_sameScale.set_zlabel(r'$z$')
            plt_sameScale.set_xlim([x_min, x_max])
            plt_sameScale.set_ylim([y_min, y_max])
            plt_sameScale.set_zlim([z_min, z_max])
            plt_sameScale.set_title('Sampled Data, Same Scale')
            plt_sameScale.view_init(*view_angles[a])
            fig_name = 'angle_{}_{}.png'.format(a,i)

            save_dir = os.path.join(plot_dir, 'sampling_trajectory_' + mat_save_name + '_' + fig_name)
            f.savefig(save_dir)
            # TB
            if TB is not None:
                TB.add_figure(fig_name, f)
            plt.close()

    # sample sample trajectories and dataset
    if save_data:
        train_save_path  = os.path.join(plot_dir, mat_save_name + '_train.mat')
        sample_save_path = os.path.join(plot_dir, mat_save_name + '_sample_trajectory.mat')
        if X_train is not None:
            scipy.io.savemat(train_save_path,  dict(data=X_train))
        scipy.io.savemat(sample_save_path, dict(data=X_sample))

### bilevel ###
def create_grid_for_eval(args, dim):
    if args.dataset_name in ['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong']:
        # width = 4
        width = 2
        n_pts = 100
        x_min, x_max = -2., 2.
        y_min, y_max = -2., 2.
        grid, grid_pad, grid_x, grid_y = create_grid_symmetric(width, n_pts, dim)
        dx = 4 / n_pts
    elif args.dataset_name in ['crowd_motion_gaussian_two_bars_uniform', 'crowd_motion_two_bars_bilevel', \
                                'crowd_motion_two_bars_uniform_bilevel', 'crowd_motion_gaussian_two_bars_gaussian',\
                                'crowd_motion_gaussian_two_bars_uniform_bilevel',\
                                'crowd_motion_gaussian_two_bars_gaussian_bilevel', 'crowd_motion_two_bars_gaussian']:
        x_min, x_max = 0., 2.
        y_min, y_max = 0., 2.
        n_pts = 100
        grid, grid_pad, grid_x, grid_y = create_grid(x_min, x_max, y_min, y_max, n_pts, dim)
        dx = 2 / n_pts
    else:
        raise NotImplementedError()
    
    return grid, grid_pad, grid_x, grid_y, dx

def create_grid(x_min, x_max, y_min, y_max, n_pts, dim):
    """Returns points on the regular grid on [-width, width]^2, with n_pts on each dimension
    dim is used to pad the grid into the desired dimension with 0's

    Returns:
        grid: n_pts^2 x 2
        grid_pad: 
        xx (grid_x): n_pts x n_pts
        yy (grid_y): n_pts x n_pts
    """
    x = np.arange(x_min, x_max, (x_max - x_min)/n_pts)
    y = np.arange(y_min, y_max, (y_max - y_min)/n_pts)

    xx, yy = np.meshgrid(x,y) # both have shape n x n
    grid = torch.Tensor(np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=-1)) # n^2 x 2
    # if dim > 2, pad the remaining coordinates as 0's
    n_sqr = grid.shape[0]
    d_pad = dim - 2
    grid_pad = torch.cat((grid, torch.zeros(n_sqr, d_pad)), dim=-1)

    return grid, grid_pad, xx, yy

def create_grid_symmetric(width, n_pts, dim):
    """Returns points on the regular grid on [-width, width]^2, with n_pts on each dimension
    dim is used to pad the grid into the desired dimension with 0's

    Returns:
        grid: n_pts^2 x 2
        grid_pad: n_pts^2 x d
        xx (grid_x): n_pts x n_pts
        yy (grid_y): n_pts x n_pts
    """
    x = np.arange(-width, width, 2*width/n_pts)
    y = np.arange(-width, width, 2*width/n_pts)

    xx, yy = np.meshgrid(x,y) # both have shape n x n
    grid = torch.Tensor(np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=-1)) # n^2 x 2
    # if dim > 2, pad the remaining coordinates as 0's
    n_sqr = grid.shape[0]
    d_pad = dim - 2
    grid_pad = torch.cat((grid, torch.zeros(n_sqr, d_pad)), dim=-1)

    return grid, grid_pad, xx, yy

def eval_obstacles_on_grid(B, B_true, grid, grid_pad, chunk_size=500):
    N = grid.shape[0]
    B_true_val = torch.cat(
                    [B_true.eval(grid[i*chunk_size:(i+1)*chunk_size]) for i in range((N//chunk_size) + 1)]
                    )
    B_val = torch.cat(
                    [B(grid_pad[i*chunk_size:(i+1)*chunk_size]) for i in range((N//chunk_size) + 1)]
                    )
    
    return B_val, B_true_val

def plot_obstacles(B_val, B_true_val, grid_x, grid_y, plot_dir, writer, tbx_logging, fig_name):
    B_val_np = B_val.detach().cpu().numpy().reshape(grid_x.shape)
    B_true_val_np = B_true_val.detach().cpu().numpy().reshape(grid_x.shape)
    fig = plt.figure(figsize=(10, 4))
    sub_fig1 = fig.add_subplot(121)
    vmin = np.min(np.minimum(B_val_np, B_true_val_np))
    vmax = np.max(np.maximum(B_val_np, B_true_val_np))
    pylab.pcolor(grid_x, grid_y, B_val_np, shading='auto', vmin=vmin, vmax=vmax)
    sub_fig2 = fig.add_subplot(122)
    pylab.pcolor(grid_x, grid_y, B_true_val_np, shading='auto', vmin=vmin, vmax=vmax)

    sub_fig1.set_title("Parametrized Obstacle")
    sub_fig1.set_xlabel('x')
    sub_fig1.set_ylabel('y')
    sub_fig2.set_title("True Obstacle")
    sub_fig2.set_xlabel('x')
    pylab.colorbar(ax=[sub_fig1, sub_fig2])
    # save
    # fig_name = 'obs_{}'.format(step+1)
    save_dir = os.path.join(plot_dir, fig_name)
    fig.savefig(save_dir)
    if tbx_logging:
        writer.add_figure(fig_name, fig)

# =================================================================================== #
#                                      Autograd                                       #
# =================================================================================== #

### mainly used in bilevel MFG to e.g. compute vhp ###

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod, device):
    # orig_params = tuple(mod.parameters())
    # orig_params = tuple([p.to(device) for p in orig_params])

    # orig_params = list(mod.parameters())
    # orig_params = list([p.to(device) for p in orig_params])

    orig_params = tuple(mod.parameters())
    orig_params = tuple([torch.Tensor(p.cpu()).to(device) for p in orig_params])

    # orig_params = list(mod.parameters())
    # orig_params = list([torch.Tensor(p.cpu()).to(device) for p in orig_params])

    # orig_params = tuple([torch.Tensor(p.detach().clone().cpu()).to(device) for p in mod.parameters()])
    # orig_params = tuple([p.requires_grad_() for p in orig_params])

    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
        
    return orig_params, names

def load_weights(model, names, params, as_params=False, device=torch.device('cuda')):
    """Loads params into model with the corresponding names

    Args:
        model: the NN model to load the parameters into
        names: names of parameters
        params: tuple of parameters
        as_params: whether to load the weights as parameters or just tensors, see below.
    """
    for name, p in zip(names, params):
        # use this when loading the weights into a model container for vhp computation
        if not as_params: 
            set_attr(model, name.split("."), p.to(device))
        # use this when loading the weights back into the model as a nn.Module
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p.to(device)))

def average_params(param_list):
    N = len(param_list)
    # each p is a tuple of the i-th parameter for all models
    new_params = [sum(p) / N for p in zip(*param_list)] 

    return new_params

def tuple_to_device(x, device):
    """Sends a tuple of tensors to the desired device

    Args:
        x (tuple): tuple of tensors, potentially of different shapes
        device: device name
    """
    # return list([a.to(device) for a in x])
    # return tuple([a.to(device) for a in x])
    for i in range(len(x)):
        x[i] = x[i].to(device)
    
    return x

def vhp_update(g, vhp, step_size=1):
    return tuple([g[i] - step_size*vhp[i] for i in range(len(g))])

def grad_step(model, beta):
    for p in model.parameters():
        # TODO: sometimes p.grad is None. I assume that's because 
        # the identity mapping is used instead of the parameters, so the gradient didn't accumulate.
        # If this understanding is correct, then it's okay to only update the parameters with gradients.
        if p.grad is not None:
            p.data = p.data - beta * p.grad.data

    return model

def set_grad(model, g):
    for i, p in enumerate(model.parameters()):
        # gradient is None means there's no dependence on this parameter, which can happen
        if g[i] is not None:
            # p.grad = g[i]
            p.grad = g[i].detach().clone()

    return model

def tuple_tensor_update(x, y, step_size=1):
    # if y has None's in it, do not update x
    return tuple([x[i] - step_size*y[i] if y[i] is not None else x[i] for i in range(len(x))])

def disable_grad(model):
    for p in model.parameters():
        p.requires_grad_(False)

    # for i in range(len(x)):
    #     if y[i] is not None:
    #         x[i] = x[i] - step_size*y[i]
    # return x

    # for i in range(len(x)):
    #     x[i].add_(-step_size*y[i])
    # return x

def get_update_from_grad(g, method='sgd'):
    if method == 'sgd':
        # for vanilla SGD, we directly use the gradients to do parameter updates 
        return g
    elif method == 'adam':
        return
    else:
        raise NotImplementedError()

def grad_clip_callback(grads):
    # clip_grad_norm_(params, 5.)

    max_norm = 5.
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
    clip_coef = max_norm / (total_norm + 1e-6)
    # clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    if clip_coef > 1.:
        clip_coef_clamped = 1.
    else:
        clip_coef_clamped = clip_coef
    for g in grads:
        if g is not None:
            g.data = g.data * clip_coef_clamped

    # max_norm = 5.
    # grads = [g for g in grads if g is not None]
    # device = grads[0].device
    # total_norm = torch.norm(torch.stack([torch.norm(g.detach()).to(device) for g in grads]))
    # clip_coef = max_norm / (total_norm + 1e-6)
    # clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    # for g in grads:
    #     g.detach().mul_(clip_coef_clamped.to(g.device))

    return grads

# =================================================================================== #
#                                       BLO                                           #
# =================================================================================== #

def check_bilevel_blowup(args, L_gap_hist, L_gap, l_gap_hist, l_gap):
    flag = np.abs(L_gap) / np.abs(np.array(L_gap_hist[-args.scheduler_gap_memory:])).mean() > \
    args.scheduler_gap_blowup_ratio or \
    np.abs(l_gap) / np.abs(np.array(l_gap_hist[-args.scheduler_gap_memory:])).mean() > \
    args.scheduler_gap_blowup_ratio

    return flag

    
# =================================================================================== #
#                                       Robotics                                      #
# =================================================================================== #

def traj_is_Feasible(space, traj):
    # traj: K x d
    point_feasibility = [space.isFeasible(p.detach().cpu().numpy()) for p in traj]

    # a trajectory is feasible iff all points on it are feasible
    return np.sum(point_feasibility) == len(point_feasibility)


def process_robot_data(x, dataset='robot_1', mode='preprocess'):
    # x: N x d or N x K x d
    if dataset == 'robot_1':
        d = 12
        I_dummy = [0, 7]
        I_keep = [i for i in range(d) if i not in I_dummy]
        if mode == 'preprocess':
            x_hat = x[..., I_keep]
        else:
            sz = torch.tensor(x.shape)
            # padd the last axis, which is the space dimension, by 2
            sz[-1] += 2
            sz = torch.Size(sz)
            x_hat = torch.zeros(sz).to(x.device)
            x_hat[..., I_keep] = x
    else:
        raise NotImplementedError()

    return x_hat

# =================================================================================== #
#                                          Misc                                       #
# =================================================================================== #

def make_scheduler(optimizer, args, option='inner'):
    if option == 'inner':
        if args.scheduler_inner == 'cyclic':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, args.lr_LL_min)
        elif args.scheduler_inner == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr_size, \
                                    gamma=args.step_lr_gamma)
        elif args.scheduler_inner == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.multi_lr_milestones, \
                                    gamma=args.multi_lr_gamma)
        elif args.scheduler_inner == 'adaptive':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', \
                                    factor=0.5, verbose=True, patience=args.patience_LL)
        else:
            scheduler = None
    elif option == 'obs':
        if args.scheduler_obs == 'adaptive':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=args.patience_obs)
        elif args.scheduler_obs == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr_size, gamma=args.step_lr_gamma)
        elif args.scheduler_obs == 'cyclic':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, args.lr_obs_min)
        elif args.scheduler_obs == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.multi_lr_milestones, gamma=args.multi_lr_gamma)
    elif option == 'NF':
        if args.scheduler_NF == 'adaptive':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=args.patience_NF)
        elif args.scheduler_NF == 'cyclic':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
        elif args.scheduler_NF == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.multi_lr_milestones, gamma=args.multi_lr_gamma)
    else:
        raise NotImplementedError()

    return scheduler


def tuple_tensor_norm_diff(X, Y):
    diff = [x - y for (x, y) in zip(X, Y)]

    return torch.sqrt(sum([torch.sum(a**2) for a in diff]))

def batch_jacobian(func, x, create_graph=False):
    # x: B x d
    def _func_sum(x):
        return func(x).sum(dim=0)
    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

def penalty_fun(x, option='id'):
    if option == 'max':
        x = torch.nn.functional.relu(x)
    elif option == 'l1':
        x = torch.abs(x)
    elif option == 'l2':
        x = x ** 2

    return x

def sanitize_args(args):
    if args.linear_transform_type == 'lu_no_perm':
        if 'coupling' in args.base_transform_type:
            args.OT_part = 'block_CL_no_perm'
        else:
            args.OT_part = 'block_AR_no_perm'
    
    # datasets
    if args.dataset_name == 'crowd_motion_gaussian' and args.lbd_F == 0:
        warnings.warn('Training on crowd motion, but the weight for F is 0.')
    
    interaction_datasets = ['crowd_motion_gaussian', 'drones_22', 'drones_23', 'drones_82', 'drones_22_obs', \
                            'crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong',
                            'crowd_motion_gaussian_nonsmooth_obs', 'crowd_motion_gaussian_close', 
                            'crowd_motion_gaussian_NN_obs', 'crowd_motion_two_bars', 'crowd_motion_gaussian_two_bars',
                            'crowd_motion_two_bars_bilevel', "crowd_motion_two_bars_uniform", "crowd_motion_two_bars_uniform_bilevel",
                            'crowd_motion_gaussian_two_bars_uniform', 'crowd_motion_gaussian_two_bars_uniform_bilevel',
                            'crowd_motion_gaussian_one_bar_uniform', 'crowd_motion_gaussian_two_bars_gaussian', 
                            'crowd_motion_flower', 'crowd_motion_gaussian_two_bars_gaussian_bilevel',
                            'crowd_motion_two_bars_gaussian',
                            'robot_1']
    if args.dataset_name in interaction_datasets:
        args.interaction = True
    
    # determine whether 
    if args.dataset_name in ['crowd_motion_gaussian', 'crowd_motion_gaussian_close', 
                             'crowd_motion_gaussian_two_bars', 'crowd_motion_gaussian_two_bars_uniform',
                            'crowd_motion_gaussian_one_bar_uniform', 'crowd_motion_gaussian_two_bars_gaussian',
                            'crowd_motion_flower']:
        args.Q_is_dist = True
        args.Q_true_is_dist = True # no difference between Q (parametrized) and Q_true here
    elif args.dataset_name in ['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong',
                                'crowd_motion_gaussian_two_bars_uniform_bilevel', 
                                'crowd_motion_gaussian_two_bars_gaussian_bilevel']:
        args.Q_is_dist = False
        args.Q_true_is_dist = True
    else:
        args.Q_is_dist = False
        args.Q_true_is_dist = False
    
    if args.dataset_name == 'robot_1':
        args.obs_dir = './results/robot_1/obs_NN_{}.t'.format(args.robot_1_obs)
        # if args.robot_1_obs == 'default':
        #     args.obs_dir = './results/robot_1/obs_NN_thick.t'
        # elif args.robot_1_obs == 'long':
        #     args.obs_dir = './results/robot_1/obs_NN_thick_long.t'
        # else:
        #     raise NotImplementedError()

    # the number of dimensions we project onto before evaluating on the obstacle
    if args.dataset_name == 'robot_1':
        args.Q_dim = 10
    elif args.Q_is_dist:
        args.Q_dim = 2
    else:
        args.Q_dim = args.gaussian_multi_dim
        
    if args.reg_OT_dir == 'gen' or args.interaction or args.NF_loss == 'KL_density'\
        or args.NF_loss == 'jeffery':
        args.sample_in_train = True
    else:
        args.sample_in_train = False

    if 'simp' in args.disc_scheme:
        args.F_disc_scheme = 'simp'
    else:
        args.F_disc_scheme = 'right_pt'

    if args.NF_model == 'single_flow':
        args.K = args.num_flow_steps
        args.num_flow_steps = 1 # use 1 flow repeatedly

    if hasattr(args, 'multi_lr_milestones'):
        args.multi_lr_milestones = [int(a) for a in args.multi_lr_milestones]

    return args

def print_args(args, save_dir='args.txt'):
    """Intended usage: print_args(sys.argv)
    """
    # If this is not the first time writing to the same args.txt file
    # don't overwrite it, just keep writing below.
    if os.path.exists(save_dir):
        with open(save_dir, 'a') as f:
            f.writelines('\n')
            f.writelines('python ')
            for s in args:
                f.writelines(s + ' ')
    else:
        with open(save_dir, 'w') as f:
            f.writelines('python ')
            for s in args:
                f.writelines(s + ' ')

class Plot_logger(object):
    def __init__(self, root_path=None):
        self.root_path = root_path

    def log(self, data_name, data):
        fpath = os.path.join(self.root_path, data_name + '.txt')
        self.file = open(fpath, 'a')
        self.file.write(str(data) + '\n')

    def log_multiple(self, data_name, data_list):
        for i in range(len(data_name)):
            name = data_name[i]
            data = data_list[i]
            fpath = os.path.join(self.root_path, name + '.txt')
            self.file = open(fpath, 'a')
            self.file.write(str(data) + '\n')
