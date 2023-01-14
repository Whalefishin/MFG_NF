import argparse
from mfp_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--exp_name', type=str, default='1')
    parser.add_argument('--dataset_name', type=str, default='crowd_motion_gaussian_bilevel',
                        choices=['crowd_motion_gaussian_bilevel', 'crowd_motion_gaussian_bilevel_strong',
                                 'crowd_motion_two_bars_bilevel', 'crowd_motion_two_bars_uniform_bilevel',
                                 'crowd_motion_gaussian_two_bars_uniform_bilevel', 
                                 'crowd_motion_gaussian_two_bars_gaussian_bilevel'],
                        help='Name of dataset to use.')
    # parser.add_argument('--train_batch_size', type=int, default=64,
    #                     help='Size of batch used for training.')
    parser.add_argument('--val_frac', type=float, default=1.,
                        help='Fraction of validation set to use.')
    # parser.add_argument('--val_batch_size', type=int, default=512,
    #                     help='Size of batch used for validation.')

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
                        choices=['relu', 'tanh', 'mish'])

    # logging and checkpoints
    parser.add_argument('--monitor_interval', type=int, default=1000,
                        help='Interval in steps at which to report training stats.')
    parser.add_argument('--plot_interval', type=int, default=100,
                        help='Interval in steps at which to report training stats.')
    parser.add_argument('--gradcheck_interval', type=int, default=50,
                        help='Interval in steps at which to report training stats.')
    parser.add_argument('--pretrain_monitor_interval', type=int, default=50,
                        help='Interval in steps at which to report training stats.')
    parser.add_argument('--save_interval', type=int, default=3000,
                        help='Interval in steps at which to report training stats.')

    # reproducibility
    parser.add_argument('--seed', type=int, default=1638128,
                        help='Random seed for PyTorch and NumPy.')

    # MFG
    parser.add_argument('--gaussian_multi_dim',     type=int, default=2)
    parser.add_argument('--gaussian_multi_a',       type=float, default=10.)
    parser.add_argument('--num_train_data',         type=int, default=50000)
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
    parser.add_argument('--interp_hist',        type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--n_interp',           type=int, default=5, 
                        help='Number of interpolated points inserted between flow points to better approximate MFG costs')
    parser.add_argument('--two_bars_sharpness', type=float, default=8.)
    parser.add_argument('--two_bars_height',    type=float, default=50.)

    # optimization
    parser.add_argument('--num_training_steps', type=int, default=500, help='Number of total training steps in the outer loop.')
    parser.add_argument('--num_pretrain_NF_steps', type=int, default=20000)
    parser.add_argument('--l2_reg_NF', type=float, default=0)
    parser.add_argument('--l2_reg_inner', type=float, default=0)
    parser.add_argument('--l2_reg_obs', type=float, default=0)
    parser.add_argument('--optimizer_NF_pretrain', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--optimizer_NF', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--optimizer_obs', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--optimizer_inner', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--sgd_inner_nesterov', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--sgd_obs_nesterov', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--sgd_inner_momentum', type=float, default=0)
    parser.add_argument('--sgd_obs_momentum', type=float, default=0, help='Typically 0.9 or 0.8 if set.')
    parser.add_argument('--scheduler_obs', type=str, default='adaptive', choices=['step', 'cyclic', 'adaptive', 'multi_step', 'gap'])
    parser.add_argument('--scheduler_NF', type=str, default='adaptive', choices=['cyclic', 'adaptive'])
    parser.add_argument('--scheduler_NF_pretrain', type=str, default='adaptive', choices=['cyclic', 'adaptive'])
    parser.add_argument('--scheduler_inner', type=str, default='none', choices=['none', 'cyclic', 'multi_step', 'step', 'gap', 'adaptive'])
    parser.add_argument('--step_lr_size', type=int, default=50)
    parser.add_argument('--step_lr_gamma', type=float, default=0.8)
    parser.add_argument('--adam_beta1_obs', type=float, default=0.9)
    parser.add_argument('--adam_beta2_obs', type=float, default=0.999)
    parser.add_argument('--lr_LL_min', type=float, default=1e-7)
    parser.add_argument('--lr_obs_min', type=float, default=1e-2)
    parser.add_argument('--multi_lr_gamma', type=float, default=0.7)
    parser.add_argument('--multi_lr_milestones', nargs='+', default=[2000,3000,4000])
    parser.add_argument('--scheduler_gap_gamma', type=float, default=0.7)
    parser.add_argument('--scheduler_gap_blowup_ratio', type=float, default=20)
    parser.add_argument('--scheduler_gap_warmup', type=int, default=100)
    parser.add_argument('--scheduler_gap_memory', type=int, default=100)
    
    # Bilevel MFG
    ## obstacle parametrization
    parser.add_argument('--h_obs',                     type=int, default=128, help='hidden dimension in the NN parametrizing the obstacle.')
    parser.add_argument('--l_obs',                     type=int, default=1, help='number of hidden layers in the NN parametrizing the obstacle.')
    parser.add_argument('--act_obs',                   type=str, default='mish', choices=
                                                        ['relu', 'softplus', 'tanh', 'elu', 'leaky_relu', 'relu_sqr', 'mish', 'swish'])
    parser.add_argument('--act_obs_out',               type=str, default='none', choices=
                                                        ['none', 'exp', 'relu', 'sqr', 'sigmoid'])
    parser.add_argument('--softplus_beta',             type=float, default=1.)
    parser.add_argument('--bn_obs',                    type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--res_link_obs',              type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--last_bias_obs',             type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--pou_obs',                   type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--n_pou_obs',                 type=int, default=2, help='number of components in the partition of unity')
    parser.add_argument('--h_pou_obs',                 type=int, default=128)
    parser.add_argument('--l_pou_obs',                 type=int, default=1)
    parser.add_argument('--act_pou_obs',               type=str, default='mish', choices=
                                                        ['relu', 'softplus', 'tanh', 'elu', 'leaky_relu', 'relu_sqr', 'mish', 'swish'])
    parser.add_argument('--res_link_pou_obs',          type=lambda x: (str(x).lower() == 'true'), default=True)
    ## Training
    parser.add_argument('--lr_NF',                     type=float, default=3e-4, help='Learning rate for NF.')
    parser.add_argument('--lr_obs',                    type=float, default=1e-2, help='Learning rate for the obstacle.')
    parser.add_argument('--num_training_steps_inner',  type=int, default=3, help='Number of total training steps in the inner loop (lower problem)')
    parser.add_argument('--step_size_inner',           type=float, default=1e-5, help='beta for the inner loop')
    parser.add_argument('--step_decay_inner',          type=str, default='none', choices=
                                                        ['none', 'sqrt'], help='Decay mode for the lower problem step size, range: [0, inf], 0 = no decay')
    parser.add_argument('--grad_clip_NF',              type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--grad_clip_obs',             type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--grad_clip_LL',              type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--grad_clip_value_NF',        type=float, default=5., help='Value by which to clip norm of gradients.')
    parser.add_argument('--grad_clip_value_obs',       type=float, default=5., help='Value by which to clip norm of gradients.')
    parser.add_argument('--grad_clip_value_LL',        type=float, default=5., help='Value by which to clip norm of gradients.')
    parser.add_argument('--patience_obs',              type=int, default=1000)
    parser.add_argument('--patience_NF',               type=int, default=1000)
    parser.add_argument('--patience_LL',               type=int, default=20)
    parser.add_argument('--verbose_logging',           type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--NF_update_interval',        type=int, default=1, help='Update NF params every x outer iterates')
    parser.add_argument('--obs_warm_up',               type=int, default=0, help='Do not update the obstacle for the first x iterates.')
    parser.add_argument('--NF_keep_params',            type=lambda x: (str(x).lower() == 'true'), default=True,\
                                                            help='whether to use the last NF parameters from the LL as the initialization on the next step.')
    # parser.add_argument('--obs_reg_mass',              type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--obs_reg_mass_loss',         type=str, default='none', choices=
                                                        ['none', 'l2', 'l1'])
    parser.add_argument('--lbd_mass',                  type=float, default=1e-5)
    parser.add_argument('--lbd_penalty',               type=float, default=1e3)
    ## BLO algorithm
    parser.add_argument('--algo',                      type=str, default='GM', choices=
                                                        ['GM', 'IAPTT', 'BDA', 'GM_true_obs', \
                                                        'penalty', 'penalty_fixNF'])
    ### BDA
    parser.add_argument('--BDA_alpha',                 type=float, default=0.5)
    parser.add_argument('--BDA_adaptive_alpha',        type=lambda x: (str(x).lower() == 'true'), default=True)
    ### penalty
    parser.add_argument('--penalty_LL_init',           type=str, default='last_iter', choices=
                                                        ['last_min', 'last_iter', 'none'])
    parser.add_argument('--penalty_fun',               type=str, default='default', choices=
                                                        ['default', 'l1', 'l2', 'max'])
    parser.add_argument('--penalty_H_approx_iter',     type=int, default=1)
    # misc.
    parser.add_argument('--num_batch_to_plot',     type=int, default=1)
    parser.add_argument('--pretrain_NF',           type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--pretrain_NF_dir',       type=str, default='')
    parser.add_argument('--pretrain_obs_dir',      type=str, default='')
    parser.add_argument('--bilevel_training_data_dir', type=str, default='')
    parser.add_argument('--pretrain_NF_grad_clip', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_pretrain_NF',      type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_pretrain_obs',     type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--compute_lip_bound',     type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--pretrain_obs_eps',      type=float, default=0., 
                                                    help='The scale of uniform noises on the pretrain obstacle parameters')
    parser.add_argument('--syn_noise',             type=float, default=0.1)
    parser.add_argument('--marker_size',           type=float, default=5)
    parser.add_argument('--color',                 type=str, default='order', choices=['order', 'radius'])
    parser.add_argument('--tabular_subset',        type=lambda x: (str(x).lower() == 'true'), default=False)

    # debugging
    parser.add_argument('--debug_obs_init', type=str, default='true', choices=['true', 'rand', 'close'])
    parser.add_argument('--debug_obs',      type=str, default='NN', choices=
                                            ['NN', 'gaussian'])
    parser.add_argument('--debug_step_NF',  type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--debug_tensor_type',  type=str, default='float', choices=['float', 'double'])
    parser.add_argument('--gradcheck_use_higher_opt', type=lambda x: (str(x).lower() == 'true'), default=True)

    args = parser.parse_args()
    args = sanitize_args(args)

    return args
