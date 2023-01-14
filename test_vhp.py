import numpy as np
import os
from mfp_utils import *
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.manifold import *
from tensorboardX import SummaryWriter
import math
import copy

fix_std              = False
plot_trajectory      = False
test_logger          = False
test_vhp             = False
test_vjp_vhp         = False
test_BLMFG           = False
test_BLMFG_corrected = False
test_BLMFG_AD        = False
test_gradcheck       = True

torch.manual_seed(69)
np.random.seed(420)

if fix_std:
    dataset  = 'power'
    # folder   = 'results_laigpu/results/' + dataset
    folder   = 'results_laigpu/' + dataset
    exp_name = 'paper_AR_OT=5e-5_norm'

    LL_file_name      = dataset + '-rq-autoregressive-log_likelihood.npy'
    OT_norm_file_name = dataset + '-rq-autoregressive-OT_cost_norm.npy'
    LL_file_dir       = os.path.join(folder, exp_name, LL_file_name)
    OT_norm_file_dir  = os.path.join(folder, exp_name, OT_norm_file_name)

    LL      = np.load(LL_file_dir)
    OT_norm = np.load(OT_norm_file_dir)


    print ('LL: {:.4f} +- {:.4f}'.format(LL.mean(), 2 * LL.std() / np.sqrt(LL.shape[0])))
    print ('OT norm:{:.4f} +- {:.4f}'.format(OT_norm.mean(), 2 * OT_norm.std() /  np.sqrt(OT_norm.shape[0])))

if plot_trajectory:
    dataset  = 'miniboone'
    folder   = 'results_laigpu/results/' + dataset
    exp_name = 'paper_AR_OT=0_reRun'

    traj_file_name  = 'last_epoch_sample_trajectory.mat'
    train_file_name = 'last_epoch_train.mat'
    traj_file_dir   = os.path.join(folder, exp_name, traj_file_name)
    train_file_dir  = os.path.join(folder, exp_name, train_file_name)

    X_sample = loadmat(traj_file_dir)['data']
    X_train  = loadmat(train_file_dir)['data']

    print ('X_sample size: {}'.format(X_sample.shape[0]))
    print ('X_train size: {}'.format(X_train.shape[0]))

    x_min = -10
    x_max = 10
    y_min = -10
    y_max = 10

    method = 'Isomap'
    # method = 'LocallyLinearEmbedding'

    plot_evolution(X_sample, X_train, os.path.join(folder, exp_name), x_min, x_max,\
                    y_min, y_max, subset=5000, marker_size=5, mat_save_name='last_epoch', \
                        dim_reduction=True, reduction_method=method, save_data=False)

if test_logger:
    timestamp = 'test_logger'
    log_dir = os.path.join('txb/gaussian_mixture', timestamp)
    writer = SummaryWriter(log_dir=log_dir, max_queue=20)

    writer.add_scalar(tag='loss', scalar_value=1.2, global_step=1)
    writer.flush()



if test_vhp:
    import torchvision.models as models
    from torch.autograd.functional import vhp

    # Utilities to make nn.Module functional
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

    def make_functional(mod):
        orig_params = tuple(mod.parameters())
        # Remove all the parameters in the model
        names = []
        for name, p in list(mod.named_parameters()):
            del_attr(mod, name.split("."))
            names.append(name)
        return orig_params, names

    # def load_weights(mod, names, params):
    #     for name, p in zip(names, params):
    #         set_attr(mod, name.split("."), p)

    def load_weights(model, names, params, as_params=False):
        for name, p in zip(names, params):
            # use this when loading the weights into a model container for vhp computation
            if not as_params: 
                set_attr(model, name.split("."), p)
            # use this when loading the weights back into the model as a nn.Module
            else:
                set_attr(model, name.split("."), torch.nn.Parameter(p))

    N = 10
    model = models.resnet18(pretrained=False).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    inputs = torch.rand([N, 3, 224, 224], device=device)
    labels = torch.rand(N, device=device).mul(10).long()

    # used for VJP
    v_1 = model(inputs) - inputs

    # make NN a pure function
    params, names = make_functional(model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    def nn_fun(*new_params):
        load_weights(model, names, new_params, as_params=False)
        out = model(inputs)

        return out

    def loss_fun(*new_params):
        load_weights(model, names, new_params, as_params=False)
        out = model(inputs)

        loss = criterion(out, labels)
        return loss

    # VJP, output should have the same shape as the parameters.
    out, a = vhp(nn_fun, params, v_1)

    # the vector you want to product H with
    v_2 = tuple(p.detach().clone().requires_grad_() for p in params)
    # out is L(input;theta), i.e. the loss value. a is the vhp product. It should have the same shape as the parameters.
    out, a = vhp(loss_fun, params, v_2)

    print ("done")


if test_vjp_vhp:
    import torchvision.models as models
    from torch.autograd.functional import vhp, vjp
    import torch.nn as nn

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

    def make_functional(mod):
        orig_params = tuple(mod.parameters())
        # Remove all the parameters in the model
        names = []
        for name, p in list(mod.named_parameters()):
            del_attr(mod, name.split("."))
            names.append(name)
        return orig_params, names

    def load_weights(model, names, params, as_params=False):
        for name, p in zip(names, params):
            # use this when loading the weights into a model container for vhp computation
            if not as_params: 
                set_attr(model, name.split("."), p)
            # use this when loading the weights back into the model as a nn.Module
            else:
                set_attr(model, name.split("."), torch.nn.Parameter(p))

    class MLP(nn.Module):
        def __init__(self, d_in):
            super(MLP, self).__init__()
            d_h = 16
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_in)
            )
        
        def forward(self, x):
            return self.net(x)

    N    = 10
    d_in = 2
    model = MLP(d_in).to(device)
    criterion = torch.nn.MSELoss()

    inputs = torch.rand([N, d_in], device=device)
    target = torch.rand([N, d_in], device=device)

    # used for VJP
    v_1 = model(inputs) - inputs

    # make NN a pure function
    params, names = make_functional(model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    def nn_fun(*new_params):
        load_weights(model, names, new_params, as_params=False)
        out = model(inputs)

        return out

    def loss_fun(*new_params):
        load_weights(model, names, new_params, as_params=False)
        out = model(inputs)

        loss = criterion(out, target)
        return loss

    # VJP, output should have the same shape as the parameters (a tuple of tensors).
    out_1, v_2 = vjp(nn_fun, params, v_1) # format: compute NN's gradient wrt params and multiple with v_1

    # out is L(input;theta), i.e. the loss value. a is the vhp product. It should have the same shape as the parameters.
    out_2, a = vhp(loss_fun, params, v_2) # format: compute loss function's Hessian wrt params and multiple with v_2


    print ("Done")



if test_BLMFG:
    import torchvision.models as models
    from torch.autograd.functional import vhp, vjp
    import torch.nn as nn

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

    def make_functional(mod):
        orig_params = tuple(mod.parameters())
        # Remove all the parameters in the model
        names = []
        for name, p in list(mod.named_parameters()):
            del_attr(mod, name.split("."))
            names.append(name)
        return orig_params, names

    def load_weights(model, names, params, as_params=False):
        for name, p in zip(names, params):
            # use this when loading the weights into a model container for vhp computation
            if not as_params: 
                set_attr(model, name.split("."), p)
            # use this when loading the weights back into the model as a nn.Module
            else:
                set_attr(model, name.split("."), torch.nn.Parameter(p))

    
    def vhp_update(g, vhp, beta):
        return tuple([g[i] - beta*vhp[i] for i in range(len(g))])


    def grad_step(model, beta):
        for p in model.parameters():
            p.data = p.data + beta * p.grad.data

        return model

    def set_grad(model, g):
        for i, p in enumerate(model.parameters()):
            p.grad = g[i]

        return model


    class Flow(nn.Module):
        def __init__(self, d_in):
            super(Flow, self).__init__()
            d_h = 16
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_in)
            )
        
        def forward(self, x):
            return self.net(x)

    class Obstacle(nn.Module):
        def __init__(self, d_in):
            super(Obstacle, self).__init__()
            d_h = 8
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h),
                nn.ReLU(),
                nn.Linear(d_h, 1)
            )
        
        def forward(self, x):
            return self.net(x)

    class Flow_Obstacle(nn.Module):
        def __init__(self, flow, obstacle):
            super(Flow_Obstacle, self).__init__()
            self.flow = flow
            self.obstacle = obstacle
            self.param_len_flow = len(tuple(flow.parameters()))

        def forward(self, x):
            out = self.flow(x)

            return self.obstacle(out)

    T    = 2
    N    = 10
    N_I  = 3
    d_in = 2
    flow = Flow(d_in).to(device)
    obstacle = Obstacle(d_in).to(device)
    flow_obstacle= Flow_Obstacle(flow, obstacle)
    optimizer_flow_obstacle = torch.optim.Adam(flow_obstacle.parameters())
    
    for t in range(T):
        flow_obstacle_fun = copy.deepcopy(flow_obstacle)
        optimizer_flow_obstacle_fun = torch.optim.Adam(flow_obstacle_fun.parameters())
        betas = [0.1] * N_I
        x_list = []
        l_list = []
        params_list = []
        i_max = 0

        # upper level obj value
        x = torch.rand([N, d_in], device=device)
        l_val = (flow(x) - x).mean()
        l_val_max = l_val

        # l_list.append(l_val)
        # x_list.append(x)
        params = tuple(flow_obstacle_fun.parameters())
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        params_list.append(params)

        for i in range(N_I):
            beta = betas[i]
            # sample a batch of training data
            x = torch.rand([N, d_in], device=device)
            # Let's pretend this is the lower level (MFG) objective
            # the important thing is that it should involve both the flow and obstacle models
            loss = flow_obstacle_fun(x).mean()
            # compute gradient on lower level obj - will be recorded on param.grad.data
            loss.backward()
            # update each param with its grad.data by taking a simple gradient step
            flow_obstacle_fun = grad_step(flow_obstacle_fun, beta)

            optimizer_flow_obstacle_fun.zero_grad()
            
            # record the param that incurs the largest upper level obj.
            l_val = (flow(x) - x).mean()
            if l_val > l_val_max:
                i_max = i+1

            # l_list.append(l_val)
            x_list.append(x)
            params = tuple(flow_obstacle_fun.parameters())
            params = tuple(p.detach().requires_grad_() for p in params)
            params_list.append(params)


        inputs = torch.rand([N, d_in], device=device)
        g = flow_obstacle_fun.flow(inputs) - inputs

        # make NN a pure function
        # theta, names_theta = make_functional(flow)
        # phi, names_phi = make_functional(obstacle)
        params, names = make_functional(flow_obstacle_fun)

        def nn_fun(*new_params):
            # load_weights(flow_obstacle.flow, names[:flow_obstacle.param_len_flow], new_params, as_params=False)
            load_weights(flow_obstacle_fun, names[:flow_obstacle_fun.param_len_flow], new_params, as_params=False)
            out = flow_obstacle_fun.flow(x)
            return out

        # def loss_fun_theta(*new_params):
        #     load_weights(flow, names_theta, new_params, as_params=False)
        #     load_weights(obstacle, names_phi, phi, as_params=False)
        #     # MFG loss
        #     out = flow(x)
        #     loss = obstacle(out)
        #     return loss

        # def loss_fun_phi(*new_params):
        #     load_weights(flow, names_theta, theta, as_params=False)
        #     load_weights(obstacle, names_phi, new_params, as_params=False)
        #     # MFG loss
        #     out = flow(x)
        #     loss = obstacle(out)
        #     return loss

        def loss_fun(*new_params):
            load_weights(flow_obstacle_fun, names, new_params, as_params=False)
            # MFG loss
            loss = flow_obstacle_fun(x).mean()
            return loss
        
        
        # compute gradients of l wrt theta, which parametrizes the NF/agents; 
        # and gradients of l wrt phi, which parametrizes the obstacle
        _, g_theta = vjp(nn_fun, params[:flow_obstacle_fun.param_len_flow], g) # g_theta has the same shape as the flow params
        g_phi = tuple([torch.zeros_like(p) for p in params[flow_obstacle_fun.param_len_flow:]])
        g = g_theta + g_phi

        for i in range(i_max, 0, -1):
            # change the input to be consistent with what was used during the lower level updates
            x = x_list[i-1]
            # # compute all the necessary vhp's, then we'll take blocks of it for different uses
            # # vhp_all should be a tuple of tensors, with the first few corresponding to theta, and the subsequent ones phi.
            # vhp_all = vhp(loss_fun, params_list[i-1], g_theta + tuple([torch.zeros_like(p) for p in obstacle.parameters()]) )
            # # this is not precisely a vhp - it is g_phi * D_phi D_theta L(x; theta, phi)
            # vhp_phi = vhp_all[:flow_obstacle_fun.param_len_flow]
            # # g_theta * D_theta^2 L(x; theta, phi)
            # vhp_theta = vhp_all[flow_obstacle_fun.param_len_flow:]
            # g_phi   = vhp_update(g_phi, vhp_phi, beta[i-1])
            # g_theta = vhp_update(g_theta, vhp_theta, beta[i-1])

            # compute all the necessary vhp's, then we'll take blocks of it for different uses
            # vhp_all should be a tuple of tensors, with the first few corresponding to theta, and the subsequent ones phi.
            _, vhp_all = vhp(loss_fun, params_list[i-1], g[:flow_obstacle_fun.param_len_flow] + g_phi )
            g = vhp_update(g, vhp_all, betas[i-1])

        # load the weights back into the network as parameters, so that it's a proper nn.Module again
        # this is so that we can do optimizer.step() 
        # load_weights(flow_obstacle, names, params, as_params=True)

        # flow_obstacle = set_grad(flow_obstacle, g_theta + g_phi)
        flow_obstacle = set_grad(flow_obstacle, g)
        # since we messed with unloading/reloading parameters, the reference between the optimizer and the 
        # NN parameters are broken, so we reset the linkage.
        # optimizer_flow_obstacle.param_groups[0]['params'] = list(flow_obstacle.parameters())
        optimizer_flow_obstacle.step()

    print ("Done")


if test_BLMFG_corrected:
    import torchvision.models as models
    from torch.autograd.functional import vhp, vjp, jacobian
    import torch.nn as nn

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

    def make_functional(mod):
        orig_params = tuple(mod.parameters())
        # Remove all the parameters in the model
        names = []
        for name, p in list(mod.named_parameters()):
            del_attr(mod, name.split("."))
            names.append(name)
        return orig_params, names

    def load_weights(model, names, params, as_params=False):
        for name, p in zip(names, params):
            # use this when loading the weights into a model container for vhp computation
            if not as_params: 
                set_attr(model, name.split("."), p)
            # use this when loading the weights back into the model as a nn.Module
            else:
                set_attr(model, name.split("."), torch.nn.Parameter(p))

    
    def vhp_update(g, vhp, step_size=1):
        return tuple([g[i] - step_size*vhp[i] for i in range(len(g))])


    def grad_step(model, beta):
        for p in model.parameters():
            p.data = p.data + beta * p.grad.data

        return model

    def set_grad(model, g):
        for i, p in enumerate(model.parameters()):
            p.grad = g[i]

        return model
        
    def batch_jacobian(func, x, create_graph=False):
        # x: B x d
        def _func_sum(*x):
            return func(x).sum(dim=0)
        return jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)


    class Flow(nn.Module):
        def __init__(self, d_in):
            super(Flow, self).__init__()
            d_h = 16
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_in)
            )
        
        def forward(self, x):
            return self.net(x)

    class Obstacle(nn.Module):
        def __init__(self, d_in):
            super(Obstacle, self).__init__()
            d_h = 8
            self.net = nn.Sequential(
                nn.Linear(d_in, d_h),
                nn.ReLU(),
                nn.Linear(d_h, 1)
            )
        
        def forward(self, x):
            return self.net(x)

    class Flow_Obstacle(nn.Module):
        def __init__(self, flow, obstacle):
            super(Flow_Obstacle, self).__init__()
            self.flow = flow
            self.obstacle = obstacle
            self.param_len_flow = len(tuple(flow.parameters()))

        def forward(self, x):
            out = self.flow(x)

            return self.obstacle(out)

    T    = 2
    N    = 2
    N_I  = 3
    d_in = 2
    flow = Flow(d_in).to(device)
    obstacle = Obstacle(d_in).to(device)
    flow_obstacle= Flow_Obstacle(flow, obstacle)
    optimizer_flow_obstacle = torch.optim.Adam(flow_obstacle.parameters())
    
    for t in range(T):
        flow_obstacle_fun = copy.deepcopy(flow_obstacle)
        optimizer_flow_obstacle_fun = torch.optim.Adam(flow_obstacle_fun.parameters())
        betas = [0.1] * N_I
        x_list = []
        l_list = []
        params_list = []
        i_max = 0

        # upper level obj value
        x_upper = torch.rand([N, d_in], device=device)
        # x_upper = torch.rand([d_in], device=device)
        l_val = (flow(x_upper) - x_upper).mean()
        l_val_max = l_val

        # l_list.append(l_val)
        # x_list.append(x)
        params = tuple(flow_obstacle_fun.parameters())
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        params_list.append(params)

        for i in range(N_I):
            beta = betas[i]
            # sample a batch of training data
            x = torch.rand([N, d_in], device=device)
            # Let's pretend this is the lower level (MFG) objective
            # the important thing is that it should involve both the flow and obstacle models
            loss = flow_obstacle_fun(x).mean()
            # compute gradient on lower level obj - will be recorded on param.grad.data
            loss.backward()
            # update each param with its grad.data by taking a simple gradient step
            flow_obstacle_fun = grad_step(flow_obstacle_fun, beta)

            optimizer_flow_obstacle_fun.zero_grad()
            
            # record the param that incurs the largest upper level obj.
            l_val = (flow(x) - x).mean()
            if l_val > l_val_max:
                i_max = i+1

            # l_list.append(l_val)
            x_list.append(x)
            params = tuple(flow_obstacle_fun.parameters())
            params = tuple(p.detach().requires_grad_() for p in params)
            params_list.append(params)


        g = flow_obstacle_fun.flow(x_upper) - x_upper

        # make NN a pure function
        params, names = make_functional(flow_obstacle_fun)

        def nn_fun(*new_params):
            # load_weights(flow_obstacle.flow, names[:flow_obstacle.param_len_flow], new_params, as_params=False)
            load_weights(flow_obstacle_fun, names[:flow_obstacle_fun.param_len_flow], new_params, as_params=False)
            out = flow_obstacle_fun.flow(x_upper)
            return out

        def loss_fun(*new_params):
            load_weights(flow_obstacle_fun, names, new_params, as_params=False)
            # MFG loss
            loss = flow_obstacle_fun(x).mean()
            return loss
        
        # compute gradients of l wrt theta, which parametrizes the NF/agents; 
        # and gradients of l wrt phi, which parametrizes the obstacle
        _, g_theta = vjp(nn_fun, params[:flow_obstacle_fun.param_len_flow], g) # g_theta has the same shape as the flow params
        # Test: manunally computing jacobian, then take the product and see if the results agree with vjp
        D = jacobian(nn_fun, params[:flow_obstacle_fun.param_len_flow]) # B x |theta|
        g0 = torch.bmm(g.reshape(N,1,-1), D[0].reshape(N, d_in, -1)).reshape(N,g_theta[0].shape[0], g_theta[0].shape[1])
        g0 = torch.sum(g0, dim=0)
        assert torch.sum(torch.abs(g0 - g_theta[0])) < 1e-2
        
        g_zeros_like_phi = tuple([torch.zeros_like(p) for p in params[flow_obstacle_fun.param_len_flow:]])
        g_phi = copy.deepcopy(g_zeros_like_phi)
        g = g_theta + g_phi
        vhp_list = [None] * i_max

        # this for-loop constructs and stores all the ingredients we need for computing the desired gradients
        # in particular, vhp_list[i] is [g_theta * D_theta^2 L(theta_i), g_theta * D_phi D_theta_i L(theta_i)]
        for i in range(i_max, 0, -1):
            # change the input to be consistent with what was used during the lower level updates
            x = x_list[i-1]
            _, vhp_all = vhp(loss_fun, params_list[i-1], g[:flow_obstacle_fun.param_len_flow] + g_zeros_like_phi)
            vhp_list[i-1] = vhp_all
            g = vhp_update(g, vhp_all, step_size=betas[i-1])
        
        g = g_theta + g_zeros_like_phi
        a = tuple([torch.zeros_like(p) for p in params[flow_obstacle_fun.param_len_flow:]])
        # this for-loop computes the desired gradients from the ingredients we got from the previous loop
        # after this loop, the theta part of g should be the same as the theta part of g after the previous loop
        for i in range(i_max):
            # These are what we want to do
            # a = a + vhp_list[i]
            # g_phi = g_phi - beta[i] * a
            # g_theta = g_theta - beta[i_max-i-1] * vhp_list[i_max-i-1][:flow_obstacle_fun.param_len_flow], which is the
            # same as g_theta = g_theta - beta[i-1] * vhp_list[i-1][:flow_obstacle_fun.param_len_flow],
            # so we combine the two grad updates as: g = g - beta[i] * [vhp_list[i-1][:flow_obstacle_fun.param_len_flow], a]
            a = vhp_update(a, vhp_list[i][flow_obstacle_fun.param_len_flow:])
            g = vhp_update(g, vhp_list[i-1][:flow_obstacle_fun.param_len_flow] + a, step_size=betas[i-1])

        flow_obstacle = set_grad(flow_obstacle, g)
        optimizer_flow_obstacle.step()

    print ("Done")


if test_BLMFG_AD:
    import torchvision.models as models
    from torch.autograd.functional import vhp, vjp, jacobian
    from torch.autograd import grad
    import torch.nn as nn

    beta = 1e-4
    x     = torch.rand(1,5,10)
    x_0   = torch.rand(1,10)
    theta_0 = torch.rand(1,1).requires_grad_()
    phi   = torch.rand(1,10).requires_grad_()

    def tuple_tensor_update(x, y, step_size=1):
        return tuple([x[i] - step_size*y[i] for i in range(len(x))])

    def L(x, theta, phi):
        return torch.exp(theta * x).sum() * (phi * x).sum()

    def l(x, theta):
        return x.sum() * theta[0]

    theta = theta_0
    L_val = L(x_0, theta, phi)
    # create_graph=True allows AD through gradients. If we don't do this, grad() still works below, but
    # will "ignore" the effect of the gradient update, i.e., it will be computing D_theta_0 l(x, theta_0),
    # whereas we want D_theta_0 l(x, theta_1), where theta_1 = theta_0 - beta * D_theta_0 L(x_0, theta_0, phi)
    # similarly, g_phi will be None if create_graph=False, again because the operation involving gradients
    # is "ignored", in which case theta_1 = theta_0 and doesn't depend on phi.
    g_lower = grad(L_val, theta, create_graph=True) 
    theta = tuple_tensor_update(theta, g_lower, step_size=beta)
    
    l_val = l(x, theta)
    # g = (g_theta, g_phi), same as calling l_val.backward(), which directly stores g on the theta.grad and phi.grad
    g = grad(l_val, (theta_0, phi), retain_graph=True, allow_unused=True) 
    # l_val.backward()
    

if test_gradcheck:
    from torch.autograd import grad, gradcheck


    def LL(x):
        return x.sum()
    
    result = gradcheck(LL, torch.rand(10,4).double().requires_grad_())
    print (result)    


print ("Finished")