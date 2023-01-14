import torch
import torch.nn as nn
import numpy as np

class NF_list(nn.Module):
    def __init__(self, flows, distributions, args):
        super().__init__()
        self.args  = args
        self.distributions = distributions
        self.flows = nn.ModuleList(flows)

    # TODO: in principle, forward() and sample() could be rewritten so that they don't involve for-loops.
    # one possibility: rework the weights in the large network to be block-diagonal of the smaller ones
    # second possibility: spawn multiple CPU threads to put each NF on the GPU.
    # https://discuss.pytorch.org/t/how-to-write-parallel-for-loop-in-models-forward-method/51652

    def forward(self, x):
        log_density_list = [] # N_p x B 
        hist_norm_list   = [] # N_p x B x K x d

        for i, F_i in enumerate(self.flows):
            log_density, _, _, hist_norm, _, _, _ = F_i.log_prob(x[i])
            log_density_list.append(log_density)
            hist_norm_list.append(torch.cat(hist_norm).reshape(len(hist_norm), hist_norm[0].shape[0],\
                                    hist_norm[0].shape[-1]).permute(1,0,2))
        
        return torch.stack(log_density_list), torch.stack(hist_norm_list)
            

    def inverse(self, z):
        hist_gen_list = []

        for i, F_i in enumerate(self.flows):
            _, _, _, hist, _ = F_i._transform.inverse(z[i], context=None)
            hist_gen_list.append(torch.cat(hist).reshape(len(hist), hist[0].shape[0],\
                                    hist[0].shape[-1]).permute(1,0,2))

        return torch.stack(hist_gen_list)


    def sample(self, n, device=0):
        z_0_list = []
        z_K_list = []
        hist_gen_list = []
        ld_gen_list   = []

        for i, F_i in enumerate(self.flows):
            z_K, ld_gen, _, hist_gen, _, z_0 = F_i.sample(n)
            z_0_list.append(z_0)
            z_K_list.append(z_K)
            ld_gen_list.append(ld_gen)
            hist_gen_list.append(torch.cat(hist_gen).reshape(len(hist_gen), hist_gen[0].shape[0],\
                                    hist_gen[0].shape[-1]).permute(1,0,2))
            

        return torch.stack(z_K_list), torch.stack(z_0_list), torch.stack(ld_gen_list), torch.stack(hist_gen_list)
