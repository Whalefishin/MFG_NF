import torch
from torch import nn
import torch.nn.functional as F


class DoubleNormalizingFlow(nn.Module):
    def __init__(self, flow_01, flow_12, args, AE=None):
        super().__init__()
        # want: F_*(P_1) = P_2. Relax to: H_*(P_0) = P_1, F_*(H_*(P_0)) = P_2
        self.flow01 = flow_01
        self.flow12 = flow_12
        self.args   = args
        self.AE     = AE

    def forward(self, x_1, x_2):
        ae_loss_1 = torch.zeros(x_1.shape[0]).to(x_1.device)
        ae_loss_2 = torch.zeros(x_2.shape[0]).to(x_2.device)

        if self.AE is not None:
            # use AE to parametrize both x_1 and x_2
            x_1_recon, mu_1, log_var_1 = self.AE(x_1)
            ae_loss_1, _, _ = self.vae_loss(x_1_recon, x_1, mu_1, log_var_1)
            x_2_recon, mu_2, log_var_2 = self.AE(x_2)
            ae_loss_2, _, _ = self.vae_loss(x_2_recon, x_2, mu_2, log_var_2)

            # use the features as input to the NF
            x_1 = mu_1
            x_2 = mu_2

        # for D(P_1 || H_*(P_0))
        log_density_1, _, _, hist_norm_01, _, _, z_1 = self.flow01.log_prob(x_1)

        # for D(P_2 || (F.H)_*(P_0)) 
        _, log_prob_F, ld_F, hist_norm_12, hist_ld_norm_12, OT_cost_norm_12, z_2 = self.flow12.log_prob(x_2)
        _, log_prob_H_F, ld_H_F, _, _, _, z_2 = self.flow01.log_prob(z_2)

        # Note: log_prob_F is not used.
        log_density_2 = log_prob_H_F + ld_H_F + ld_F

        return log_density_1, log_density_2, hist_norm_01, hist_norm_12, z_1, z_2, OT_cost_norm_12, ae_loss_1, ae_loss_2
    
    def map_12(self, x_1):
        """map samples from P1 to P2

        Args:
            x_1 (tensor): samples from P1

        Returns:
            x_2: samples from P2
        """
        if self.AE is not None:
            # use the features as input to the NF
            _, mu_1, _ = self.AE(x_1)
            x_1 = mu_1

        return self.flow12._transform.inverse(x_1)

    # def sample_and_transform_P1(self, x_1):
    #     x_2, logabsdet, OT_cost, hist, hist_ld = self.flow12._transform.inverse(x_1)

    # def vae_loss(self, recon_x, x, mu, log_var):
    #     B   = x.shape[0]
    #     BCE = F.binary_cross_entropy(recon_x, x.view(B, -1), reduction='sum')
    #     KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    #     return BCE + KLD, BCE, KLD

    def vae_loss(self, recon_x, x, mu, log_var):
        B   = x.shape[0]
        # BCE = F.binary_cross_entropy(recon_x, x.view(B, -1), reduction='none')
        # KLD = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        # BCE = torch.mean(F.binary_cross_entropy(recon_x, x.view(B, -1), reduction='none'), dim=-1)
        # KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        BCE = torch.sum(F.binary_cross_entropy(recon_x, x.view(B, -1), reduction='none'), dim=-1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return BCE + KLD, BCE, KLD


# class DoubleNormalizingFlow_VAE(nn.Module):
#     def __init__(self, flow_01, flow_12, VAE, args):
#         super().__init__()
#         # want: F_*(P_1) = P_2. Relax to: H_*(P_0) = P_1, F_*(H_*(P_0)) = P_2
#         self.flow01 = flow_01
#         self.flow12 = flow_12
#         self.VAE    = VAE
#         self.args   = args
    
#     def forward(self, x_1, x_2):
#         # for D(P_1 || H_*(P_0))
#         log_density_1, _, _, hist_norm_01, _, _, z_1 = self.flow01.log_prob(x_1)

#         # for D(P_2 || (F.H)_*(P_0)) 
#         _, log_prob_F, ld_F, hist_norm_12, hist_ld_norm_12, OT_cost_norm_12, z_2 = self.flow12.log_prob(x_2)
#         _, log_prob_H_F, ld_H_F, _, _, _, z_2 = self.flow01.log_prob(z_2)

#         # Note: log_prob_F is not used.
#         log_density_2 = log_prob_H_F + ld_H_F + ld_F

#         return log_density_1, log_density_2, hist_norm_01, hist_norm_12, z_1, z_2, OT_cost_norm_12
    
#     def vae_loss(self, recon_x, x, mu, log_var):
#         B   = x.shape[0]
#         BCE = F.binary_cross_entropy(recon_x, x.view(B, -1), reduction='sum')
#         KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

#         return BCE + KLD, BCE, KLD