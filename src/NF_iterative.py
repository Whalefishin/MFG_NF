import torch
import torch.nn as nn
from torch.nn import functional as F, init
import numpy as np

from utils.torchutils import logabsdet
from nde.transforms import splines
import utils

import warnings



def _share_across_batch(params, batch_size):
    return params[None,...].expand(batch_size, *params.shape)


class NF_iterative(nn.Module):
    """Implements the NF forward and backward as iterative instead of recursive processes."""

    def __init__(self, transform, distribution, K):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        # super().__init__()
        super(NF_iterative, self).__init__()
        # self._transform = nn.ModuleList(transform)
        # self._transform = transform

        self.layer_dict = nn.ModuleDict()
        for i in range(len(transform)):
            self.layer_dict['transform_{}'.format(i)] = transform[i]


        self._distribution = distribution
        self.K = K


    def forward(self, x, context=None):
        outputs = x
        B       = x.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(x.device)]
        total_logabsdet = torch.zeros(B).to(x.device)
        total_OT_cost   = torch.zeros(B).to(x.device)

        # for f in self._transform:
        #     outputs, logabsdet, OT_cost, outputs_list, ld_list = f(outputs, context)
        #     total_logabsdet  += logabsdet
        #     total_OT_cost    += OT_cost
        #     total_ld         += ld_list
        #     total_outputs    += outputs_list

        for i in range(len(self.layer_dict)):
            outputs, logabsdet, OT_cost, outputs_list, ld_list = self.layer_dict['transform_{}'.format(i)](outputs, context)
            total_logabsdet  += logabsdet
            total_OT_cost    += OT_cost
            total_ld         += ld_list
            total_outputs    += outputs_list


        # partition history
        I = [3*i for i in range(self.K+1)]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def inverse(self, z, context=None):
        outputs = z
        B       = z.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(z.device)]
        total_logabsdet = torch.zeros(B).to(z.device)
        total_OT_cost   = torch.zeros(B).to(z.device)

        # for f in self._transform[::-1]:
        #     outputs, logabsdet, OT_cost, outputs_list, ld_list = f.inverse(outputs, context)
        # for i in range(len(self._transform)-1, -1, -1):
        #     outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform[i].inverse(outputs, context)
        for i in range(len(self.layer_dict)-1, -1, -1):
            outputs, logabsdet, OT_cost, outputs_list, ld_list = self.layer_dict['transform_{}'.format(i)].inverse(outputs, context)
            total_logabsdet  += logabsdet
            total_OT_cost    += OT_cost
            total_ld         += ld_list
            total_outputs    += outputs_list

        # partition history
        I = [3*i for i in range(self.K+1)]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def log_prob(self, inputs, context=None):
        # noise, logabsdet, OT_cost, hist, hist_ld = self._transform(inputs, context=context)
        noise, logabsdet, OT_cost, hist, hist_ld = self.forward(inputs, context=context)
        
        log_prob = self._distribution.log_prob(noise, context=context)
        return log_prob + logabsdet, log_prob, logabsdet, hist, hist_ld, OT_cost, noise


class NF_iterative_flatVar(nn.Module):
    """Implements the NF forward and backward as iterative instead of recursive processes."""

    def __init__(self, transform, distribution, K):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        # super().__init__()
        super(NF_iterative_flatVar, self).__init__()
        # self._transform = nn.ModuleList(transform)
        # self._transform = transform
        for i, f in enumerate(transform):
            setattr(self, 'transform_{}'.format(i), f)

        self._distribution = distribution
        self.n_transforms = len(transform)
        self.K = K


    def forward(self, x, context=None):
        outputs = x
        B       = x.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(x.device)]
        total_logabsdet = torch.zeros(B).to(x.device)
        total_OT_cost   = torch.zeros(B).to(x.device)

        for i in range(self.n_transforms):
            outputs, logabsdet, OT_cost, outputs_list, ld_list = getattr(self, 'transform_{}'.format(i))(outputs, context)
            total_logabsdet  += logabsdet
            total_OT_cost    += OT_cost
            total_ld         += ld_list
            total_outputs    += outputs_list


        # partition history
        I = [3*i for i in range(self.K+1)]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def inverse(self, z, context=None):
        outputs = z
        B       = z.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(z.device)]
        total_logabsdet = torch.zeros(B).to(z.device)
        total_OT_cost   = torch.zeros(B).to(z.device)

        for i in range(self.n_transforms-1, -1, -1):
            outputs, logabsdet, OT_cost, outputs_list, ld_list = \
                    getattr(self, 'transform_{}'.format(i)).inverse(outputs, context)
            total_logabsdet  += logabsdet
            total_OT_cost    += OT_cost
            total_ld         += ld_list
            total_outputs    += outputs_list

        # partition history
        I = [3*i for i in range(self.K+1)]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def log_prob(self, inputs, context=None):
        # noise, logabsdet, OT_cost, hist, hist_ld = self._transform(inputs, context=context)
        noise, logabsdet, OT_cost, hist, hist_ld = self.forward(inputs, context=context)
        
        log_prob = self._distribution.log_prob(noise, context=context)
        return log_prob + logabsdet, log_prob, logabsdet, hist, hist_ld, OT_cost, noise


    def sample(self, n_samples, context=None):
        noise = self._distribution.sample(n_samples, context=context)

        samples, logabsdet, OT_cost, hist, hist_ld = self.inverse(noise, context=context)

        return samples, logabsdet, OT_cost, hist, hist_ld, noise



class NF_iterative_test(nn.Module):
    """Implements the NF forward and backward as iterative instead of recursive processes."""

    def __init__(self, transform, distribution, K):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        # super().__init__()
        super(NF_iterative_test, self).__init__()
        # self._transform = nn.ModuleList(transform)
        self._transform0 = transform[0]
        # self._transform1 = transform[1]
        # self._transform2 = transform[2]
        self._distribution = distribution
        self.K = 1


    def forward(self, x, context=None):
        outputs = x
        B       = x.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(x.device)]
        total_logabsdet = torch.zeros(B).to(x.device)
        total_OT_cost   = torch.zeros(B).to(x.device)

        outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform0(outputs, context)
        total_logabsdet  += logabsdet
        total_OT_cost    += OT_cost
        total_ld         += ld_list
        total_outputs    += outputs_list

        # outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform1(outputs, context)
        # total_logabsdet  += logabsdet
        # total_OT_cost    += OT_cost
        # total_ld         += ld_list
        # total_outputs    += outputs_list

        # outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform2(outputs, context)
        # total_logabsdet  += logabsdet
        # total_OT_cost    += OT_cost
        # total_ld         += ld_list
        # total_outputs    += outputs_list

        # partition history
        I = [0,-1]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def inverse(self, z, context=None):
        outputs = z
        B       = z.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(z.device)]
        total_logabsdet = torch.zeros(B).to(z.device)
        total_OT_cost   = torch.zeros(B).to(z.device)

        # outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform2.inverse(outputs, context)
        # total_logabsdet  += logabsdet
        # total_OT_cost    += OT_cost
        # total_ld         += ld_list
        # total_outputs    += outputs_list

        # outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform1.inverse(outputs, context)
        # total_logabsdet  += logabsdet
        # total_OT_cost    += OT_cost
        # total_ld         += ld_list
        # total_outputs    += outputs_list

        outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform0.inverse(outputs, context)
        total_logabsdet  += logabsdet
        total_OT_cost    += OT_cost
        total_ld         += ld_list
        total_outputs    += outputs_list

        # partition history
        I = [0,-1]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def log_prob(self, inputs, context=None):
        # noise, logabsdet, OT_cost, hist, hist_ld = self._transform(inputs, context=context)
        noise, logabsdet, OT_cost, hist, hist_ld = self.forward(inputs, context=context)
        
        log_prob = self._distribution.log_prob(noise, context=context)
        return log_prob + logabsdet, log_prob, logabsdet, hist, hist_ld, OT_cost, noise


class NF_iterative_LU(nn.Module):
    """Implements the NF forward and backward as iterative instead of recursive processes."""

    def __init__(self, transform, distribution, K):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        # super().__init__()
        super(NF_iterative_LU, self).__init__()
        # self._transform = nn.ModuleList(transform)
        self._transform = transform
        self._distribution = distribution
        self.K = 1


    def forward(self, x, context=None):
        outputs = x
        B       = x.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(x.device)]
        total_logabsdet = torch.zeros(B).to(x.device)
        total_OT_cost   = torch.zeros(B).to(x.device)

        outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform(outputs, context)
        total_logabsdet  += logabsdet
        total_OT_cost    += OT_cost
        total_ld         += ld_list
        total_outputs    += outputs_list

        # partition history
        I = [0, -1]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def inverse(self, z, context=None):
        outputs = z
        B       = z.shape[0]
        total_outputs   = [outputs]
        total_ld        = [torch.zeros(B,1).to(z.device)]
        total_logabsdet = torch.zeros(B).to(z.device)
        total_OT_cost   = torch.zeros(B).to(z.device)

        # for f in self._transform[::-1]:
        #     outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform[i].inverse(outputs, context)
        # for i in range(len(self._transform)-1, -1, -1):

        outputs, logabsdet, OT_cost, outputs_list, ld_list = self._transform.inverse(outputs, context)
        total_logabsdet  += logabsdet
        total_OT_cost    += OT_cost
        total_ld         += ld_list
        total_outputs    += outputs_list

        # partition history
        I = [0, -1]
        hist = torch.stack(total_outputs)
        hist = hist[I].permute(1,0,2) # B x K+1 x d
        hist_ld = torch.stack(total_ld)
        hist_ld = hist_ld[I].permute(1,0,2) # B x K+1 x d

        return outputs, total_logabsdet, total_OT_cost, hist, hist_ld


    def log_prob(self, inputs, context=None):
        # noise, logabsdet, OT_cost, hist, hist_ld = self._transform(inputs, context=context)
        noise, logabsdet, OT_cost, hist, hist_ld = self.forward(inputs, context=context)
        
        log_prob = self._distribution.log_prob(noise, context=context)
        return log_prob + logabsdet, log_prob, logabsdet, hist, hist_ld, OT_cost, noise



class PiecewiseRationalQuadraticCouplingTransform(nn.Module):
    def __init__(self, mask, transform_net_create_fn,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 apply_unconditional_transform=False,
                 img_shape=None,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative
            )
        else:
            unconditional_transform = None


        # init() from CouplingTransform
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError('Mask must be a 1-dim tensor.')
        if mask.numel() <= 0:
            raise ValueError('Mask can\'t be empty.')

        super(PiecewiseRationalQuadraticCouplingTransform, self).__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer('identity_features', features_vector.masked_select(mask <= 0))
        self.register_buffer('transform_features', features_vector.masked_select(mask > 0))

        # self.identity_features  = features_vector.masked_select(mask <= 0)
        # self.transform_features = features_vector.masked_select(mask > 0)

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self._transform_dim_multiplier()
        )

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(
                features=self.num_identity_features
            )


    

        # super().__init__(mask, transform_net_create_fn,
        #                  unconditional_transform=unconditional_transform)

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def _transform_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins:]

        if hasattr(self.transform_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, 'hidden_channels'):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn('Inputs to the softmax are not scaled down: initialization might be bad.')

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)
            # transform_params = transform_params.clone().reshape(b, d, -1)

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, utils.sum_except_batch(logabsdet)

    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)


    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        identity_split  = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split,
            transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity =\
                self.unconditional_transform(identity_split, context)
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        OT_cost = torch.norm(inputs - outputs, dim=-1)**2

        return outputs, logabsdet, OT_cost, [outputs], [logabsdet.unsqueeze(-1)]

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError('Inputs must be a 2D or a 4D tensor.')

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(identity_split,
                                                                             context)

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split,
            transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        # OT
        OT_cost = torch.norm(inputs - outputs, dim=-1)**2

        return outputs, logabsdet, OT_cost, [outputs], [logabsdet.unsqueeze(-1)]





class PiecewiseRationalQuadraticCDF(nn.Module):
    def __init__(self,
                 shape,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 identity_init=False,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE):
        
        super(PiecewiseRationalQuadraticCDF, self).__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.tail_bound = tail_bound
        self.tails = tails

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = (num_bins - 1) if self.tails == 'linear' else (num_bins + 1)
            self.unnormalized_derivatives = nn.Parameter(constant * torch.ones(*shape,
                                                                               num_derivatives))
        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))

            num_derivatives = (num_bins - 1) if self.tails == 'linear' else (num_bins + 1)
            self.unnormalized_derivatives = nn.Parameter(torch.rand(*shape, num_derivatives))

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths=_share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights=_share_across_batch(self.unnormalized_heights, batch_size)
        unnormalized_derivatives=_share_across_batch(self.unnormalized_derivatives, batch_size)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)





class LinearCache(object):
    """Helper class to store the cache of a linear transform.

    The cache consists of: the weight matrix, its inverse and its log absolute determinant.
    """

    def __init__(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

class LULinear_iterative(nn.Module):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        # super().__init__(features, using_cache)
        super(LULinear_iterative, self).__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        # Caching flag and values.
        self.using_cache = using_cache
        self.cache = LinearCache()

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        # lower - L, upper - U
        lower, upper = self._create_lower_upper()
        outputs      = F.linear(inputs, upper)
        outputs      = F.linear(outputs, lower, self.bias)
        logabsdet    = self.logabsdet() * inputs.new_ones(outputs.shape[0])

        # no need to compute OT on the forward pass
        OT_cost   = torch.norm(inputs - outputs, dim=-1)**2

        return outputs, logabsdet, OT_cost, [outputs], [logabsdet.unsqueeze(-1)]

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs, _ = torch.triangular_solve(outputs.t(), lower, upper=False, unitriangular=True)
        outputs, _ = torch.triangular_solve(outputs, upper, upper=True, unitriangular=False)
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        # OT cost
        OT_cost = torch.norm(inputs - outputs, dim=-1)**2

        return outputs, logabsdet, OT_cost, [outputs], [logabsdet.unsqueeze(-1)]

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.trtrs(identity, lower, upper=False, unitriangular=True)
        weight_inverse, _ = torch.trtrs(lower_inverse, upper, upper=True, unitriangular=False)
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log(self.upper_diag))

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()

        elif self.cache.weight is None:
            self.cache.weight = self.weight()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = (-self.cache.logabsdet) * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            self.cache.inverse, self.cache.logabsdet = self.weight_inverse_and_logabsdet()

        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

class LULinear_test(nn.Module):
    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        # super().__init__(features, using_cache)
        super(LULinear_test, self).__init__()

        self.f = nn.Linear(features, features)
        self.f_inv = nn.Linear(features, features)


    def forward(self, inputs, context):
        B = inputs.shape[0]
        outputs = self.f(inputs)
        logabsdet = torch.slogdet(self.f.weight)[-1].repeat(B)
        OT_cost = 0

        return outputs, logabsdet, OT_cost, [outputs], [logabsdet.unsqueeze(-1)]

    def inverse(self, inputs, context):
        B = inputs.shape[0]
        outputs = self.f_inv(inputs)
        logabsdet = torch.slogdet(self.f_inv.weight)[-1].repeat(B)
        OT_cost = 0

        return outputs, logabsdet, OT_cost, [outputs], [logabsdet.unsqueeze(-1)]



class NN_test(nn.Module):
    def __init__(self, d_in, d_out):
        # super().__init__(features, using_cache)
        super(NN_test, self).__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, d_out)
        )

    def forward(self, x, context):
        return self.net(x)