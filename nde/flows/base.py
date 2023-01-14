"""Basic definitions for the flows module."""
import utils

from nde import distributions


class Flow(distributions.Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution

    def _log_prob(self, inputs, context):
        noise, logabsdet, OT_cost, hist, hist_ld = self._transform(inputs, context=context)
        log_prob                  = self._distribution.log_prob(noise, context=context)
        return log_prob + logabsdet, log_prob, logabsdet, hist, hist_ld, OT_cost, noise

    def _sample(self, num_samples, context):
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise   = utils.merge_leading_dims(noise, num_dims=2)
            context = utils.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet, OT_cost, hist, hist_ld = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = utils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples, logabsdet, OT_cost, hist, hist_ld, noise

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = utils.merge_leading_dims(noise, num_dims=2)
            context = utils.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = utils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = utils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _, _, _, _ = self._transform(inputs, context=context)
        return noise

    def forward_ret_z(self, inputs):
        return self._transform(inputs, context=None)[0]

    # def forward_bilevel(self, inputs):
    #     noise, logabsdet, OT_cost, hist, hist_ld = self._transform(inputs)
    #     return hist # B x K x d


class SingleFlow(distributions.Distribution):
    def __init__(self, transform, distribution, K):
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        self.K = K

    def _log_prob(self, inputs, context):
        ld_sum      = 0
        OT_cost_sum = 0
        hist_all    = []
        hist_ld_all = []

        x_k = inputs
        for k in range(self.K):
            x_k, logabsdet, OT_cost, hist, hist_ld = self._transform(x_k, context=context)
            ld_sum      += logabsdet
            OT_cost_sum += OT_cost
            hist_all    += hist
            hist_ld_all += hist_ld
        
        log_prob = self._distribution.log_prob(x_k, context=context)

        return log_prob + ld_sum, hist_all, hist_ld_all, OT_cost_sum

    def _sample(self, num_samples, context):
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise   = utils.merge_leading_dims(noise, num_dims=2)
            context = utils.repeat_rows(context, num_reps=num_samples)

        ld_sum      = 0
        OT_cost_sum = 0
        hist_all    = []
        hist_ld_all = []

        z_k = noise
        for k in range(self.K):
            z_k, logabsdet, OT_cost, hist, hist_ld = self._transform.inverse(z_k, context=context)
            ld_sum      += logabsdet
            OT_cost_sum += OT_cost
            hist_all    += hist
            hist_ld_all += hist_ld

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = utils.split_leading_dim(samples, shape=[-1, num_samples])

        return z_k, ld_sum, OT_cost_sum, hist_all, hist_ld_all, noise

    def inverse(self, x_in):
        ld_sum      = 0
        OT_cost_sum = 0
        hist_all    = []
        hist_ld_all = []

        x = x_in
        for k in range(self.K):
            x, logabsdet, OT_cost, hist, hist_ld = self._transform.inverse(x)
            ld_sum      += logabsdet
            OT_cost_sum += OT_cost
            hist_all    += hist
            hist_ld_all += hist_ld

        return x, ld_sum, OT_cost_sum, hist_all, hist_ld_all

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = utils.merge_leading_dims(noise, num_dims=2)
            context = utils.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = utils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = utils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _, _, _, _ = self._transform(inputs, context=context)
        return noise
