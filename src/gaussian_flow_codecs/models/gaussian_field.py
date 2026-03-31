import torch
import torch.nn.functional as F


class GaussianField:
    def __init__(
        self,
        mu,
        sigma,
        amp,
        raw_mu=None,
        raw_sigma=None,
        raw_covariance=None,
        raw_beta_shape=None,
        min_sigma=1e-3,
        max_sigma=0.5,
        blend_mode="normalized",
        covariance_mode="diagonal",
        kernel_type="gaussian",
        beta_shape=6.0,
        min_beta_shape=1.01,
        max_beta_shape=20.0,
    ):
        """
        Parameters
        ----------
        mu:
            Kernel centers with shape (N, 3).
        sigma:
            Initial axis scales with shape (N, 3). These are used directly for
            diagonal kernels and as the diagonal initialization for full
            covariance kernels.
        amp:
            Kernel amplitudes with shape (N, 3).
        raw_mu:
            Optional unconstrained center parameterization.
        raw_sigma:
            Optional unconstrained diagonal scale parameterization.
        raw_covariance:
            Optional unconstrained lower-triangular Cholesky parameterization
            with shape (N, 6), ordered as [l00, l10, l11, l20, l21, l22].
        """

        self._mu = mu
        self.raw_mu = raw_mu
        self._sigma = sigma
        self.raw_sigma = raw_sigma
        self.raw_covariance = raw_covariance
        self.raw_beta_shape = raw_beta_shape
        self.amp = amp
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.blend_mode = blend_mode
        self.covariance_mode = covariance_mode
        self.kernel_type = kernel_type
        self._beta_shape = float(beta_shape)
        self.min_beta_shape = min_beta_shape
        self.max_beta_shape = max_beta_shape

        if covariance_mode not in {"diagonal", "full"}:
            raise ValueError(f"Unsupported covariance_mode: {covariance_mode}")
        if covariance_mode == "full" and raw_covariance is None:
            raise ValueError("raw_covariance is required when covariance_mode='full'.")
        if kernel_type not in {"gaussian", "beta"}:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")
        if kernel_type == "beta" and covariance_mode != "diagonal":
            raise ValueError("kernel_type='beta' currently supports covariance_mode='diagonal' only.")
        if beta_shape <= 1.0:
            raise ValueError("beta_shape must be greater than 1.0.")
        if max_beta_shape <= min_beta_shape:
            raise ValueError("max_beta_shape must be greater than min_beta_shape.")

    @property
    def mu(self):
        if self.raw_mu is None:
            return self._mu
        return torch.sigmoid(self.raw_mu)

    def _bounded_diagonal(self, raw_diagonal):
        sigma = self.min_sigma + F.softplus(raw_diagonal)
        return torch.clamp(sigma, max=self.max_sigma)

    @property
    def cholesky_factor(self):
        if self.covariance_mode != "full":
            raise AttributeError("cholesky_factor is only defined for full covariance kernels.")

        raw = self.raw_covariance
        factor = torch.zeros((raw.shape[0], 3, 3), dtype=raw.dtype, device=raw.device)
        factor[:, 0, 0] = self._bounded_diagonal(raw[:, 0])
        factor[:, 1, 0] = torch.tanh(raw[:, 1]) * self.max_sigma
        factor[:, 1, 1] = self._bounded_diagonal(raw[:, 2])
        factor[:, 2, 0] = torch.tanh(raw[:, 3]) * self.max_sigma
        factor[:, 2, 1] = torch.tanh(raw[:, 4]) * self.max_sigma
        factor[:, 2, 2] = self._bounded_diagonal(raw[:, 5])
        return factor

    @property
    def covariance_matrix(self):
        if self.covariance_mode == "diagonal":
            sigma_sq = self.sigma**2
            return torch.diag_embed(sigma_sq)
        factor = self.cholesky_factor
        return factor @ factor.transpose(-1, -2)

    @property
    def sigma(self):
        if self.covariance_mode == "diagonal":
            if self.raw_sigma is None:
                return self._sigma
            return self._bounded_diagonal(self.raw_sigma)

        covariance = self.covariance_matrix
        return torch.sqrt(torch.clamp(torch.diagonal(covariance, dim1=-2, dim2=-1), min=1e-12))

    @property
    def shape_parameter_count(self):
        count = self.mu.shape[0] * (6 if self.covariance_mode == "full" else 3)
        if self.kernel_type == "beta":
            count += self.mu.shape[0] * 3
        return count

    @property
    def beta_shape(self):
        if self.kernel_type != "beta":
            return None
        if self.raw_beta_shape is None:
            return torch.full_like(self.sigma, self._beta_shape)
        beta = self.min_beta_shape + F.softplus(self.raw_beta_shape)
        return torch.clamp(beta, max=self.max_beta_shape)

    def get_shape_state(self):
        state = {}
        if self.covariance_mode == "full":
            state["raw_covariance"] = self.raw_covariance.detach().clone()
        elif self.raw_sigma is not None:
            state["raw_sigma"] = self.raw_sigma.detach().clone()
        else:
            state["sigma"] = self._sigma.detach().clone()

        if self.kernel_type == "beta":
            if self.raw_beta_shape is not None:
                state["raw_beta_shape"] = self.raw_beta_shape.detach().clone()
            else:
                state["beta_shape"] = self.beta_shape.detach().clone()
        return state

    def load_shape_state_(self, state):
        if self.covariance_mode == "full":
            self.raw_covariance.copy_(state["raw_covariance"])
        elif self.raw_sigma is not None:
            self.raw_sigma.copy_(state["raw_sigma"])
        elif "sigma" in state:
            self._sigma.copy_(state["sigma"])

        if self.kernel_type == "beta" and self.raw_beta_shape is not None and "raw_beta_shape" in state:
            self.raw_beta_shape.copy_(state["raw_beta_shape"])

    def evaluate(self, x):
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)

        if self.kernel_type == "beta":
            weights = self._evaluate_beta_weights(diff)
        elif self.covariance_mode == "diagonal":
            scaled_diff = diff / self.sigma.unsqueeze(0)
            dist2 = torch.sum(scaled_diff**2, dim=-1)
            weights = torch.exp(-dist2)
        else:
            factor = self.cholesky_factor.unsqueeze(0)
            rhs = diff.unsqueeze(-1)
            solved = torch.linalg.solve_triangular(factor, rhs, upper=False)
            dist2 = torch.sum(solved.squeeze(-1) ** 2, dim=-1)
            weights = torch.exp(-dist2)

        if self.blend_mode == "normalized":
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
        elif self.blend_mode != "raw":
            raise ValueError(f"Unsupported blend_mode: {self.blend_mode}")

        return torch.einsum("bn,nc->bc", weights, self.amp)

    def _evaluate_beta_weights(self, diff):
        """
        Compact-support separable beta bump.

        The support is limited to |x - mu| < sigma along each axis, which makes
        the kernel sharper than a Gaussian and therefore worth testing on
        discontinuity-prone data such as shocks.
        """

        sigma = self.sigma.unsqueeze(0)
        local = diff / (sigma + 1e-12)
        t = 0.5 * (local + 1.0)
        interior = (t > 0.0) & (t < 1.0)

        a = self.beta_shape.unsqueeze(0)
        beta_profile = torch.where(
            interior,
            torch.pow(torch.clamp(t, min=1e-12), a - 1.0)
            * torch.pow(torch.clamp(1.0 - t, min=1e-12), a - 1.0),
            torch.zeros_like(t),
        )
        return torch.prod(beta_profile, dim=-1)
