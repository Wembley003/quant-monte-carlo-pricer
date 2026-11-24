"""
Layer: Business rules — the stochastic processes that drive the simulation.
Stochastic processes: simulation and analytical results.

This module is the explicit stochastic-processes core of the project. It
covers the two processes a quant interview will ask about, each with its
exact solution, a discretization scheme, and the analytical moments used
to validate the simulation.

Geometric Brownian motion (GBM)
-------------------------------
The workhorse of Black-Scholes:

    dS_t = mu * S_t dt + sigma * S_t dW_t

Ito's lemma applied to Y_t = ln S_t gives the drift correction:

    dY_t = (mu - sigma^2 / 2) dt + sigma dW_t

so the exact solution is

    S_T = S_0 * exp((mu - sigma^2 / 2) * T + sigma * W_T)

with analytical moments

    E[S_T]   = S_0 * exp(mu * T)
    Var(S_T) = S_0^2 * exp(2 * mu * T) * (exp(sigma^2 * T) - 1)

and E[ln S_T] = ln S_0 + (mu - sigma^2 / 2) * T, Var(ln S_T) = sigma^2 * T.
The ``-sigma^2 / 2`` term is the Ito correction: the median of S_T grows
at mu - sigma^2 / 2, not at mu.

Ornstein-Uhlenbeck (OU) — mean reversion
-----------------------------------------
    dX_t = theta * (mu - X_t) dt + sigma dW_t

Exact solution:

    X_t = X_0 * exp(-theta * t) + mu * (1 - exp(-theta * t))
          + sigma * integral_0^t exp(-theta * (t - s)) dW_s

with analytical moments

    E[X_t]   = X_0 * exp(-theta * t) + mu * (1 - exp(-theta * t))
    Var(X_t) = sigma^2 / (2 * theta) * (1 - exp(-2 * theta * t))

and stationary (long-run) distribution N(mu, sigma^2 / (2 * theta)).
OU is the standard model for mean-reverting quantities (interest rates
in Vasicek, spreads, volatility) and the natural counterpart to GBM when
a series reverts instead of trending.

Discretization schemes
----------------------
Euler-Maruyama (strong order 1/2) for dX = a(X) dt + b(X) dW:

    X_{n+1} = X_n + a(X_n) dt + b(X_n) sqrt(dt) Z

Milstein (strong order 1) adds the Ito-Taylor correction:

    X_{n+1} = X_n + a(X_n) dt + b(X_n) sqrt(dt) Z
              + 0.5 * b(X_n) * b'(X_n) * (dt * Z^2 - dt)

For GBM, b(S) = sigma * S and b'(S) = sigma, so the correction is
0.5 * sigma^2 * S_n * (dt * Z^2 - dt). For OU, b is constant, so
b' = 0 and Milstein coincides with Euler-Maruyama; that is why only
the exact and Euler-Maruyama schemes are provided for OU.

All simulators return ``(times, paths)`` with ``paths`` of shape
``(n_steps + 1, n_paths)``. Every simulator draws its standard normals
as a single ``rng.standard_normal((n_steps, n_paths))`` call, so two
simulators run with the same seed share the same Brownian increments
and can be compared path by path (common random numbers).
"""

import numpy as np

__all__ = [
    "ito_log_drift",
    "gbm_moments",
    "ou_moments",
    "ou_stationary_moments",
    "simulate_gbm_exact",
    "simulate_gbm_euler_maruyama",
    "simulate_gbm_milstein",
    "simulate_ou_exact",
    "simulate_ou_euler_maruyama",
]


def _rng(seed):
    return np.random.default_rng(seed)


def _grid(t, n_steps):
    dt = t / n_steps
    times = np.linspace(0.0, t, n_steps + 1)
    return dt, times


# ---------------------------------------------------------------------------
# Analytical results
# ---------------------------------------------------------------------------


def ito_log_drift(mu, sigma):
    """Drift of ln S_t for a GBM with drift mu and volatility sigma.

    Ito's lemma: d(ln S_t) = (mu - sigma^2 / 2) dt + sigma dW_t.
    The ``-sigma^2 / 2`` term is the Ito correction.
    """
    return mu - 0.5 * sigma**2


def gbm_moments(s0, mu, sigma, t):
    """Analytical (mean, variance) of S_t for GBM starting at ``s0``."""
    mean = s0 * np.exp(mu * t)
    var = s0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1.0)
    return float(mean), float(var)


def ou_moments(x0, theta, mu, sigma, t):
    """Analytical (mean, variance) of X_t for OU starting at ``x0``."""
    decay = np.exp(-theta * t)
    mean = x0 * decay + mu * (1.0 - decay)
    var = sigma**2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * t))
    return float(mean), float(var)


def ou_stationary_moments(theta, mu, sigma):
    """Stationary (mean, variance) of the OU process: N(mu, sigma^2 / 2theta)."""
    return float(mu), float(sigma**2 / (2.0 * theta))


# ---------------------------------------------------------------------------
# Geometric Brownian motion
# ---------------------------------------------------------------------------


def simulate_gbm_exact(s0, mu, sigma, t, n_steps, n_paths, seed=None):
    """GBM paths from the exact solution (no discretization error).

    S_{n+1} = S_n * exp((mu - sigma^2 / 2) dt + sigma sqrt(dt) Z).
    """
    rng = _rng(seed)
    dt, times = _grid(t, n_steps)
    z = rng.standard_normal((n_steps, n_paths))
    log_increments = ito_log_drift(mu, sigma) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.log(s0) + np.vstack(
        [np.zeros(n_paths), np.cumsum(log_increments, axis=0)]
    )
    paths = np.exp(log_paths)
    paths[0] = s0  # avoid last-ulp drift from exp(log(s0))
    return times, paths


def simulate_gbm_euler_maruyama(s0, mu, sigma, t, n_steps, n_paths, seed=None):
    """GBM paths by Euler-Maruyama (strong order 1/2).

    S_{n+1} = S_n + mu S_n dt + sigma S_n sqrt(dt) Z.
    """
    rng = _rng(seed)
    dt, times = _grid(t, n_steps)
    z = rng.standard_normal((n_steps, n_paths))
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = s0
    sqrt_dt = np.sqrt(dt)
    for n in range(n_steps):
        s = paths[n]
        paths[n + 1] = s + mu * s * dt + sigma * s * sqrt_dt * z[n]
    return times, paths


def simulate_gbm_milstein(s0, mu, sigma, t, n_steps, n_paths, seed=None):
    """GBM paths by the Milstein scheme (strong order 1).

    S_{n+1} = S_n + mu S_n dt + sigma S_n sqrt(dt) Z
              + 0.5 sigma^2 S_n (dt Z^2 - dt).
    """
    rng = _rng(seed)
    dt, times = _grid(t, n_steps)
    z = rng.standard_normal((n_steps, n_paths))
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = s0
    sqrt_dt = np.sqrt(dt)
    for n in range(n_steps):
        s = paths[n]
        zn = z[n]
        paths[n + 1] = (
            s
            + mu * s * dt
            + sigma * s * sqrt_dt * zn
            + 0.5 * sigma**2 * s * (dt * zn**2 - dt)
        )
    return times, paths


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck (mean reversion)
# ---------------------------------------------------------------------------


def simulate_ou_exact(x0, theta, mu, sigma, t, n_steps, n_paths, seed=None):
    """OU paths from the exact Gaussian transition.

    X_{n+1} = X_n exp(-theta dt) + mu (1 - exp(-theta dt))
              + sigma sqrt((1 - exp(-2 theta dt)) / (2 theta)) Z.
    """
    rng = _rng(seed)
    dt, times = _grid(t, n_steps)
    z = rng.standard_normal((n_steps, n_paths))
    decay = np.exp(-theta * dt)
    shock_sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = x0
    for n in range(n_steps):
        paths[n + 1] = paths[n] * decay + mu * (1.0 - decay) + shock_sd * z[n]
    return times, paths


def simulate_ou_euler_maruyama(x0, theta, mu, sigma, t, n_steps, n_paths, seed=None):
    """OU paths by Euler-Maruyama (strong order 1/2).

    X_{n+1} = X_n + theta (mu - X_n) dt + sigma sqrt(dt) Z.

    Milstein adds nothing here: the diffusion coefficient is constant,
    so b'(X) = 0 and the Milstein correction vanishes.
    """
    rng = _rng(seed)
    dt, times = _grid(t, n_steps)
    z = rng.standard_normal((n_steps, n_paths))
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = x0
    sqrt_dt = np.sqrt(dt)
    for n in range(n_steps):
        x = paths[n]
        paths[n + 1] = x + theta * (mu - x) * dt + sigma * sqrt_dt * z[n]
    return times, paths
