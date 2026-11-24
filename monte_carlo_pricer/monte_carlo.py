"""
Layer: Business rules — Monte Carlo pricing under the risk-neutral measure.
Monte Carlo pricing of European options under the risk-neutral measure.

The underlying follows a geometric Brownian motion with drift ``r`` (the
risk-free rate) and volatility ``sigma``. The price is the discounted
expected payoff:

    V = exp(-r T) * E[max(S_T - K, 0)]          (call)
    V = exp(-r T) * E[max(K - S_T, 0)]          (put)

Antithetic variates (each normal draw Z is paired with -Z) roughly halve
the standard error at no extra simulation cost. The function returns the
price together with its Monte Carlo standard error, so every result
carries its own uncertainty.
"""

import numpy as np

from .black_scholes import black_scholes_price

__all__ = ["mc_terminal_prices", "mc_european_price"]


def mc_terminal_prices(s0, r, sigma, t, n_paths, seed=None, antithetic=True):
    """Simulate terminal prices S_T of a GBM under the risk-neutral measure.

    Uses the exact GBM solution (Ito), so there is no discretization error:
    S_T = S_0 * exp((r - sigma^2 / 2) T + sigma sqrt(T) Z).

    With ``antithetic=True`` (default), half the draws are mirrored, so the
    returned array has ``n_paths`` entries built from ``n_paths // 2``
    independent normals.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        n_half = (n_paths + 1) // 2
        z = rng.standard_normal(n_half)
        z = np.concatenate([z, -z])[:n_paths]
    else:
        z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * t
    return s0 * np.exp(drift + sigma * np.sqrt(t) * z)


def mc_european_price(s0, k, t, r, sigma, n_paths=200_000, option="call",
                      seed=None, antithetic=True):
    """Monte Carlo price of a European option.

    Returns ``(price, standard_error)``. The standard error is the
    standard error of the discounted payoff mean, i.e. the Monte Carlo
    uncertainty of the estimate.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if t <= 0:
        intrinsic = max(s0 - k, 0.0) if option == "call" else max(k - s0, 0.0)
        return float(intrinsic), 0.0

    s_t = mc_terminal_prices(s0, r, sigma, t, n_paths, seed=seed,
                             antithetic=antithetic)
    if option == "call":
        payoffs = np.maximum(s_t - k, 0.0)
    else:
        payoffs = np.maximum(k - s_t, 0.0)
    disc = np.exp(-r * t)
    discounted = disc * payoffs
    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / np.sqrt(n_paths))
    return price, std_error


def mc_vs_bs_error(s0, k, t, r, sigma, n_paths=200_000, option="call",
                   seed=None):
    """Convenience helper: (mc_price, bs_price, error, std_error)."""
    mc_price, se = mc_european_price(s0, k, t, r, sigma, n_paths=n_paths,
                                     option=option, seed=seed)
    bs_price = black_scholes_price(s0, k, t, r, sigma, option=option)
    return mc_price, bs_price, mc_price - bs_price, se
