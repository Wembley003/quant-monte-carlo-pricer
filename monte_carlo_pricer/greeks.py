"""Greeks by finite differences (Monte Carlo) and analytical formulas.

Monte Carlo Greeks use bump-and-revalue with common random numbers: the
base price and every bumped price are computed with the same seed, so the
same Brownian draws are reused and the difference is not drowned in
independent Monte Carlo noise. Central differences are used for delta,
gamma and vega; forward differences for theta and rho.

All Greeks are returned per unit of the bumped variable (vega per unit of
volatility, theta per year, rho per unit of rate).
"""

import numpy as np

from .black_scholes import black_scholes_greeks
from .monte_carlo import mc_european_price

__all__ = ["mc_greeks", "compare_greeks"]


def mc_greeks(s0, k, t, r, sigma, n_paths=200_000, option="call", seed=42,
              d_s=0.5, d_sigma=0.01, d_t=1.0 / 365.0, d_r=0.0001):
    """Monte Carlo Greeks by finite differences with common random numbers."""
    base, _ = mc_european_price(s0, k, t, r, sigma, n_paths=n_paths,
                                option=option, seed=seed)

    up, _ = mc_european_price(s0 + d_s, k, t, r, sigma, n_paths=n_paths,
                              option=option, seed=seed)
    down, _ = mc_european_price(s0 - d_s, k, t, r, sigma, n_paths=n_paths,
                                option=option, seed=seed)
    delta = (up - down) / (2 * d_s)
    gamma = (up - 2 * base + down) / d_s**2

    v_up, _ = mc_european_price(s0, k, t, r, sigma + d_sigma, n_paths=n_paths,
                                option=option, seed=seed)
    v_down, _ = mc_european_price(s0, k, t, r, sigma - d_sigma, n_paths=n_paths,
                                  option=option, seed=seed)
    vega = (v_up - v_down) / (2 * d_sigma)

    # Theta: price change when time to expiry shrinks by d_t.
    t_down = max(t - d_t, 1e-6)
    th, _ = mc_european_price(s0, k, t_down, r, sigma, n_paths=n_paths,
                              option=option, seed=seed)
    theta = (th - base) / (t - t_down)

    r_up, _ = mc_european_price(s0, k, t, r + d_r, sigma, n_paths=n_paths,
                                option=option, seed=seed)
    r_down, _ = mc_european_price(s0, k, t, r - d_r, sigma, n_paths=n_paths,
                                  option=option, seed=seed)
    rho = (r_up - r_down) / (2 * d_r)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def compare_greeks(s0, k, t, r, sigma, n_paths=200_000, option="call",
                   seed=42):
    """Return a dict of {greek: (mc_value, analytical_value, abs_error)}."""
    mc = mc_greeks(s0, k, t, r, sigma, n_paths=n_paths, option=option, seed=seed)
    an = black_scholes_greeks(s0, k, t, r, sigma, option=option)
    return {g: (mc[g], an[g], abs(mc[g] - an[g])) for g in an}
