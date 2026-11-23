"""Closed-form Black-Scholes pricing and analytical Greeks.

The reference solution against which the Monte Carlo pricer is validated.
Only the standard normal CDF/PDF from ``scipy.stats`` is used; no pricing
library is involved.

Conventions
-----------
* ``option`` is ``"call"`` or ``"put"``.
* ``r`` is the continuously-compounded risk-free rate, ``sigma`` the
  annualized volatility, ``t`` the time to expiry in years.
* Vega is quoted per unit of volatility (divide by 100 for the usual
  "per 1 vol point" quote) and theta per year (divide by 365 for per-day).
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(s, k, t, r, sigma):
    """Return (d1, d2) of the Black-Scholes formula."""
    if t <= 0:
        raise ValueError("t must be positive for d1/d2; use intrinsic value at expiry")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    vol = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / vol
    d2 = d1 - vol
    return d1, d2


def black_scholes_price(s, k, t, r, sigma, option="call"):
    """European option price under Black-Scholes.

    Parameters
    ----------
    s : float
        Spot price of the underlying.
    k : float
        Strike.
    t : float
        Time to expiry in years (t = 0 returns the intrinsic value).
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Annualized volatility.
    option : {"call", "put"}

    Returns
    -------
    float
        The option price.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if t <= 0:
        return max(s - k, 0.0) if option == "call" else max(k - s, 0.0)
    if sigma <= 0:
        # Deterministic forward: discounted intrinsic value of the forward.
        forward = s * np.exp(r * t)
        payoff = max(forward - k, 0.0) if option == "call" else max(k - forward, 0.0)
        return float(np.exp(-r * t) * payoff)

    d1, d2 = _d1_d2(s, k, t, r, sigma)
    disc = np.exp(-r * t)
    if option == "call":
        return float(s * norm.cdf(d1) - k * disc * norm.cdf(d2))
    return float(k * disc * norm.cdf(-d2) - s * norm.cdf(-d1))


def black_scholes_greeks(s, k, t, r, sigma, option="call"):
    """Analytical Black-Scholes Greeks.

    Returns a dict with delta, gamma, vega (per unit vol), theta (per year)
    and rho (per unit rate). Gamma is identical for calls and puts.
    """
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    if t <= 0 or sigma <= 0:
        raise ValueError("Greeks require t > 0 and sigma > 0")

    d1, d2 = _d1_d2(s, k, t, r, sigma)
    disc = np.exp(-r * t)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(t)

    gamma = pdf_d1 / (s * sigma * sqrt_t)
    vega = s * pdf_d1 * sqrt_t
    if option == "call":
        delta = norm.cdf(d1)
        theta = -(s * pdf_d1 * sigma) / (2 * sqrt_t) - r * k * disc * norm.cdf(d2)
        rho = k * t * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = -(s * pdf_d1 * sigma) / (2 * sqrt_t) + r * k * disc * norm.cdf(-d2)
        rho = -k * t * disc * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
