"""
Layer: Business rules — inverting Black-Scholes for the implied volatility.
Implied volatility: invert the Black-Scholes formula.

Given a market price, find the sigma such that the Black-Scholes price
matches it. The Black-Scholes price is monotonically increasing in sigma
for vanilla options, so bisection on a wide bracket is robust; a
Newton-Raphson step with the analytical vega is offered as a faster
alternative once the bracket is small.
"""

import numpy as np

from .black_scholes import black_scholes_greeks, black_scholes_price

__all__ = ["implied_volatility"]


def implied_volatility(price, s, k, t, r, option="call", tol=1e-8,
                       max_iter=200):
    """Implied volatility by bisection, with a Newton polish step.

    Parameters
    ----------
    price : float
        Observed market price of the option.
    s, k, t, r : float
        Spot, strike, time to expiry (years), risk-free rate.
    option : {"call", "put"}

    Returns
    -------
    float
        The implied volatility.

    Raises
    ------
    ValueError
        If the price is outside the no-arbitrage bounds (below intrinsic
        value or above the maximum possible option value).
    """
    if t <= 0:
        raise ValueError("implied volatility requires t > 0")
    intrinsic = max(s - k, 0.0) if option == "call" else max(k - s, 0.0)
    max_value = s if option == "call" else k * np.exp(-r * t)
    if price < intrinsic - 1e-12 or price > max_value + 1e-12:
        raise ValueError(
            f"price {price} outside no-arbitrage bounds "
            f"[{intrinsic:.6f}, {max_value:.6f}]"
        )

    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mid_price = black_scholes_price(s, k, t, r, mid, option=option)
        if abs(mid_price - price) < tol:
            return float(mid)
        if mid_price < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break

    # Newton polish from the bisection estimate.
    sigma = 0.5 * (lo + hi)
    for _ in range(50):
        diff = black_scholes_price(s, k, t, r, sigma, option=option) - price
        if abs(diff) < tol:
            break
        vega = black_scholes_greeks(s, k, t, r, sigma, option=option)["vega"]
        if vega <= 0:
            break
        sigma = float(np.clip(sigma - diff / vega, 1e-6, 5.0))
    return float(sigma)
