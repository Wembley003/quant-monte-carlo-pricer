"""Monte Carlo option pricer built from scratch.

Modules
-------
black_scholes
    Closed-form Black-Scholes prices and analytical Greeks.
stochastic_processes
    Simulation of stochastic processes (GBM, Ornstein-Uhlenbeck) with
    exact solutions, Euler-Maruyama and Milstein discretizations, and
    analytical moments for validation.
monte_carlo
    Monte Carlo pricing of European options under the risk-neutral measure.
greeks
    Greeks by finite differences (bump-and-revalue with common random
    numbers) and by the analytical Black-Scholes formulas.
implied_vol
    Implied volatility by inversion of the Black-Scholes formula.
"""

from .black_scholes import black_scholes_price, black_scholes_greeks
from .monte_carlo import mc_european_price
from .implied_vol import implied_volatility

__all__ = [
    "black_scholes_price",
    "black_scholes_greeks",
    "mc_european_price",
    "implied_volatility",
]
