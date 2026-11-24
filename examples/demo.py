"""
Layer: Client (presentation) — the demo the user runs to see the pricer work.
Demo: Monte Carlo vs Black-Scholes, Greeks, implied vol, and the
stochastic-processes module (GBM schemes and Ornstein-Uhlenbeck).

Run from the repository root:

    python examples/demo.py

It prints the validation tables and saves a convergence chart to
``results/mc_convergence.png``.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monte_carlo_pricer.black_scholes import black_scholes_price
from monte_carlo_pricer.greeks import compare_greeks
from monte_carlo_pricer.implied_vol import implied_volatility
from monte_carlo_pricer.monte_carlo import mc_vs_bs_error
from monte_carlo_pricer.stochastic_processes import (
    gbm_moments,
    ou_moments,
    ou_stationary_moments,
    simulate_gbm_euler_maruyama,
    simulate_gbm_exact,
    simulate_gbm_milstein,
    simulate_ou_exact,
)

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    section("1. Monte Carlo vs Black-Scholes (European call and put)")
    for option in ("call", "put"):
        mc, bs, err, se = mc_vs_bs_error(S0, K, T, R, SIGMA, n_paths=1_000_000,
                                         option=option, seed=42)
        print(f"{option:>4}: MC = {mc:.4f}  (SE {se:.4f})   "
              f"BS = {bs:.4f}   error = {err:+.4f}  ({abs(err) / se:.2f} SE)")

    section("2. Convergence: Monte Carlo error vs number of paths (call)")
    n_paths_list = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    bs_call = black_scholes_price(S0, K, T, R, SIGMA, "call")
    errors, ses = [], []
    for n in n_paths_list:
        mc, _, err, se = mc_vs_bs_error(S0, K, T, R, SIGMA, n_paths=n,
                                        option="call", seed=42)
        errors.append(abs(err))
        ses.append(se)
        print(f"n = {n:>8,}: MC = {mc:.4f}   |error| = {abs(err):.4f}   SE = {se:.4f}")
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots()
    ax.loglog(n_paths_list, errors, "o-", label="|MC - BS|")
    ax.loglog(n_paths_list, ses, "s--", label="standard error")
    ax.loglog(n_paths_list, [1.0 / np.sqrt(n) for n in n_paths_list], ":",
              label="1/sqrt(n) reference")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Absolute error")
    ax.set_title("Monte Carlo convergence (European call, S=K=100, T=1, r=5%, vol=20%)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/mc_convergence.png", dpi=150)
    print("Chart saved to results/mc_convergence.png")

    section("3. Greeks: Monte Carlo (finite differences) vs analytical")
    comp = compare_greeks(S0, K, T, R, SIGMA, n_paths=500_000, option="call")
    print(f"{'Greek':>6} {'MC':>12} {'Analytical':>12} {'|error|':>10}")
    for g, (mc_v, an_v, err) in comp.items():
        print(f"{g:>6} {mc_v:12.4f} {an_v:12.4f} {err:10.4f}")

    section("4. Implied volatility round-trip")
    for sigma_true in (0.15, 0.20, 0.35):
        price = black_scholes_price(S0, K, T, R, sigma_true, "call")
        iv = implied_volatility(price, S0, K, T, R, "call")
        print(f"true vol = {sigma_true:.2f} -> BS price = {price:.4f} "
              f"-> implied vol = {iv:.6f}")

    section("5. Stochastic processes: GBM schemes (strong error, same Brownian path)")
    s0, mu, sigma, t = 100.0, 0.05, 0.30, 1.0
    n_steps, n_paths, seed = 20, 5_000, 7
    dt = t / n_steps
    z = np.random.default_rng(seed).standard_normal((n_steps, n_paths))
    exact_terminal = s0 * np.exp(
        (mu - 0.5 * sigma**2) * t + sigma * np.sqrt(dt) * z.sum(axis=0)
    )
    _, euler = simulate_gbm_euler_maruyama(s0, mu, sigma, t, n_steps, n_paths, seed=seed)
    _, milstein = simulate_gbm_milstein(s0, mu, sigma, t, n_steps, n_paths, seed=seed)
    rmse_euler = float(np.sqrt(np.mean((euler[-1] - exact_terminal) ** 2)))
    rmse_milstein = float(np.sqrt(np.mean((milstein[-1] - exact_terminal) ** 2)))
    print(f"dt = {dt:.3f}, {n_paths:,} paths")
    print(f"Euler-Maruyama RMSE = {rmse_euler:.4f}")
    print(f"Milstein        RMSE = {rmse_milstein:.4f} "
          f"({rmse_milstein / rmse_euler:.1%} of Euler — strong order 1 vs 1/2)")

    section("6. Stochastic processes: GBM and OU moments vs analytical")
    mean, var = gbm_moments(100.0, 0.08, 0.30, 1.0)
    _, paths = simulate_gbm_exact(100.0, 0.08, 0.30, 1.0, n_steps=252,
                                  n_paths=100_000, seed=123)
    print(f"GBM  S_T: simulated mean = {paths[-1].mean():.3f} (analytical {mean:.3f}), "
          f"var = {paths[-1].var(ddof=1):.1f} (analytical {var:.1f})")
    x0, theta, mu_ou, sigma_ou, t_ou = 0.0, 1.5, 2.0, 0.4, 2.0
    mean_ou, var_ou = ou_moments(x0, theta, mu_ou, sigma_ou, t_ou)
    _, ou_paths = simulate_ou_exact(x0, theta, mu_ou, sigma_ou, t_ou,
                                    n_steps=252, n_paths=100_000, seed=123)
    print(f"OU   X_T: simulated mean = {ou_paths[-1].mean():.4f} (analytical {mean_ou:.4f}), "
          f"var = {ou_paths[-1].var(ddof=1):.4f} (analytical {var_ou:.4f})")
    stat_mean, stat_var = ou_stationary_moments(theta, mu_ou, sigma_ou)
    _, ou_long = simulate_ou_exact(0.0, theta, mu_ou, sigma_ou, t=20.0,
                                   n_steps=500, n_paths=50_000, seed=123)
    print(f"OU stationary: simulated mean = {ou_long[-1].mean():.4f} "
          f"(analytical {stat_mean:.4f}), var = {ou_long[-1].var(ddof=1):.4f} "
          f"(analytical {stat_var:.4f})")


if __name__ == "__main__":
    main()
