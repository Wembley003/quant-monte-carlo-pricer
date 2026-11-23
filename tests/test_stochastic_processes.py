"""Tests for the stochastic-processes module.

Every process is validated against its analytical result:
GBM terminal mean/variance and the Ito-corrected log drift, and OU
terminal moments plus the stationary distribution N(mu, sigma^2 / 2theta).
The Milstein scheme is checked to beat Euler-Maruyama in strong error
with the same Brownian increments (common random numbers).
"""

import numpy as np
import pytest

from monte_carlo_pricer.stochastic_processes import (
    gbm_moments,
    ito_log_drift,
    ou_moments,
    ou_stationary_moments,
    simulate_gbm_euler_maruyama,
    simulate_gbm_exact,
    simulate_gbm_milstein,
    simulate_ou_euler_maruyama,
    simulate_ou_exact,
)


# ---------------------------------------------------------------------------
# Geometric Brownian motion
# ---------------------------------------------------------------------------


def test_gbm_exact_terminal_moments_match_analytic():
    s0, mu, sigma, t = 100.0, 0.08, 0.30, 1.0
    _, paths = simulate_gbm_exact(s0, mu, sigma, t, n_steps=252,
                                  n_paths=100_000, seed=123)
    terminal = paths[-1]
    mean, var = gbm_moments(s0, mu, sigma, t)
    # Monte Carlo SE of the mean is ~0.10; 0.5 is ~5 SE.
    assert terminal.mean() == pytest.approx(mean, abs=0.5)
    assert terminal.var(ddof=1) == pytest.approx(var, rel=0.03)


def test_gbm_ito_log_drift():
    # E[ln S_T] = ln S_0 + (mu - sigma^2 / 2) T (Ito correction).
    s0, mu, sigma, t = 100.0, 0.08, 0.30, 1.0
    _, paths = simulate_gbm_exact(s0, mu, sigma, t, n_steps=252,
                                  n_paths=100_000, seed=123)
    expected = np.log(s0) + ito_log_drift(mu, sigma) * t
    assert np.log(paths[-1]).mean() == pytest.approx(expected, abs=0.005)
    assert ito_log_drift(mu, sigma) == pytest.approx(0.08 - 0.045)


def test_gbm_euler_maruyama_converges_to_analytic_moments():
    s0, mu, sigma, t = 100.0, 0.08, 0.30, 1.0
    _, paths = simulate_gbm_euler_maruyama(s0, mu, sigma, t, n_steps=1000,
                                           n_paths=100_000, seed=123)
    mean, var = gbm_moments(s0, mu, sigma, t)
    assert paths[-1].mean() == pytest.approx(mean, rel=0.01)
    assert paths[-1].var(ddof=1) == pytest.approx(var, rel=0.05)


def test_gbm_milstein_strong_error_beats_euler():
    # Same Brownian increments for all three (same seed, same draw order):
    # the exact terminal value is S_0 exp((mu - sigma^2/2) T + sigma sqrt(dt) sum Z).
    s0, mu, sigma, t = 100.0, 0.05, 0.30, 1.0
    n_steps, n_paths, seed = 20, 5_000, 7
    dt = t / n_steps
    z = np.random.default_rng(seed).standard_normal((n_steps, n_paths))
    exact_terminal = s0 * np.exp(
        ito_log_drift(mu, sigma) * t + sigma * np.sqrt(dt) * z.sum(axis=0)
    )
    _, euler = simulate_gbm_euler_maruyama(s0, mu, sigma, t, n_steps, n_paths, seed=seed)
    _, milstein = simulate_gbm_milstein(s0, mu, sigma, t, n_steps, n_paths, seed=seed)
    rmse_euler = float(np.sqrt(np.mean((euler[-1] - exact_terminal) ** 2)))
    rmse_milstein = float(np.sqrt(np.mean((milstein[-1] - exact_terminal) ** 2)))
    assert rmse_milstein < rmse_euler


def test_gbm_paths_shape_and_start():
    times, paths = simulate_gbm_exact(100.0, 0.05, 0.2, 1.0, n_steps=10,
                                      n_paths=50, seed=1)
    assert times.shape == (11,)
    assert paths.shape == (11, 50)
    assert np.all(paths[0] == 100.0)
    assert np.all(paths > 0)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck
# ---------------------------------------------------------------------------


def test_ou_exact_terminal_moments_match_analytic():
    x0, theta, mu, sigma, t = 0.0, 1.5, 2.0, 0.4, 2.0
    _, paths = simulate_ou_exact(x0, theta, mu, sigma, t, n_steps=252,
                                 n_paths=100_000, seed=123)
    mean, var = ou_moments(x0, theta, mu, sigma, t)
    assert paths[-1].mean() == pytest.approx(mean, abs=0.005)
    assert paths[-1].var(ddof=1) == pytest.approx(var, rel=0.02)


def test_ou_stationary_distribution():
    theta, mu, sigma = 1.5, 2.0, 0.4
    _, paths = simulate_ou_exact(0.0, theta, mu, sigma, t=20.0, n_steps=500,
                                 n_paths=50_000, seed=123)
    stat_mean, stat_var = ou_stationary_moments(theta, mu, sigma)
    assert paths[-1].mean() == pytest.approx(stat_mean, abs=0.01)
    assert paths[-1].var(ddof=1) == pytest.approx(stat_var, rel=0.03)


def test_ou_euler_maruyama_converges_to_analytic_moments():
    x0, theta, mu, sigma, t = 0.0, 1.5, 2.0, 0.4, 2.0
    _, paths = simulate_ou_euler_maruyama(x0, theta, mu, sigma, t,
                                          n_steps=1000, n_paths=50_000,
                                          seed=123)
    mean, var = ou_moments(x0, theta, mu, sigma, t)
    assert paths[-1].mean() == pytest.approx(mean, abs=0.02)
    assert paths[-1].var(ddof=1) == pytest.approx(var, rel=0.05)


def test_ou_mean_reverts_towards_mu():
    # Starting far above mu, the average path must drift down towards mu.
    _, paths = simulate_ou_exact(10.0, 2.0, 0.0, 0.1, t=3.0, n_steps=300,
                                 n_paths=50_000, seed=5)
    assert paths[-1].mean() == pytest.approx(0.0, abs=0.05)
    assert paths[len(paths) // 2].mean() < 10.0
