"""Tests for Monte Carlo Greeks vs analytical Black-Scholes Greeks."""

import pytest

from monte_carlo_pricer.black_scholes import black_scholes_greeks
from monte_carlo_pricer.greeks import compare_greeks, mc_greeks

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_mc_delta_matches_analytic():
    comp = compare_greeks(S0, K, T, R, SIGMA, n_paths=200_000, option="call")
    assert comp["delta"][2] < 0.01  # |MC - analytic|


def test_mc_gamma_matches_analytic():
    comp = compare_greeks(S0, K, T, R, SIGMA, n_paths=200_000, option="call")
    assert comp["gamma"][2] < 0.002


def test_mc_vega_matches_analytic():
    comp = compare_greeks(S0, K, T, R, SIGMA, n_paths=200_000, option="call")
    assert comp["vega"][2] < 1.5


def test_mc_theta_and_rho_match_analytic():
    comp = compare_greeks(S0, K, T, R, SIGMA, n_paths=200_000, option="call")
    assert comp["theta"][2] < 0.5
    assert comp["rho"][2] < 2.0


def test_mc_put_delta_is_negative():
    g = mc_greeks(S0, K, T, R, SIGMA, n_paths=200_000, option="put")
    analytic = black_scholes_greeks(S0, K, T, R, SIGMA, "put")
    assert g["delta"] < 0
    assert g["delta"] == pytest.approx(analytic["delta"], abs=0.01)
