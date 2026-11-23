"""Tests for the Monte Carlo pricer against Black-Scholes."""

import numpy as np
import pytest

from monte_carlo_pricer.black_scholes import black_scholes_price
from monte_carlo_pricer.monte_carlo import mc_european_price, mc_vs_bs_error

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_mc_call_matches_black_scholes_within_error():
    mc_price, bs_price, error, se = mc_vs_bs_error(
        S0, K, T, R, SIGMA, n_paths=200_000, option="call", seed=42
    )
    assert abs(error) < max(3 * se, 0.05)
    assert se < 0.05


def test_mc_put_matches_black_scholes_within_error():
    mc_price, bs_price, error, se = mc_vs_bs_error(
        S0, K, T, R, SIGMA, n_paths=200_000, option="put", seed=42
    )
    assert abs(error) < max(3 * se, 0.05)


def test_mc_put_call_parity():
    call, _ = mc_european_price(S0, K, T, R, SIGMA, n_paths=200_000,
                                option="call", seed=42)
    put, _ = mc_european_price(S0, K, T, R, SIGMA, n_paths=200_000,
                               option="put", seed=42)
    assert call - put == pytest.approx(S0 - K * np.exp(-R * T), abs=0.1)


def test_mc_antithetic_reduces_standard_error():
    _, se_plain = mc_european_price(S0, K, T, R, SIGMA, n_paths=100_000,
                                    option="call", seed=1, antithetic=False)
    _, se_anti = mc_european_price(S0, K, T, R, SIGMA, n_paths=100_000,
                                   option="call", seed=1, antithetic=True)
    assert se_anti < se_plain


def test_mc_deep_out_of_the_money_is_small():
    price, se = mc_european_price(100.0, 200.0, T, R, SIGMA, n_paths=200_000,
                                  option="call", seed=42)
    bs = black_scholes_price(100.0, 200.0, T, R, SIGMA, "call")
    assert price == pytest.approx(bs, abs=max(3 * se, 0.02))
    assert price < 0.1


def test_mc_expiry_is_intrinsic():
    price, se = mc_european_price(105.0, 100.0, 0.0, R, SIGMA, option="call")
    assert price == pytest.approx(5.0)
    assert se == 0.0


def test_mc_invalid_option_raises():
    with pytest.raises(ValueError):
        mc_european_price(S0, K, T, R, SIGMA, option="straddle")
