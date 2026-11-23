"""Tests for the closed-form Black-Scholes formulas."""

import numpy as np
import pytest

from monte_carlo_pricer.black_scholes import (
    black_scholes_greeks,
    black_scholes_price,
)

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_known_call_and_put_prices():
    # Textbook values for S=K=100, T=1, r=5%, sigma=20%.
    assert black_scholes_price(S0, K, T, R, SIGMA, "call") == pytest.approx(10.4506, abs=1e-3)
    assert black_scholes_price(S0, K, T, R, SIGMA, "put") == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    call = black_scholes_price(S0, K, T, R, SIGMA, "call")
    put = black_scholes_price(S0, K, T, R, SIGMA, "put")
    assert call - put == pytest.approx(S0 - K * np.exp(-R * T), abs=1e-10)


def test_expiry_is_intrinsic_value():
    assert black_scholes_price(105, 100, 0.0, R, SIGMA, "call") == pytest.approx(5.0)
    assert black_scholes_price(95, 100, 0.0, R, SIGMA, "put") == pytest.approx(5.0)
    assert black_scholes_price(95, 100, 0.0, R, SIGMA, "call") == pytest.approx(0.0)


def test_zero_volatility_is_discounted_intrinsic():
    # With sigma = 0 the forward is deterministic: S_T = S_0 e^{rT}.
    price = black_scholes_price(S0, K, T, R, 0.0, "call")
    expected = np.exp(-R * T) * max(S0 * np.exp(R * T) - K, 0.0)
    assert price == pytest.approx(expected, abs=1e-10)


def test_known_analytical_greeks_call():
    g = black_scholes_greeks(S0, K, T, R, SIGMA, "call")
    assert g["delta"] == pytest.approx(0.6368, abs=1e-3)
    assert g["gamma"] == pytest.approx(0.0188, abs=1e-3)
    assert g["vega"] == pytest.approx(37.52, abs=0.05)
    assert g["theta"] == pytest.approx(-6.414, abs=0.01)
    assert g["rho"] == pytest.approx(53.23, abs=0.05)


def test_put_delta_is_call_delta_minus_one():
    call = black_scholes_greeks(S0, K, T, R, SIGMA, "call")
    put = black_scholes_greeks(S0, K, T, R, SIGMA, "put")
    assert put["delta"] == pytest.approx(call["delta"] - 1.0, abs=1e-12)
    assert put["gamma"] == pytest.approx(call["gamma"], abs=1e-12)
    assert put["vega"] == pytest.approx(call["vega"], abs=1e-12)


def test_invalid_option_raises():
    with pytest.raises(ValueError):
        black_scholes_price(S0, K, T, R, SIGMA, "straddle")
    with pytest.raises(ValueError):
        black_scholes_greeks(S0, K, 0.0, R, SIGMA, "call")
