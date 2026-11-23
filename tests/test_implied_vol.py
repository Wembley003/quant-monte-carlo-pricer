"""Tests for implied-volatility inversion."""

import pytest

from monte_carlo_pricer.black_scholes import black_scholes_price
from monte_carlo_pricer.implied_vol import implied_volatility

S0, K, T, R = 100.0, 100.0, 1.0, 0.05


def test_iv_round_trip_call():
    sigma = 0.25
    price = black_scholes_price(S0, K, T, R, sigma, "call")
    assert implied_volatility(price, S0, K, T, R, "call") == pytest.approx(sigma, abs=1e-6)


def test_iv_round_trip_put():
    sigma = 0.35
    price = black_scholes_price(S0, K, T, R, sigma, "put")
    assert implied_volatility(price, S0, K, T, R, "put") == pytest.approx(sigma, abs=1e-6)


def test_iv_recovers_known_volatility():
    price = black_scholes_price(S0, K, T, R, 0.20, "call")
    assert implied_volatility(price, S0, K, T, R, "call") == pytest.approx(0.20, abs=1e-6)


def test_iv_out_of_bounds_raises():
    # Put with S=90, K=100: intrinsic value is 10, so a price of 5 is impossible.
    with pytest.raises(ValueError):
        implied_volatility(5.0, 90.0, 100.0, T, R, "put")
    # Call price above the maximum possible value (the spot itself).
    with pytest.raises(ValueError):
        implied_volatility(200.0, S0, K, T, R, "call")


def test_iv_deep_in_the_money():
    sigma = 0.30
    price = black_scholes_price(150.0, 100.0, T, R, sigma, "call")
    assert implied_volatility(price, 150.0, 100.0, T, R, "call") == pytest.approx(sigma, abs=1e-6)
