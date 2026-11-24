"""Pruebas del extracto listo para Tableau."""

import csv

import pytest

from monte_carlo_pricer import tableau_extract


def test_rejilla_36_filas_y_columnas(tmp_path):
    path = tmp_path / "grid.csv"
    n = tableau_extract.write_price_grid_csv(path)
    assert n == 36  # 9 valores de S0 x 4 de sigma
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 36
    assert list(rows[0].keys()) == [
        "s0", "strike", "t_anios", "r", "sigma",
        "bs_call", "bs_put", "delta_call", "vega_call",
    ]


def test_celda_central_es_el_call_conocido(tmp_path):
    # En el centro de la rejilla (S0 = K = 100, sigma = 0.20) el call
    # tiene que ser el de Black-Scholes que ya conocemos: 10.4506.
    path = tmp_path / "grid.csv"
    tableau_extract.write_price_grid_csv(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    centro = next(
        r for r in rows
        if float(r["s0"]) == 100.0 and float(r["sigma"]) == 0.20
    )
    assert float(centro["bs_call"]) == pytest.approx(10.4506, abs=1e-4)
    assert float(centro["delta_call"]) == pytest.approx(0.6368, abs=1e-4)
    # Y la paridad put-call tiene que cumplirse en cada fila:
    # call - put = S0 - K*exp(-rT).
    import math
    for r in rows:
        lhs = float(r["bs_call"]) - float(r["bs_put"])
        rhs = float(r["s0"]) - 100.0 * math.exp(-0.05 * 1.0)
        assert lhs == pytest.approx(rhs, abs=1e-4)
