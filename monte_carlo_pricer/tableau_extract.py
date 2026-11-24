"""
Layer: Queries & analysis — the price grid as a Tableau-ready CSV.
Extracto listo para Tableau: una rejilla de precios Black-Scholes.

Esto NO es "integración con Tableau" — no hay conector ni tablero
construido, y decir lo contrario sería mentir. Lo que hay es un CSV
ordenado (una fila por combinación de precio del subyacente y
volatilidad) que se importa en Tableau con dos clics, más la idea del
tablero en el README. La gracia de la rejilla es que deja ver de un
vistazo cómo se mueve el precio cuando se mueve el mercado: eso es lo
que un tablero debe mostrar, no una tabla de números sueltos.

Los precios son Black-Scholes (fórmula cerrada, instantánea), no Monte
Carlo: para una rejilla de 36 celdas no tiene caso pagar el costo de
simular cada una.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .black_scholes import black_scholes_greeks, black_scholes_price

__all__ = ["write_price_grid_csv"]


def write_price_grid_csv(path, k=100.0, t=1.0, r=0.05,
                         s0_values=None, sigma_values=(0.10, 0.20, 0.30, 0.40)):
    """Escribe la rejilla S0 x sigma con precios y Griegas de call.

    Columnas: s0, strike, t_anios, r, sigma, bs_call, bs_put,
    delta_call, vega_call. Una fila por cada (s0, sigma).
    """
    if s0_values is None:
        s0_values = [80.0, 85.0, 90.0, 95.0, 100.0,
                     105.0, 110.0, 115.0, 120.0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["s0", "strike", "t_anios", "r", "sigma",
                         "bs_call", "bs_put", "delta_call", "vega_call"])
        for s0 in s0_values:
            for sigma in sigma_values:
                call = black_scholes_price(s0, k, t, r, sigma, option="call")
                put = black_scholes_price(s0, k, t, r, sigma, option="put")
                greeks = black_scholes_greeks(s0, k, t, r, sigma, option="call")
                writer.writerow([s0, k, t, r, sigma, round(call, 6),
                                 round(put, 6), round(greeks["delta"], 6),
                                 round(greeks["vega"], 6)])
                rows += 1
    return rows
