# Monte Carlo Option Pricer

A European option pricer built from scratch in Python (NumPy + SciPy for the
normal CDF only — no pricing library). It prices with Monte Carlo simulation
and validates every result against the closed-form Black-Scholes solution,
including Greeks and implied volatility. A dedicated module covers the
stochastic processes underneath: geometric Brownian motion (GBM), Itô's
lemma, Euler–Maruyama and Milstein discretizations, and the
Ornstein–Uhlenbeck (OU) mean-reversion process.

## What it demonstrates

- **Stochastic processes** (`monte_carlo_pricer/stochastic_processes.py`):
  GBM with its exact solution, Euler–Maruyama (strong order 1/2) and
  Milstein (strong order 1) schemes; OU with its exact Gaussian transition
  and Euler–Maruyama; analytical moments for every process, used to
  validate the simulations.
- **Monte Carlo pricing**: GBM under the risk-neutral measure, antithetic
  variates, and a standard error reported with every price.
- **Greeks**: delta, gamma, vega, theta, rho — analytical (Black-Scholes)
  and by finite differences with common random numbers.
- **Implied volatility**: inversion of Black-Scholes by bisection with a
  Newton polish step.
- **Validation**: the Monte Carlo price is checked against Black-Scholes
  within a documented tolerance, and every process is checked against its
  analytical distribution.

## Theory in brief

**GBM and Itô's lemma.** The underlying follows
dS_t = μ S_t dt + σ S_t dW_t. By Itô's lemma, for Y_t = ln S_t:

dY_t = (μ − σ²/2) dt + σ dW_t,

so S_T = S_0 exp((μ − σ²/2) T + σ W_T). The −σ²/2 term is the Itô
correction: the median of S_T grows at μ − σ²/2, not at μ. Moments:
E[S_T] = S_0 e^{μT}, Var(S_T) = S_0² e^{2μT} (e^{σ²T} − 1).

**Discretization.** Euler–Maruyama has strong order 1/2; Milstein adds
the Itô–Taylor term ½ b b′ (dt Z² − dt) and reaches strong order 1. For
GBM the correction is ½σ²S_n (dt Z² − dt); for OU the diffusion is
constant, so Milstein coincides with Euler–Maruyama.

**Ornstein–Uhlenbeck (mean reversion).** dX_t = θ(μ − X_t) dt + σ dW_t.
E[X_t] = X_0 e^{−θt} + μ(1 − e^{−θt}),
Var(X_t) = σ²/(2θ) · (1 − e^{−2θt}), and the stationary distribution is
N(μ, σ²/(2θ)). OU is the standard model for mean-reverting quantities
(rates in Vasicek, spreads, volatility).

**Black-Scholes.** The Monte Carlo price is the discounted expected
payoff under the risk-neutral GBM (drift r). Greeks by finite differences
reuse the same Brownian draws (common random numbers) so the differences
are not drowned in independent Monte Carlo noise.

## Requirements

- Python 3.10 or higher.

## Installation

```bash
git clone https://github.com/Wembley003/quant-monte-carlo-pricer.git
cd quant-monte-carlo-pricer
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Demo

```bash
python examples/demo.py
```

It prints the validation tables and saves a convergence chart to
`results/mc_convergence.png`.

## Results (run 23 Nov 2026)

European options, S = K = 100, T = 1 year, r = 5%, σ = 20%,
1,000,000 paths with antithetic variates, seed 42:

| Option | Monte Carlo | Std. error | Black-Scholes | Error | Error / SE |
|--------|-------------|------------|---------------|-------|------------|
| Call   | 10.4565     | 0.0147     | 10.4506       | +0.0060 | 0.40 |
| Put    | 5.5762      | 0.0087     | 5.5735        | +0.0027 | 0.31 |

The Monte Carlo price agrees with Black-Scholes to well within one
standard error. The error shrinks like 1/√n (see
`results/mc_convergence.png`).

Greeks (call, same parameters; Monte Carlo with 500,000 paths, finite
differences with common random numbers):

| Greek | Monte Carlo | Analytical | \|Error\| |
|-------|-------------|------------|---------|
| Delta | 0.6368      | 0.6368     | 0.0000 |
| Gamma | 0.0186      | 0.0188     | 0.0001 |
| Vega  | 37.57       | 37.52      | 0.04 |
| Theta | −6.42       | −6.41      | 0.01 |
| Rho   | 53.23       | 53.23      | 0.00 |

Implied volatility round-trips exactly: a call priced at σ = 0.15, 0.20
and 0.35 inverts back to the same volatility to 6 decimal places.

Stochastic processes (seed 7; 5,000 paths; dt = 0.05):

| Scheme | Strong RMSE vs exact GBM terminal value |
|--------|------------------------------------------|
| Euler–Maruyama | 1.60 |
| Milstein       | 0.11 (6.5% of Euler) |

GBM terminal moments (100,000 paths): simulated mean 108.45 vs
analytical 108.33; variance 1,115 vs 1,105. OU terminal moments
(100,000 paths): simulated mean 1.9010 vs analytical 1.9004; variance
0.0532 vs 0.0532. OU stationary distribution (T = 20): mean 2.0009 vs
2.0000; variance 0.0533 vs 0.0533.

## Tests

```bash
pytest
```

49 tests, all passing. They check Black-Scholes against textbook values
and put–call parity, every process against its analytical moments and
stationary distribution, Milstein beating Euler–Maruyama in strong error
with identical Brownian increments, Monte Carlo prices against
Black-Scholes within 3 standard errors, Monte Carlo Greeks against the
analytical ones, and implied-volatility round-trips. The layer tests in
`tests/test_layers.py` check the full chain (inputs → simulation →
stored results) and that the layers do not mix: the business rules run
with openpyxl and sqlite3 blocked.

```bash
./run_tests.sh   # lo mismo, buscando el venv solo
```

## Layers

Three layers; Bronze/Silver/Gold does **not** apply here (there is no
data pipeline with raw → clean → aggregate staging — the inputs are
scalar parameters and there is nothing to clean, so forcing it would
be labeling, not architecture):

- **Client (presentation)** — `examples/demo.py`: the demo the user
  runs. It never touches storage directly.
- **Business rules** — `black_scholes.py`, `stochastic_processes.py`,
  `monte_carlo.py`, `greeks.py`, `implied_vol.py`: the models, the
  simulation and the Greeks. Pure math: no sqlite3, no openpyxl, no
  calls to the client or to storage.
- **Queries & analysis** — `persistence.py`, `excel_export.py`,
  `tableau_extract.py`, `pivot_table.py` (plus `sql/schema_access.sql`
  and `access/`): the Monte Carlo/Greeks results, saved in SQLite and
  queried/exported. The analysis here is Python, not SQL: comparing
  and storing the simulated results.

## Guardar corridas: SQLite + Excel (nuevo)

Antes cada simulación salía por pantalla y se perdía. Ahora queda
guardada con sus parámetros, su precio de Monte Carlo, su error
estándar, el precio exacto de Black-Scholes y las cinco Griegas, en
una base SQLite (`persistence.py`), y se puede exportar a Excel
(`excel_export.py`) con tres hojas: las corridas, las Griegas y un
resumen con el error medio entre Monte Carlo y Black-Scholes. Si ese
error medio es mucho mayor que el error estándar típico, la simulación
tiene un problema.

```python
from monte_carlo_pricer import persistence, excel_export

run_id = persistence.record_mc_run("runs.db", 100, 100, 1.0, 0.05, 0.20)
excel_export.export_workbook("runs.db", "runs.xlsx")
```

## Tablas dinámicas y Access

El Excel exportado trae dos tablas dinámicas nativas, no valores
pegados: "Dinámica por opción" agrupa las corridas por call/put con
el precio MC promedio, el precio Black-Scholes promedio y el error
estándar promedio; "Dinámica por Griega" agrupa por Griega con el
error absoluto promedio y el valor MC promedio. Son objetos
pivotTable reales (definición + caché + registros dentro del .xlsx),
así que Excel las recalcula al abrir y se pueden filtrar y reordenar
como cualquier tabla dinámica. Si la base está vacía, la exportación
sigue funcionando sin pivotes.

En `access/` está además la base real de Access
(`monte-carlo-pricer.accdb`, formato ACE14): tablas `runs` y `greeks`
con sus filas, construida con Jackcess desde SQLite y verificada con
mdbtools. Se regenera con `bash access/build_access.sh runs.db
monte-carlo-pricer.accdb` (ver `access/README.md`). Los CSV de
`access/csv/` y el esquema son el pack de importación en texto.

## Tableau y Access: extractos, no integración (nuevo)

Sin mentir: no hay conector con Tableau. Lo que hay es un CSV
ordenado (`tableau_extract.write_price_grid_csv`) con una rejilla de
precios Black-Scholes — S0 de 80 a 120 contra volatilidad de 10% a
40% — que se importa en Tableau tal cual. Para Access ya no hace
falta el esquema SQL de `sql/schema_access.sql`: la base `.accdb`
real está en `access/` (ver la sección anterior). La idea del
tablero: precio del call contra S0, con una línea por volatilidad, y
un mapa de calor de la delta. Eso deja ver de un vistazo cómo
responde el precio cuando se mueve el mercado, que es justo lo que
un tablero debe mostrar.

## Structure

```text
quant-monte-carlo-pricer/
├── monte_carlo_pricer/
│   ├── __init__.py
│   ├── black_scholes.py         # closed-form prices and analytical Greeks
│   ├── stochastic_processes.py  # GBM (exact/Euler/Milstein), OU, Ito
│   ├── monte_carlo.py           # Monte Carlo pricing, antithetic variates
│   ├── greeks.py                # finite-difference Greeks (common random numbers)
│   ├── implied_vol.py           # implied volatility by bisection + Newton
│   ├── persistence.py           # SQLite: guarda corridas y Griegas
│   ├── excel_export.py          # exporta la base a Excel (3 hojas + 2 pivotes)
│   ├── pivot_table.py           # tablas dinámicas nativas de Excel
│   └── tableau_extract.py       # CSV rejilla S0 x sigma, listo para Tableau
├── access/                       # .accdb real + CSV/schema + constructor Java
│   ├── monte-carlo-pricer.accdb
│   ├── dump_access.py
│   ├── AccessBuilder.java
│   └── build_access.sh
├── sql/
│   └── schema_access.sql        # DDL para Access (solo esquema, sin probar en Access)
├── tests/
│   ├── test_black_scholes.py
│   ├── test_stochastic_processes.py
│   ├── test_monte_carlo.py
│   ├── test_greeks.py
│   ├── test_implied_vol.py
│   ├── test_persistence.py
│   ├── test_excel_export.py
│   ├── test_access.py
│   └── test_tableau_extract.py
├── run_tests.sh                 # corre pytest buscando el venv
├── examples/
│   └── demo.py                  # full validation run
├── results/
│   └── mc_convergence.png       # generated by the demo
├── requirements.txt
├── LICENSE
└── README.md
```

## Assumptions and limitations

- European exercise only (no American / Bermudan options).
- Constant volatility and interest rate; no dividends; no jumps.
- GBM only for pricing — the OU process is included as a stochastic
  process with its own validation, not as a pricing underlying.
- Monte Carlo Greeks use finite differences; for production use,
  pathwise or likelihood-ratio estimators would be more efficient.

## Author

Wembley Emmanuel Ibarra Tepato — BSc in Physics (UAM Iztapalapa),
MSc in Physics candidate (UAM), BSc in Mathematics student (UNAM),
Balance Risk Specialist at Bancomext.

## License

MIT. See `LICENSE`.
