"""Pruebas de la persistencia en SQLite.

La idea es simple: si guardo una corrida, la tengo que poder leer
después con los mismos números. Y si guardo dos, la segunda no puede
pisar a la primera. El precio de Monte Carlo se compara contra el de
Black-Scholes con su propio error estándar como vara de medir — si se
aleja más de 3 errores estándar, algo anda mal en la simulación, no
en la base de datos.
"""

import pytest

from monte_carlo_pricer import persistence

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_record_mc_run_guarda_y_lee(tmp_path):
    db = tmp_path / "runs.db"
    run_id = persistence.record_mc_run(
        db, S0, K, T, R, SIGMA, n_paths=20_000, seed=42
    )
    assert run_id == 1
    run = persistence.get_run(db, run_id)
    assert run is not None
    assert run["s0"] == S0 and run["k"] == K and run["sigma"] == SIGMA
    assert run["option"] == "call"
    # El precio de Monte Carlo tiene que estar cerca del exacto, medido
    # en sus propios errores estándar.
    assert abs(run["mc_price"] - run["bs_price"]) < 3 * run["mc_se"]
    assert run["bs_price"] == pytest.approx(10.4506, abs=1e-3)
    # Las cinco Griegas, cada una con su valor MC y su valor exacto.
    assert {g["name"] for g in run["greeks"]} == {
        "delta", "gamma", "vega", "theta", "rho",
    }
    delta = next(g for g in run["greeks"] if g["name"] == "delta")
    assert delta["bs_value"] == pytest.approx(0.6368, abs=1e-3)


def test_list_runs_ordena_recientes_primero(tmp_path):
    db = tmp_path / "runs.db"
    id1 = persistence.record_mc_run(db, S0, K, T, R, SIGMA, n_paths=5_000, seed=1)
    id2 = persistence.record_mc_run(db, S0, 110.0, T, R, SIGMA, n_paths=5_000, seed=2)
    assert id2 == id1 + 1
    runs = persistence.list_runs(db)
    assert [r["id"] for r in runs] == [id2, id1]
    assert runs[0]["k"] == 110.0


def test_get_run_inexistente_devuelve_none(tmp_path):
    db = tmp_path / "runs.db"
    persistence.init_db(db)
    assert persistence.get_run(db, 99) is None


def test_store_run_sin_griegas(tmp_path):
    db = tmp_path / "runs.db"
    run_id = persistence.store_run(
        db, s0=S0, k=K, t=T, r=R, sigma=SIGMA, n_paths=1_000,
        option="put", seed=None, mc_price=5.5, mc_se=0.05, bs_price=5.57,
    )
    run = persistence.get_run(db, run_id)
    assert run["option"] == "put"
    assert run["seed"] is None
    assert run["greeks"] == []
