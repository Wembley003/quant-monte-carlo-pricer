"""
Layer: Queries & analysis — runs and Greeks saved here to be queried later.
Guardar corridas del pricer en SQLite.

El problema que resuelve es aburrido y por eso importante: cada vez
que corro una simulación de Monte Carlo, el número sale por pantalla
y se pierde. Si una semana después quiero comparar cuánto cambió el
precio al mover la volatilidad, tengo que volver a correr todo y rezar
porque haya usado la misma semilla. Con esto, cada corrida queda
guardada con sus parámetros, su precio de Monte Carlo, su error
estándar, el precio de Black-Scholes y las Griegas — y se pueden
comparar con una consulta SQL en vez de con la memoria.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .black_scholes import black_scholes_greeks, black_scholes_price
from .greeks import compare_greeks
from .monte_carlo import mc_european_price

__all__ = [
    "init_db",
    "store_run",
    "record_mc_run",
    "list_runs",
    "get_run",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    s0           REAL NOT NULL,
    k            REAL NOT NULL,
    t            REAL NOT NULL,
    r            REAL NOT NULL,
    sigma        REAL NOT NULL,
    n_paths      INTEGER NOT NULL,
    option       TEXT NOT NULL,
    seed         INTEGER,
    mc_price     REAL NOT NULL,
    mc_se        REAL NOT NULL,
    bs_price     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS greeks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES runs (id),
    name      TEXT NOT NULL,
    mc_value  REAL NOT NULL,
    bs_value  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_greeks_run ON greeks (run_id);
"""


def _connect(db_path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path):
    """Crea las tablas runs y greeks si no existen."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


def store_run(db_path, *, s0, k, t, r, sigma, n_paths, option, seed,
              mc_price, mc_se, bs_price, greeks=None):
    """Guarda una corrida ya calculada y devuelve su id.

    ``greeks`` es un dict nombre -> (valor_mc, valor_bs), como el que
    devuelve ``compare_greeks``. Si no se pasan Griegas, se guarda solo
    la corrida.
    """
    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO runs
               (created_at, s0, k, t, r, sigma, n_paths, option, seed,
                mc_price, mc_se, bs_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (created_at, s0, k, t, r, sigma, n_paths, option, seed,
             mc_price, mc_se, bs_price),
        )
        run_id = cur.lastrowid
        for name, (mc_value, bs_value) in (greeks or {}).items():
            conn.execute(
                "INSERT INTO greeks (run_id, name, mc_value, bs_value)"
                " VALUES (?, ?, ?, ?)",
                (run_id, name, mc_value, bs_value),
            )
    return run_id


def record_mc_run(db_path, s0, k, t, r, sigma, n_paths=100_000,
                  option="call", seed=42):
    """Corre la simulación y la guarda en la base, todo en un paso.

    Calcula el precio de Monte Carlo con su error estándar, el precio
    exacto de Black-Scholes y las cinco Griegas por los dos métodos
    (Monte Carlo contra analítico). Es el atajo para cuando uno quiere
    el número Y el registro, sin andar pasando valores a mano entre
    funciones.
    """
    mc_price, mc_se = mc_european_price(
        s0, k, t, r, sigma, n_paths=n_paths, option=option, seed=seed
    )
    bs_price = black_scholes_price(s0, k, t, r, sigma, option=option)
    compared = compare_greeks(
        s0, k, t, r, sigma, n_paths=n_paths, option=option, seed=seed
    )
    greeks = {name: (mc, bs) for name, (mc, bs, _err) in compared.items()}
    return store_run(
        db_path,
        s0=s0, k=k, t=t, r=r, sigma=sigma, n_paths=n_paths,
        option=option, seed=seed, mc_price=mc_price, mc_se=mc_se,
        bs_price=bs_price, greeks=greeks,
    )


def list_runs(db_path):
    """Todas las corridas guardadas, de la más reciente a la más vieja."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_run(db_path, run_id):
    """Una corrida con sus Griegas, o None si no existe."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        run = dict(row)
        run["greeks"] = [
            dict(g)
            for g in conn.execute(
                "SELECT name, mc_value, bs_value FROM greeks"
                " WHERE run_id = ? ORDER BY name",
                (run_id,),
            ).fetchall()
        ]
    return run
