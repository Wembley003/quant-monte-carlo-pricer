"""Pruebas de la estructura por capas del pricer.

Las tres capas del proyecto (Bronce/Plata/Oro NO aplica aquí: no hay
un pipeline de datos con staging crudo → limpio → agregado; los
insumos son parámetros escalares y no hay nada que limpiar):

    Client (presentation):  examples/demo.py (la demo que corre el usuario)
    Business rules:         black_scholes, stochastic_processes,
                            monte_carlo, greeks, implied_vol
                            (los modelos, la simulación y las Griegas)
    Queries & analysis:     persistence, excel_export, tableau_extract,
                            pivot_table (los resultados, guardados y
                            preguntados; en Python, no en SQL)

La cadena es la del cálculo: las Reglas producen los números y las
Consultas solo los guardan y los preguntan. Nada se salta de capa: si
cambian las reglas, cambia todo lo demás. Y las reglas no dependen del
formato de salida: el precio se calcula igual sin Excel ni SQLite.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from monte_carlo_pricer import (
    black_scholes,
    greeks,
    monte_carlo,
    persistence,
    stochastic_processes,
)

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
_PROJECT = Path(__file__).resolve().parent.parent
_PACKAGE = _PROJECT / "monte_carlo_pricer"
_PAQUETE = "monte_carlo_pricer"

_CAPAS = {
    "cliente": {"demo"},
    "reglas": {"black_scholes", "stochastic_processes", "monte_carlo", "greeks", "implied_vol"},
    "consultas": {"persistence", "excel_export", "tableau_extract", "pivot_table"},
}
# Módulos de almacenamiento/salida que las reglas no deben conocer.
_SALIDA = {"persistence", "excel_export", "tableau_extract", "pivot_table"}
# Librerías de almacenamiento/salida que cliente y reglas no deben importar.
_LIBS_SALIDA = {"openpyxl", "sqlite3"}


def test_capas_en_orden_precio_griegas_guardado(tmp_path):
    # Reglas: el proceso que mueve el precio (GBM exacto) y la fórmula
    # cerrada. La media de los caminos simulados debe acercarse a la
    # media teórica del proceso — si no, la simulación está mal desde
    # el origen y nada de lo que sigue vale.
    times, paths = stochastic_processes.simulate_gbm_exact(
        S0, R, SIGMA, T, n_steps=10, n_paths=5_000, seed=1
    )
    assert paths.shape == (11, 5_000)
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(T)
    mean, _var = stochastic_processes.gbm_moments(S0, R, SIGMA, T)
    assert paths[-1].mean() == pytest.approx(mean, rel=0.05)

    # Reglas: Monte Carlo con esos mismos insumos, comparado con la
    # fórmula cerrada.
    mc_price, mc_se = monte_carlo.mc_european_price(
        S0, K, T, R, SIGMA, n_paths=50_000, seed=42
    )
    bs_price = black_scholes.black_scholes_price(S0, K, T, R, SIGMA, "call")
    assert mc_price == pytest.approx(bs_price, abs=4 * mc_se)

    # Reglas: las Griegas derivan la sensibilidad del mismo precio
    # que acabamos de calcular, no de otro.
    compared = greeks.compare_greeks(S0, K, T, R, SIGMA, n_paths=50_000, seed=42)
    assert "delta" in compared
    delta_mc, delta_bs, _err = compared["delta"]
    assert delta_mc == pytest.approx(delta_bs, abs=0.05)

    # Consultas: lo que se guarda es exactamente lo que calcularon
    # las reglas, sin re-calcular nada por otro camino.
    run_id = persistence.record_mc_run(
        tmp_path / "runs.db", S0, K, T, R, SIGMA, n_paths=50_000, seed=42
    )
    run = persistence.get_run(tmp_path / "runs.db", run_id)
    assert run["mc_price"] == pytest.approx(mc_price)
    assert run["bs_price"] == pytest.approx(bs_price)
    greeks_guardadas = {g["name"]: g["mc_value"] for g in run["greeks"]}
    assert greeks_guardadas["delta"] == pytest.approx(delta_mc)


def test_cambiar_los_insumos_cambia_los_resultados():
    # Si la volatilidad sube, el precio de la call sube: la capa de
    # Consultas no tiene opinión propia, refleja las Reglas.
    low = monte_carlo.mc_european_price(S0, K, T, R, 0.10, n_paths=50_000, seed=42)[0]
    high = monte_carlo.mc_european_price(S0, K, T, R, 0.40, n_paths=50_000, seed=42)[0]
    assert high > low


# --- Las capas no se mezclan (análisis estático de imports) -------------

def _modulos_importados(path: Path) -> set[str]:
    """Nombres de módulo que un archivo importa (resueltos a su raíz)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nombres = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                partes = alias.name.split(".")
                nombres.add(partes[1] if partes[0] == _PAQUETE and len(partes) > 1 else partes[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                partes = node.module.split(".")
                nombres.add(partes[1] if partes[0] == _PAQUETE and len(partes) > 1 else partes[0])
            else:
                for alias in node.names:
                    nombres.add(alias.name.split(".")[0])
    return nombres


def _capa_de(stem: str) -> str:
    for capa, modulos in _CAPAS.items():
        if stem in modulos:
            return capa
    raise AssertionError(f"módulo sin capa asignada: {stem}")


def test_todos_los_modulos_tienen_capa():
    # Cada módulo del paquete pertenece a exactamente una capa; si
    # alguien agrega un archivo y se le olvida ubicarlo, esta prueba
    # lo delata. __init__.py es la fachada pública, no una capa.
    vistos = set()
    for path in sorted(_PACKAGE.glob("*.py")):
        if path.stem == "__init__":
            continue
        vistos.add(path.stem)
        _capa_de(path.stem)
    assert vistos == _CAPAS["reglas"] | _CAPAS["consultas"]
    assert (_PROJECT / "examples" / "demo.py").exists()


def test_capas_no_se_mezclan():
    # Reglas: pura matemática, sin almacenamiento ni salida. Cliente:
    # sin tocar el almacenamiento directo. Consultas: sin llamar al
    # cliente. La demo (examples/) es orquestadora: arma las capas,
    # así que ahí solo se prohíbe tocar las librerías de salida.
    for path in sorted(_PACKAGE.glob("*.py")):
        if path.stem == "__init__":
            continue
        capa = _capa_de(path.stem)
        importados = _modulos_importados(path)
        if capa == "reglas":
            assert not (importados & _SALIDA), (
                f"{path.stem} (reglas) importa módulos de salida: {importados & _SALIDA}"
            )
            assert not (importados & _CAPAS["cliente"]), (
                f"{path.stem} (reglas) importa del cliente: {importados & _CAPAS['cliente']}"
            )
            assert not (importados & _LIBS_SALIDA), (
                f"{path.stem} (reglas) importa librerías de salida: {importados & _LIBS_SALIDA}"
            )
        elif capa == "consultas":
            assert not (importados & _CAPAS["cliente"]), (
                f"{path.stem} (consultas) importa del cliente: {importados & _CAPAS['cliente']}"
            )
    demo = _modulos_importados(_PROJECT / "examples" / "demo.py")
    assert not (demo & _LIBS_SALIDA), (
        f"demo.py (cliente) toca librerías de salida: {demo & _LIBS_SALIDA}"
    )
    assert not (demo & _CAPAS["consultas"]), (
        f"demo.py (cliente) importa de consultas: {demo & _CAPAS['consultas']}"
    )


def test_reglas_no_dependen_del_formato_de_salida():
    # La prueba fuerte: con openpyxl y sqlite3 bloqueados, las reglas
    # siguen dando el precio y la delta. Si el cálculo dependiera de
    # Excel o de la base, esto se cae.
    codigo = textwrap.dedent(
        """
        import importlib.abc
        import sys
        sys.path.insert(0, r"%s")

        class _Bloqueo(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"openpyxl", "sqlite3"}:
                    raise ImportError("bloqueado en esta prueba: " + fullname)
                return None

        sys.meta_path.insert(0, _Bloqueo())
        from monte_carlo_pricer import black_scholes, greeks, monte_carlo
        precio_mc, _se = monte_carlo.mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20, n_paths=50_000, seed=42)
        precio_bs = black_scholes.black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.20, "call")
        assert abs(precio_mc - precio_bs) < 0.15, (precio_mc, precio_bs)
        delta_mc, delta_bs, _err = greeks.compare_greeks(100.0, 100.0, 1.0, 0.05, 0.20, n_paths=50_000, seed=42)["delta"]
        assert abs(delta_mc - delta_bs) < 0.05, (delta_mc, delta_bs)
        print("reglas OK sin openpyxl ni sqlite3")
        """
        % str(_PROJECT)
    )
    resultado = subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT),
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "reglas OK sin openpyxl ni sqlite3" in resultado.stdout
