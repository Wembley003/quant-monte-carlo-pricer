"""Pruebas del volcado a Access: CSV + esquema y .accdb real.

Access no se puede fabricar con Python en Linux, así que el camino
es en dos pasos: primero se vuelca la base SQLite a CSV + esquema
(``access/dump_access.py``) y después Java con Jackcess construye el
.accdb real (``access/build_access.sh``). El primer test cubre el
volcado; el segundo, la construcción completa, pero solo si el
toolchain está disponible (JDK, curl, mdbtools y los jars
descargables de Maven Central): en una máquina sin eso, el test se
salta en vez de fallar por algo que no es el código.
"""

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from monte_carlo_pricer import persistence

ACCESS_DIR = Path(__file__).resolve().parent.parent / "access"


def _cargar_dump_access():
    spec = importlib.util.spec_from_file_location(
        "dump_access", ACCESS_DIR / "dump_access.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base_juguete(tmp_path_factory):
    """Dos corridas chicas: lo justo para que haya runs y greeks."""
    db = tmp_path_factory.mktemp("db") / "runs.db"
    persistence.record_mc_run(db, 100.0, 100.0, 1.0, 0.05, 0.20,
                              n_paths=1_000, seed=1)
    persistence.record_mc_run(db, 100.0, 110.0, 1.0, 0.05, 0.20,
                              n_paths=1_000, seed=2)
    return db


def _filas_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_volcado_produce_csvs_y_esquema(base_juguete, tmp_path):
    dump_access = _cargar_dump_access()
    out_dir = tmp_path / "access"
    resumen = dump_access.dump(base_juguete, out_dir)

    por_nombre = {t["name"]: t for t in resumen["tables"]}
    assert set(por_nombre) == {"runs", "greeks"}
    assert por_nombre["runs"]["rows"] == 2
    assert por_nombre["greeks"]["rows"] == 10  # 5 griegas x 2 corridas

    for tabla in ("runs", "greeks"):
        filas = _filas_csv(out_dir / "csv" / f"{tabla}.csv")
        assert filas[0]  # encabezados
        assert len(filas) - 1 == por_nombre[tabla]["rows"]
    # La primera corrida es un call con precio MC cerca del de BS.
    encabezados = _filas_csv(out_dir / "csv" / "runs.csv")[0]
    primera = _filas_csv(out_dir / "csv" / "runs.csv")[1]
    fila = dict(zip(encabezados, primera))
    assert fila["option"] == "call"
    assert abs(float(fila["mc_price"]) - float(fila["bs_price"])) < 1.0

    esquema = (out_dir / "schema.txt").read_text(encoding="utf-8")
    assert "TABLE|runs" in esquema
    assert "TABLE|greeks" in esquema
    assert "COL|mc_price|DOUBLE|" in esquema
    resumen_json = json.loads((out_dir / "schema.json").read_text())
    assert [t["name"] for t in resumen_json["tables"]] == ["runs", "greeks"]


def _toolchain_disponible():
    """JDK + curl + mdbtools y los jars de Jackcess descargables."""
    for herramienta in ("java", "javac", "curl", "mdb-tables", "mdb-export"):
        if shutil.which(herramienta) is None:
            return False, f"falta {herramienta}"
    jar = ("https://repo1.maven.org/maven2/com/healthmarketscience/"
           "jackcess/jackcess/4.0.5/jackcess-4.0.5.jar")
    cabezal = subprocess.run(
        ["curl", "-sI", "--max-time", "10", jar],
        capture_output=True, text=True,
    )
    if cabezal.returncode != 0 or "200" not in cabezal.stdout.splitlines()[0]:
        return False, "no se puede descargar Jackcess de Maven Central"
    return True, ""


def test_construye_accdb_real(base_juguete, tmp_path):
    disponible, motivo = _toolchain_disponible()
    if not disponible:
        pytest.skip(f"sin toolchain de Access: {motivo}")
    # Copia del directorio access/ a tmp: el script escribe el .accdb,
    # los CSV y los jars dentro de su propio directorio.
    trabajo = tmp_path / "access"
    shutil.copytree(ACCESS_DIR, trabajo)
    accdb = trabajo / "monte-carlo-pricer.accdb"
    proc = subprocess.run(
        ["bash", str(trabajo / "build_access.sh"),
         str(base_juguete), "monte-carlo-pricer.accdb"],
        capture_output=True, text=True, cwd=trabajo,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert accdb.exists() and accdb.stat().st_size > 0

    tablas = subprocess.run(
        ["mdb-tables", "-1", str(accdb)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert set(tablas) == {"runs", "greeks"}
    for tabla, esperadas in (("runs", 2), ("greeks", 10)):
        salida = subprocess.run(
            ["mdb-export", str(accdb), tabla],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        assert len(salida) - 1 == esperadas, salida
    # El contenido también tiene que ser el mismo, no solo el conteo.
    salida = subprocess.run(
        ["mdb-export", str(accdb), "runs"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "call" in salida


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
