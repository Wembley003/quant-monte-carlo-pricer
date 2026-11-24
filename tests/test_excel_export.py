"""Pruebas de la exportación a Excel."""

import zipfile

import pytest
from openpyxl import load_workbook

from monte_carlo_pricer import excel_export, persistence


def _dos_corridas(db):
    persistence.record_mc_run(db, 100.0, 100.0, 1.0, 0.05, 0.20,
                              n_paths=5_000, seed=1)
    persistence.record_mc_run(db, 100.0, 110.0, 1.0, 0.05, 0.20,
                              n_paths=5_000, seed=2)


def test_export_crea_tres_hojas_con_datos(tmp_path):
    db = tmp_path / "runs.db"
    xlsx = tmp_path / "runs.xlsx"
    _dos_corridas(db)
    out = excel_export.export_workbook(db, xlsx)
    assert out == xlsx
    wb = load_workbook(xlsx)
    assert wb.sheetnames == ["Runs", "Greeks", "Resumen",
                             "Dinámica por opción", "Dinámica por Griega"]
    runs = wb["Runs"]
    assert runs.max_row == 3  # encabezado + 2 corridas
    assert runs["A2"].value == 1
    # Columna K = precio MC, columna M = precio Black-Scholes.
    assert runs["K2"].value == pytest.approx(10.45, abs=0.5)
    assert runs["M2"].value == pytest.approx(10.4506, abs=1e-3)
    greeks = wb["Greeks"]
    assert greeks.max_row == 11  # encabezado + 5 griegas x 2 corridas
    resumen = wb["Resumen"]
    assert resumen["A2"].value == "Corridas guardadas"
    assert resumen["B2"].value == 2


def test_export_con_base_vacia(tmp_path):
    db = tmp_path / "vacia.db"
    xlsx = tmp_path / "vacia.xlsx"
    persistence.init_db(db)
    excel_export.export_workbook(db, xlsx)
    wb = load_workbook(xlsx)
    assert wb["Runs"].max_row == 1  # solo el encabezado
    assert wb["Resumen"]["A2"].value == "Corridas guardadas"
    assert wb["Resumen"]["B2"].value == 0


def test_export_incluye_tablas_dinamicas_nativas(tmp_path):
    # Una tabla dinámica nativa no es una hoja más: son tres piezas
    # dentro del zip (definición, caché y registros) por tabla. Si
    # falta una, Excel dice que el archivo está dañado, así que el
    # test abre el .xlsx como zip y exige las seis piezas, dos por
    # pivote, además de las hojas donde Excel las dibuja.
    db = tmp_path / "runs.db"
    xlsx = tmp_path / "runs.xlsx"
    _dos_corridas(db)
    excel_export.export_workbook(db, xlsx)
    with zipfile.ZipFile(xlsx) as zf:
        partes = set(zf.namelist())
    for i in (1, 2):
        assert f"xl/pivotTables/pivotTable{i}.xml" in partes
        assert f"xl/pivotCache/pivotCacheDefinition{i}.xml" in partes
        assert f"xl/pivotCache/pivotCacheRecords{i}.xml" in partes
    wb = load_workbook(xlsx)
    assert "Dinámica por opción" in wb.sheetnames
    assert "Dinámica por Griega" in wb.sheetnames
