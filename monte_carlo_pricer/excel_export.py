"""
Layer: Queries & analysis — the saved runs as an Excel workbook.
Exportar las corridas guardadas a Excel (openpyxl).

La base de SQLite es para las máquinas; el Excel es para las personas:
un archivo que se puede abrir, filtrar y mandar por correo sin pedirle
a nadie que instale nada. Salen tres hojas: las corridas, las Griegas
y un resumen chico con los errores medios entre Monte Carlo y
Black-Scholes, que es lo primero que uno revisa para saber si la
simulación está sana.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

from .pivot_table import add_pivot_table

__all__ = ["export_workbook"]

RUN_COLUMNS = [
    ("id", "ID"),
    ("created_at", "Fecha (UTC)"),
    ("s0", "S0"),
    ("k", "K"),
    ("t", "T (años)"),
    ("r", "r"),
    ("sigma", "sigma"),
    ("n_paths", "Trayectorias"),
    ("option", "Opción"),
    ("seed", "Semilla"),
    ("mc_price", "Precio MC"),
    ("mc_se", "Error estándar MC"),
    ("bs_price", "Precio Black-Scholes"),
]


def _write_sheet(wb, title, headers, rows):
    ws = wb.create_sheet(title)
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in rows:
        ws.append(row)
    for column in ws.columns:
        width = max(len(str(cell.value)) if cell.value is not None else 0
                    for cell in column)
        ws.column_dimensions[column[0].column_letter].width = min(width + 2, 30)
    return ws


def export_workbook(db_path, xlsx_path):
    """Lee la base de SQLite y escribe el libro de Excel."""
    db_path = Path(db_path)
    xlsx_path = Path(xlsx_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        runs = [dict(r) for r in conn.execute(
            "SELECT * FROM runs ORDER BY id"
        ).fetchall()]
        greeks = [dict(g) for g in conn.execute(
            "SELECT run_id, name, mc_value, bs_value FROM greeks"
            " ORDER BY run_id, name"
        ).fetchall()]

    wb = Workbook()
    wb.remove(wb.active)

    run_rows = []
    for run in runs:
        run_rows.append([run[key] for key, _ in RUN_COLUMNS])
    runs_ws = _write_sheet(
        wb, "Runs", [label for _, label in RUN_COLUMNS], run_rows)

    greek_rows = []
    for g in greeks:
        greek_rows.append([
            g["run_id"], g["name"], g["mc_value"], g["bs_value"],
            abs(g["mc_value"] - g["bs_value"]),
        ])
    greeks_ws = _write_sheet(
        wb, "Greeks",
        ["Run ID", "Griega", "Valor MC", "Valor Black-Scholes", "Error abs."],
        greek_rows,
    )

    # El resumen es lo primero que se mira: si el error medio entre MC
    # y Black-Scholes es mucho mayor que el error estándar típico, la
    # simulación tiene un problema y no los datos.
    if runs:
        errors = [abs(r["mc_price"] - r["bs_price"]) for r in runs]
        ses = [r["mc_se"] for r in runs]
        summary_rows = [
            ["Corridas guardadas", len(runs)],
            ["Error medio |MC - BS|", sum(errors) / len(errors)],
            ["Error máximo |MC - BS|", max(errors)],
            ["Error estándar MC medio", sum(ses) / len(ses)],
            ["Error medio en unidades de SE",
             (sum(errors) / len(errors)) / (sum(ses) / len(ses))],
        ]
    else:
        summary_rows = [["Corridas guardadas", 0]]
    _write_sheet(wb, "Resumen", ["Métrica", "Valor"], summary_rows)

    # Tablas dinámicas NATIVAS de Excel: no son fórmulas ni valores
    # pegados, sino el objeto pivotTable que Excel recalcula al abrir.
    # ¿Por qué promedios por opción y por Griega? Porque lo que uno
    # pregunta al mirar las corridas no es el precio de una corrida
    # suelta, sino si el método converge igual para calls que para
    # puts (y con qué error estándar), y qué Griega es la que más se
    # desvía del valor analítico. El promedio agrupa ese ruido de
    # Monte Carlo — que cae como 1/√n — en un número por grupo.
    # Si una hoja de origen no tiene filas, add_pivot_table avisa con
    # ValueError y ese pivote se salta: una base vacía sigue
    # exportando sus tres hojas sin romperse.
    try:
        add_pivot_table(
            wb,
            source_sheet=runs_ws,
            pivot_sheet_title="Dinámica por opción",
            pivot_name="PivotOpcion",
            row_fields=["Opción"],
            data_fields=[
                ("Precio MC", "Precio MC promedio", "average"),
                ("Precio Black-Scholes", "Precio BS promedio", "average"),
                ("Error estándar MC", "Error estándar promedio", "average"),
            ],
        )
    except ValueError:
        pass
    try:
        add_pivot_table(
            wb,
            source_sheet=greeks_ws,
            pivot_sheet_title="Dinámica por Griega",
            pivot_name="PivotGriega",
            row_fields=["Griega"],
            data_fields=[
                ("Error abs.", "Error abs. promedio", "average"),
                ("Valor MC", "Valor MC promedio", "average"),
            ],
        )
    except ValueError:
        pass

    wb.save(xlsx_path)
    return xlsx_path
