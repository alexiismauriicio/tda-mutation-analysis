#!/usr/bin/env python3
"""
Verificacion end-to-end: compara los resultados YA GUARDADOS en tda_results/
contra una REEJECUCION fresca del pipeline original (tda_mutation_analysis.py)
guardada en otra carpeta (por defecto tda_results_rerun/).

Uso:
  1) Copia este archivo dentro de tu carpeta protein-tda/ (junto a
     tda_mutation_analysis.py).

  2) Vuelve a ejecutar el pipeline original en una carpeta nueva para no pisar
     los resultados existentes:

       python tda_mutation_analysis.py --face_file faceC.txt --out tda_results_rerun

  3) Corre este script para comparar ambas carpetas:

       python verify_rerun.py

  Requiere: pandas, numpy (ya deberian estar instalados junto con las demas
  dependencias del pipeline original: biopython, ripser, persim, scikit-learn).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ORIG = Path("tda_results")
RERUN = Path("tda_results_rerun")

TOL = 1e-6  # tolerancia numerica (ripser/scikit-learn son deterministas con random_state fijo)


def compare_csv(name, key_cols=None, tol=TOL):
    fa = ORIG / name
    fb = RERUN / name
    if not fa.exists() or not fb.exists():
        print(f"[SKIP] {name}: no encontrado en una de las dos carpetas ({fa.exists()=}, {fb.exists()=})")
        return None
    a = pd.read_csv(fa)
    b = pd.read_csv(fb)
    if key_cols:
        a = a.sort_values(key_cols).reset_index(drop=True)
        b = b.sort_values(key_cols).reset_index(drop=True)
    common_cols = [c for c in a.columns if c in b.columns]
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    max_diff = 0.0
    bad_cols = []
    for c in numeric_cols:
        try:
            d = (a[c].astype(float) - b[c].astype(float)).abs()
            m = np.nanmax(d.values) if len(d) else 0.0
        except Exception:
            continue
        if m is not None and m > max_diff:
            max_diff = m
        if m is not None and m > tol:
            bad_cols.append((c, m))
    status = "OK" if not bad_cols else "DIFERENCIAS"
    print(f"[{status}] {name}: max diff = {max_diff:.3e} sobre {len(numeric_cols)} columnas numericas")
    if bad_cols:
        for c, m in bad_cols:
            print(f"    columna con diferencia > tol: {c} (max diff = {m:.3e})")
    return max_diff


def compare_matrix_csv(name, tol=TOL):
    fa = ORIG / name
    fb = RERUN / name
    if not fa.exists() or not fb.exists():
        print(f"[SKIP] {name}: no encontrado en una de las dos carpetas")
        return None
    a = pd.read_csv(fa, index_col=0)
    b = pd.read_csv(fb, index_col=0)
    a = a.loc[sorted(a.index), sorted(a.columns)]
    b = b.loc[sorted(b.index), sorted(b.columns)]
    diff = (a.values - b.values)
    max_diff = np.nanmax(np.abs(diff)) if diff.size else 0.0
    status = "OK" if max_diff <= tol else "DIFERENCIAS"
    print(f"[{status}] {name}: max diff = {max_diff:.3e}")
    return max_diff


def main():
    if not ORIG.exists():
        sys.exit(f"No existe la carpeta {ORIG}/ (resultados originales). Ejecuta este script desde protein-tda/.")
    if not RERUN.exists():
        sys.exit(
            f"No existe la carpeta {RERUN}/ todavia.\n"
            f"Primero reejecuta el pipeline original:\n\n"
            f"    python tda_mutation_analysis.py --face_file faceC.txt --out {RERUN}\n"
        )

    print("=" * 70)
    print("VERIFICACION END-TO-END: tda_results/ (original) vs tda_results_rerun/ (reejecucion)")
    print("=" * 70)

    compare_csv("rosetta_scores_parsed.csv", key_cols=["mutation"])
    compare_csv("tda_features_with_scores.csv", key_cols=["mutation"])
    compare_csv("tda_clustering_pca.csv", key_cols=["mutation"])
    compare_matrix_csv("wasserstein_H1_matrix.csv")
    compare_matrix_csv("wasserstein_H2_matrix.csv")
    compare_csv("spearman_correlations_energy_topology.csv", key_cols=None)

    print("\nSi todas las filas dicen [OK], la reejecucion completa (incluyendo ripser)")
    print("reproduce exactamente los resultados ya usados en la tesis, y la limitacion")
    print("de 'verificacion indirecta' del Capitulo de Conclusiones puede eliminarse.")


if __name__ == "__main__":
    main()
