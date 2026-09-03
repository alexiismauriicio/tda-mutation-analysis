#!/usr/bin/env python3
"""
Combina los descriptores topologicos (salida de tda_mutation_analysis_v2.py) con los
objetivos de fitness de cada experimento (energy_score [+ dLL] [+ nmut]) para obtener
un indice de calidad de la mutacion en el espacio topologico.

Funciona para:
  - single-objective (1 objetivo: energy_score; nmut se calcula solo con fines descriptivos)
  - bi-obj           (2 objetivos: energy_score, dll)
  - run1 3-obj       (3 objetivos: energy_score, dll, nmut) -- usar topo_quality_index.py (el original)

Ejemplos:

  python topo_quality_index_v2.py \
      --tda_csv tda_results_single_objective/tda_features_with_scores.csv \
      --fitness_csv single_objective_all.csv \
      --objectives score:min \
      --out topo_quality_single_objective.csv

  python topo_quality_index_v2.py \
      --tda_csv tda_results_biobj_run12/tda_features_with_scores.csv \
      --fitness_csv bi_obj_unique_with_pareto.csv \
      --objectives score:min,ll:max \
      --out topo_quality_biobj_run12.csv
"""
import argparse
import re
import numpy as np
import pandas as pd


def pareto_rank(df, cols_directions):
    n = len(df)
    vals = []
    for col, direction in cols_directions:
        v = df[col].to_numpy(dtype=float)
        vals.append(v if direction == "min" else -v)
    vals = np.vstack(vals).T

    remaining = set(range(n))
    ranks = np.full(n, -1, dtype=int)
    current_rank = 0
    while remaining:
        idx = np.array(sorted(remaining))
        pts = vals[idx]
        m = len(idx)
        is_front = np.ones(m, dtype=bool)
        for i in range(m):
            if not is_front[i]:
                continue
            better_or_eq = np.all(pts <= pts[i], axis=1)
            strictly_better = np.any(pts < pts[i], axis=1)
            dominators = better_or_eq & strictly_better
            dominators[i] = False
            if dominators.any():
                is_front[i] = False
        front_idx = idx[is_front]
        ranks[front_idx] = current_rank
        remaining -= set(front_idx.tolist())
        current_rank += 1
    return ranks


def norm_key(name):
    """Fallback key normalizer for simple single-run cases (no run/generation
    context needed -- see build_fit_key for the multi-run case)."""
    name = re.sub(r"\.pdb$", "", str(name))
    name = re.sub(r"_0001$", "", name)
    return name


def build_fit_key(row, pdb_col):
    """Build a fitness-CSV key that matches the composite keys produced by
    tda_mutation_analysis_v2.py's safe_name(path, base_dir=...).

    IMPORTANT: if the tda_csv was generated with --pdb_dir pointing at a
    parent folder containing several independent run subfolders (e.g.
    --pdb_dir single-objective, which contains run1/ and run2/), the
    "mutation" key there looks like "run1_g0_g0_00" (run + generation +
    individual, to avoid collisions between e.g. run1/g0/g0_00.pdb and
    run2/g0/g0_00.pdb). This function rebuilds that same composite string
    from the fitness CSV's own run/generation/pdb_file columns so both
    sides match unambiguously instead of silently colliding on the bare
    "g0_00" filename, which used to corrupt the merge.

    Different fitness CSVs encode the pdb path differently:
      - single_objective_all.csv: pdb_col is a bare filename ("g0_00.pdb") ->
        the run/generation prefix must be built from the separate "run" /
        "generation" columns.
      - bi_obj_unique_with_pareto.csv: pdb_col ("file") already contains the
        run/generation path ("run12/g1/g1_24.pdb") -> just normalize that
        path directly; re-prepending run/generation here would double it up
        and produce a key that matches nothing (this was the bug that gave
        "0 de 750 topologicos" on the bi-obj run).
    """
    raw = str(row[pdb_col])
    if "/" in raw or "\\" in raw:
        key = re.sub(r"\.pdb$", "", raw)
        key = re.sub(r"_0001$", "", key)
        key = re.sub(r"[\\/]+", "_", key)
        return key

    base = norm_key(raw)
    parts = []
    if "run" in row.index and pd.notna(row["run"]):
        run_val = row["run"]
        run_str = str(int(run_val)) if float(run_val).is_integer() else str(run_val)
        parts.append(f"run{run_str}")
    if "generation" in row.index and pd.notna(row["generation"]):
        gen_val = row["generation"]
        gen_str = str(int(gen_val)) if float(gen_val).is_integer() else str(gen_val)
        parts.append(f"g{gen_str}")
    parts.append(base)
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tda_csv", required=True)
    ap.add_argument("--fitness_csv", required=True,
                     help="CSV con una columna de nombre de archivo pdb (pdb_file o file) y las columnas de objetivos")
    ap.add_argument("--pdb_col", default=None, help="Nombre de la columna con el archivo pdb en fitness_csv (auto-detecta pdb_file/file si se omite)")
    ap.add_argument("--objectives", required=True,
                     help="Lista 'columna:direccion' separada por comas, p.ej. 'energy_score:min,dll:max'")
    ap.add_argument("--out", default="topo_quality.csv")
    args = ap.parse_args()

    tda = pd.read_csv(args.tda_csv)
    fit = pd.read_csv(args.fitness_csv)

    pdb_col = args.pdb_col
    if pdb_col is None:
        for cand in ["pdb_file", "file"]:
            if cand in fit.columns:
                pdb_col = cand
                break
    if pdb_col is None:
        raise SystemExit("No se encontro columna de archivo pdb en fitness_csv; usa --pdb_col")

    objectives = []
    for tok in args.objectives.split(","):
        col, direction = tok.split(":")
        col, direction = col.strip(), direction.strip().lower()
        assert direction in ("min", "max")
        objectives.append((col, direction))

    tda["_key"] = tda["mutation"].map(norm_key)
    if "run" in fit.columns or "generation" in fit.columns:
        fit["_key"] = fit.apply(lambda r: build_fit_key(r, pdb_col), axis=1)
    else:
        fit["_key"] = fit[pdb_col].map(norm_key)

    # Sanity check: if either side has duplicate keys, the merge will silently
    # drop/mispair rows (this is exactly the bug that used to corrupt this
    # analysis when run1/run2 filenames collided). Warn loudly instead of
    # failing silently.
    tda_dupes = tda["_key"].duplicated().sum()
    fit_dupes = fit["_key"].duplicated().sum()
    if tda_dupes:
        print(f"ADVERTENCIA: {tda_dupes} claves duplicadas en tda_csv (columna 'mutation') -- "
              f"revisa que tda_mutation_analysis_v2.py se haya corrido con --pdb_dir apuntando "
              f"a la carpeta que contiene TODAS las subcarpetas de corrida (run1/, run2/, ...).")
    if fit_dupes:
        print(f"ADVERTENCIA: {fit_dupes} claves duplicadas en fitness_csv -- probablemente faltan "
              f"columnas 'run'/'generation' para desambiguar individuos con el mismo nombre de archivo "
              f"en corridas distintas.")

    keep_cols = ["_key"] + [c for c, _ in objectives]
    if "generation" in fit.columns and "generation" not in [c for c, _ in objectives]:
        keep_cols.append("generation")
    if "run" in fit.columns:
        keep_cols.append("run")

    merged = tda.merge(fit[keep_cols].drop_duplicates("_key"), on="_key", how="inner")
    print(f"Individuos combinados (topologia + fitness): {len(merged)} de {len(tda)} topologicos / {len(fit)} con fitness")
    if len(merged) == 0:
        raise SystemExit("0 coincidencias -- revisa que los nombres de archivo pdb coincidan entre ambos CSV (columna 'mutation' en tda_csv vs. pdb_col en fitness_csv).")

    if len(objectives) == 1:
        col, direction = objectives[0]
        merged["quality_rank"] = merged[col].rank(ascending=(direction == "min")).astype(int) - 1
    else:
        merged["quality_rank"] = pareto_rank(merged, objectives)

    feature_cols = [c for c in merged.columns if re.match(r"H[012]_", c) or c in
                    ["H1H2_sum_persistence", "H1H2_count", "n_interface_points"]]

    targets = [c for c, _ in objectives] + ["quality_rank"]
    print(f"\nCorrelacion de Spearman: descriptores topologicos vs. {targets}")
    corr = merged[feature_cols + targets].corr(method="spearman")[targets].loc[feature_cols]
    corr.to_csv(args.out.replace(".csv", "_correlations.csv"))
    print(corr.round(3).to_string())

    merged.to_csv(args.out, index=False)
    print(f"\nGuardado: {args.out}")
    print(f"Guardado: {args.out.replace('.csv', '_correlations.csv')}")


if __name__ == "__main__":
    main()
