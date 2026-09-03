#!/usr/bin/env python3
"""
Distancia de Wasserstein entre el individuo silvestre y un mutante,
y datos de apoyo para el analisis interpretativo fisico/biologico
solicitado por el tutor de la tesis (Capitulo de Resultados, seccion
"Distancia de Wasserstein entre el individuo silvestre y el mejor mutante").

Requiere el mismo entorno que tda_mutation_analysis_v2.py:
    pip install numpy pandas matplotlib biopython ripser persim

Uso tipico (ajustar rutas a tu copia local de los datos):
    python wasserstein_wt_vs_mutant.py \
        --wt run1/g0/g0_00.pdb \
        --mut run1/g19/g19_24.pdb \
        --face_file faceC.txt \
        --chain_mut C --chain_partner A --cutoff 8.0 \
        --out wasserstein_wt_vs_best

Que hace:
    1. Extrae la nube de puntos de interfaz (modo "sidechain", igual que el
       resto de la tesis) para ambas estructuras.
    2. Calcula sus diagramas de persistencia H0/H1/H2 con ripser.
    3. Calcula la distancia de Wasserstein W1 entre los diagramas H1 y H2
       de ambas estructuras (persim.wasserstein).
    4. Guarda los diagramas de persistencia superpuestos (una figura) y un
       resumen en texto con los descriptores de cada estructura lado a lado,
       para facilitar la redaccion del analisis interpretativo fisico y
       biologico (que descriptor cambia mas, en que dimension, etc.).

Este script reutiliza la misma logica de extraccion de interfaz que
tda_mutation_analysis_v2.py (mismo modulo, ver ese archivo en este
repositorio) para asegurar que la comparacion sea exactamente consistente
con el resto del pipeline de la tesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from Bio.PDB import PDBParser
except ImportError as exc:
    raise SystemExit("Falta biopython. Instalar con: pip install biopython") from exc

try:
    from ripser import ripser
except ImportError as exc:
    raise SystemExit("Falta ripser. Instalar con: pip install ripser") from exc

try:
    from persim import wasserstein, plot_diagrams
except ImportError as exc:
    raise SystemExit("Falta persim. Instalar con: pip install persim") from exc


def load_face_file(face_file):
    if not face_file:
        return None
    residues = set()
    with open(face_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    residues.add((parts[0], int(parts[1])))
                except ValueError:
                    continue
    return residues


def residue_ca_coords(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_file))
    coords = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if "CA" not in residue:
                    continue
                res_id = int(residue.get_id()[1])
                coords.append((res_id, residue["CA"].get_coord().astype(float)))
        break
    return coords


def residue_heavy_atom_coords(pdb_file, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_file))
    out = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if "CA" not in residue:
                    continue
                res_id = int(residue.get_id()[1])
                atoms = [a.get_coord().astype(float) for a in residue if a.element not in ("H", "D")]
                out[res_id] = atoms
        break
    return out


def extract_interface_coords(pdb_file, chain_mut, chain_partner, cutoff, face_residues, atom_mode="sidechain"):
    mut_ca = residue_ca_coords(pdb_file, chain_mut)
    partner_ca = residue_ca_coords(pdb_file, chain_partner)
    mut_ca_map = dict(mut_ca)

    if face_residues is not None:
        mut_selected_ids = [rid for rid, _ in mut_ca if (chain_mut, rid) in face_residues]
    else:
        mut_selected_ids = [rid for rid, _ in mut_ca]

    partner_selected_ids = []
    if mut_selected_ids:
        mut_mat = np.vstack([mut_ca_map[rid] for rid in mut_selected_ids])
        for rid, coord in partner_ca:
            dmin = np.min(np.linalg.norm(mut_mat - coord, axis=1))
            if dmin <= cutoff:
                partner_selected_ids.append(rid)

    if atom_mode == "sidechain":
        mut_atoms_by_res = residue_heavy_atom_coords(pdb_file, chain_mut)
        partner_atoms_by_res = residue_heavy_atom_coords(pdb_file, chain_partner)
        mut_coords = [c for rid in mut_selected_ids for c in mut_atoms_by_res.get(rid, [])]
        partner_coords = [c for rid in partner_selected_ids for c in partner_atoms_by_res.get(rid, [])]
    else:
        partner_ca_map = dict(partner_ca)
        mut_coords = [mut_ca_map[rid] for rid in mut_selected_ids]
        partner_coords = [partner_ca_map[rid] for rid in partner_selected_ids]

    coords = mut_coords + partner_coords
    meta = {
        "n_mut_chain_residues": len(mut_selected_ids),
        "n_partner_residues": len(partner_selected_ids),
        "n_interface_points": len(coords),
    }
    return (np.vstack(coords) if coords else np.empty((0, 3))), meta


def finite_intervals(dgm):
    dgm = np.asarray(dgm, dtype=float)
    if dgm.size == 0:
        return np.empty((0, 2))
    return dgm[np.isfinite(dgm[:, 1])]


def topological_features(dgms):
    features = {}
    for dim in range(3):
        dgm = finite_intervals(dgms[dim]) if dim < len(dgms) else np.empty((0, 2))
        lifetimes = dgm[:, 1] - dgm[:, 0] if len(dgm) else np.array([])
        features[f"H{dim}_count"] = float(len(lifetimes))
        features[f"H{dim}_sum_persistence"] = float(np.sum(lifetimes)) if len(lifetimes) else 0.0
        features[f"H{dim}_mean_persistence"] = float(np.mean(lifetimes)) if len(lifetimes) else 0.0
        features[f"H{dim}_max_persistence"] = float(np.max(lifetimes)) if len(lifetimes) else 0.0
    return features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wt", required=True, help="Ruta al .pdb del individuo silvestre (g0_00)")
    ap.add_argument("--mut", required=True, help="Ruta al .pdb del mutante a comparar (p.ej. el mejor individuo)")
    ap.add_argument("--face_file", default=None, help="Archivo con los residuos de interfaz (chain resnum por linea)")
    ap.add_argument("--chain_mut", default="C")
    ap.add_argument("--chain_partner", default="A")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--atom_mode", choices=["sidechain", "ca"], default="sidechain")
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--out", default="wasserstein_wt_vs_mutant_out")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    face_residues = load_face_file(args.face_file)

    results = {}
    diagrams = {}
    for label, pdb_path in [("wildtype", args.wt), ("mutant", args.mut)]:
        coords, meta = extract_interface_coords(
            pdb_path, args.chain_mut, args.chain_partner, args.cutoff, face_residues, args.atom_mode
        )
        if coords.shape[0] < 4:
            raise SystemExit(f"{label}: muy pocos puntos de interfaz ({coords.shape[0]}) -- revisa las rutas/cadenas.")
        dgms = ripser(coords, maxdim=args.maxdim)["dgms"]
        diagrams[label] = dgms
        feat = topological_features(dgms)
        feat.update(meta)
        feat["pdb_file"] = str(pdb_path)
        results[label] = feat
        print(f"[{label}] {meta['n_interface_points']} puntos de interfaz "
              f"({meta['n_mut_chain_residues']} residuos mutados/cadena + {meta['n_partner_residues']} pareja)")

    # Distancias de Wasserstein H1 y H2 entre el silvestre y el mutante.
    w1 = float(wasserstein(finite_intervals(diagrams["wildtype"][1]),
                            finite_intervals(diagrams["mutant"][1]), matching=False))
    w2 = float(wasserstein(finite_intervals(diagrams["wildtype"][2]),
                            finite_intervals(diagrams["mutant"][2]), matching=False))
    print(f"\nDistancia de Wasserstein W1 (H1, ciclos):    {w1:.4f}")
    print(f"Distancia de Wasserstein W1 (H2, cavidades):  {w2:.4f}")

    results["wasserstein_H1"] = w1
    results["wasserstein_H2"] = w2

    with open(outdir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Tabla comparativa de descriptores lado a lado (para redactar la
    # interpretacion fisica/biologica: que dimension y que estadistico
    # cambia mas entre el silvestre y el mutante).
    print("\nComparación de descriptores (silvestre vs. mutante):")
    print(f"{'descriptor':28s} {'wildtype':>12s} {'mutant':>12s} {'delta':>12s}")
    for key in results["wildtype"]:
        if key in ("pdb_file",) or key not in results["mutant"]:
            continue
        wt_v = results["wildtype"][key]
        mut_v = results["mutant"][key]
        if isinstance(wt_v, (int, float)) and isinstance(mut_v, (int, float)):
            print(f"{key:28s} {wt_v:12.4f} {mut_v:12.4f} {mut_v - wt_v:12.4f}")

    # Diagramas de persistencia superpuestos, para inspeccion visual directa.
    plt.figure(figsize=(7, 5))
    plot_diagrams(diagrams["wildtype"], show=False, labels=[f"WT H{d}" for d in range(3)])
    plt.title("Silvestre vs. mutante: diagramas de persistencia (silvestre)")
    plt.tight_layout()
    plt.savefig(outdir / "PD_wildtype.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plot_diagrams(diagrams["mutant"], show=False, labels=[f"MUT H{d}" for d in range(3)])
    plt.title("Silvestre vs. mutante: diagramas de persistencia (mutante)")
    plt.tight_layout()
    plt.savefig(outdir / "PD_mutant.png", dpi=220)
    plt.close()

    print(f"\nListo. Resultados guardados en: {outdir}/")
    print("  - comparison_summary.json (W1, W2 y descriptores de ambas estructuras)")
    print("  - PD_wildtype.png, PD_mutant.png (diagramas de persistencia para inspeccion visual)")


if __name__ == "__main__":
    main()
