#!/usr/bin/env python3
"""
TDA analysis for protein mutation interfaces.

Pipeline inspired by Xia & Wei's use of persistent homology as
multi-scale topological fingerprints for biomolecular structures:
1) parse Rosetta .pdb and .pdb.sc files,
2) extract interface point clouds,
3) compute persistent homology H0/H1/H2,
4) summarize persistence features,
5) compare mutants by Wasserstein/bottleneck-like distances,
6) relate topological descriptors to Rosetta energy terms.

Recommended install:
    pip install numpy pandas matplotlib seaborn biopython ripser persim scikit-learn scipy

Example:
    python tda_mutation_analysis.py --pdb_dir best_individuals/pdb --score_dir . --face_file faceC.txt \
      --chain_mut C --chain_partner A --cutoff 8.0 --out results_tda

If no face_file is provided, all CA atoms of chain_mut are considered possible interface residues.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from Bio.PDB import PDBParser
except ImportError as exc:
    raise SystemExit("Missing dependency: biopython. Install with: pip install biopython") from exc

try:
    from ripser import ripser
except ImportError as exc:
    raise SystemExit("Missing dependency: ripser. Install with: pip install ripser") from exc

try:
    from persim import wasserstein, plot_diagrams
    HAS_PERSIM = True
except Exception:
    HAS_PERSIM = False
    wasserstein = None
    plot_diagrams = None

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering


def safe_name(path: str | Path, base_dir: str | Path | None = None) -> str:
    """Build a unique key for a structure/score file.

    IMPORTANT: using only the bare filename (Path(path).name) causes key
    collisions when the same individual-naming convention (e.g. "g0_00")
    repeats across independent run folders (run1/g0/g0_00.pdb vs.
    run2/g0/g0_00.pdb). That collision corrupts downstream merges (many-to-many
    joins that silently duplicate/mispair rows). To avoid this, when base_dir
    is given the key is built from the path *relative to base_dir* (so it
    keeps the run/generation folder context), with path separators flattened
    to underscores so the key is also safe to use inside output filenames.
    """
    p = Path(path)
    if base_dir is not None:
        try:
            rel = p.resolve().relative_to(Path(base_dir).resolve())
        except ValueError:
            rel = Path(p.name)
    else:
        rel = Path(p.name)
    name = str(rel)
    name = re.sub(r"\.pdb$", "", name)
    name = re.sub(r"\.pdb\.sc$", "", name)
    name = re.sub(r"_0001$", "", name)
    name = re.sub(r"[\\/]+", "_", name)
    return name


def load_face_file(face_file: Optional[str]) -> Optional[set[Tuple[str, int]]]:
    if not face_file:
        return None
    residues: set[Tuple[str, int]] = set()
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


def parse_rosetta_score_file(sc_file: str | Path, base_dir: str | Path | None = None) -> Optional[Dict[str, float | str]]:
    """Parse one Rosetta .sc file with SCORE header and SCORE values."""
    header = None
    values = None
    with open(sc_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("SCORE:"):
                parts = line.split()
                # Header line contains column names; value line contains numeric values.
                if len(parts) > 2 and parts[1] == "total_score":
                    header = parts[1:]
                elif header is not None:
                    values = parts[1:]
                    break
    if header is None or values is None:
        return None
    row: Dict[str, float | str] = {}
    for key, val in zip(header, values):
        try:
            row[key] = float(val)
        except ValueError:
            row[key] = val
    row["mutation"] = safe_name(sc_file, base_dir=base_dir)
    row["score_file"] = str(sc_file)
    return row


def collect_scores(score_dir: str | Path, base_dir: str | Path | None = None) -> pd.DataFrame:
    if base_dir is None:
        base_dir = score_dir
    files = sorted(glob.glob(str(Path(score_dir) / "**" / "*.pdb.sc"), recursive=True))
    rows = [parse_rosetta_score_file(f, base_dir=base_dir) for f in files]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)


def residue_ca_coords(pdb_file: str | Path, chain_id: str) -> List[Tuple[int, np.ndarray]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_file))
    coords: List[Tuple[int, np.ndarray]] = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                # ignore waters/hetero residues unless they still have CA and normal resseq
                if "CA" not in residue:
                    continue
                res_id = int(residue.get_id()[1])
                coords.append((res_id, residue["CA"].get_coord().astype(float)))
        break  # first model only
    return coords


def residue_heavy_atom_coords(pdb_file: str | Path, chain_id: str) -> Dict[int, List[np.ndarray]]:
    """Return {residue_id: [heavy-atom coords]} for one chain (all non-hydrogen atoms).

    Used for atom_mode="sidechain": under a fixed-backbone design protocol the
    C-alpha trace is identical across every mutant (verified empirically: 0.0 A
    max C-alpha shift between the wildtype and a 13-mutation individual 19
    generations later), so a C-alpha-only point cloud is blind to mutation by
    construction. Side-chain atoms differ with residue identity even when the
    backbone never moves, so including them lets the point cloud/persistent
    homology actually respond to which mutation was made.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_file))
    out: Dict[int, List[np.ndarray]] = {}
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if "CA" not in residue:
                    continue
                res_id = int(residue.get_id()[1])
                atoms = [
                    atom.get_coord().astype(float)
                    for atom in residue
                    if atom.element not in ("H", "D")
                ]
                out[res_id] = atoms
        break  # first model only
    return out


def extract_interface_coords(
    pdb_file: str | Path,
    chain_mut: str,
    chain_partner: str,
    cutoff: float,
    face_residues: Optional[set[Tuple[str, int]]] = None,
    include_partner_close_to_mut: bool = True,
    atom_mode: str = "sidechain",
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Return point cloud for the interface.

    If face_residues is provided, use only those residues from chain_mut. Otherwise all CA in chain_mut.
    Partner chain residues are included if within cutoff of selected chain_mut CA atoms
    (residue selection is always geometry-based, on C-alpha distances, regardless of atom_mode).

    atom_mode:
      - "sidechain" (default): the point cloud for each SELECTED residue includes all its
        heavy atoms (backbone + side chain), so the topology can respond to which amino
        acid is present at each interface position, not just its (fixed) backbone location.
      - "ca": legacy behavior, one point per selected residue (C-alpha only). Kept for
        reproducing the original thesis chapter's methodology on datasets where the
        backbone genuinely moves between variants (e.g. full Rosetta relax, not fixbb).
    """
    mut_ca = residue_ca_coords(pdb_file, chain_mut)
    partner_ca = residue_ca_coords(pdb_file, chain_partner)
    mut_ca_map = dict(mut_ca)
    partner_ca_map = dict(partner_ca)

    if face_residues is not None:
        mut_selected_ids = [rid for rid, _ in mut_ca if (chain_mut, rid) in face_residues]
    else:
        mut_selected_ids = [rid for rid, _ in mut_ca]

    partner_selected_ids: List[int] = []
    if include_partner_close_to_mut and mut_selected_ids:
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
    elif atom_mode == "ca":
        mut_coords = [mut_ca_map[rid] for rid in mut_selected_ids]
        partner_coords = [partner_ca_map[rid] for rid in partner_selected_ids]
    else:
        raise ValueError(f"Unknown atom_mode: {atom_mode!r} (expected 'sidechain' or 'ca')")

    coords = mut_coords + partner_coords
    metadata = {
        "n_mut_chain_residues": len(mut_selected_ids),
        "n_partner_residues": len(partner_selected_ids),
        "n_mut_chain_points": len(mut_coords),
        "n_partner_points": len(partner_coords),
        "n_interface_points": len(coords),
    }
    if len(coords) == 0:
        return np.empty((0, 3)), metadata
    return np.vstack(coords), metadata


def compute_persistence(coords: np.ndarray, maxdim: int = 2, thresh: Optional[float] = None):
    if coords.shape[0] < 4:
        return None
    kwargs = {"maxdim": maxdim}
    if thresh is not None:
        kwargs["thresh"] = thresh
    return ripser(coords, **kwargs)["dgms"]


def finite_intervals(dgm: np.ndarray) -> np.ndarray:
    if dgm is None or len(dgm) == 0:
        return np.empty((0, 2))
    dgm = np.asarray(dgm, dtype=float)
    return dgm[np.isfinite(dgm[:, 1])]


def topological_features(dgms: Sequence[np.ndarray]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for dim in range(3):
        dgm = finite_intervals(dgms[dim]) if dim < len(dgms) else np.empty((0, 2))
        lifetimes = dgm[:, 1] - dgm[:, 0] if len(dgm) else np.array([])
        prefix = f"H{dim}"
        features[f"{prefix}_count"] = float(len(lifetimes))
        features[f"{prefix}_sum_persistence"] = float(np.sum(lifetimes)) if len(lifetimes) else 0.0
        features[f"{prefix}_mean_persistence"] = float(np.mean(lifetimes)) if len(lifetimes) else 0.0
        features[f"{prefix}_max_persistence"] = float(np.max(lifetimes)) if len(lifetimes) else 0.0
        features[f"{prefix}_birth_mean"] = float(np.mean(dgm[:, 0])) if len(dgm) else 0.0
        features[f"{prefix}_death_mean"] = float(np.mean(dgm[:, 1])) if len(dgm) else 0.0
    # Combined quantities often useful for stability/interfacial topology.
    features["H1H2_sum_persistence"] = features["H1_sum_persistence"] + features["H2_sum_persistence"]
    features["H1H2_count"] = features["H1_count"] + features["H2_count"]
    return features


def plot_pd(name: str, dgms: Sequence[np.ndarray], outdir: Path):
    plt.figure(figsize=(7, 5))
    if HAS_PERSIM:
        plot_diagrams(dgms, show=False)
    else:
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for dim, dgm in enumerate(dgms[:3]):
            d = np.asarray(dgm)
            if len(d):
                plt.scatter(d[:, 0], d[:, 1], s=18, label=f"H{dim}", color=colors[dim])
        lim = plt.xlim()[1]
        plt.plot([0, lim], [0, lim], "k--", lw=1)
        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.legend()
    plt.title(f"{name} - Interface topology")
    plt.tight_layout()
    plt.savefig(outdir / f"PD_{name}.png", dpi=220)
    plt.close()


def plot_heatmap(matrix: np.ndarray, labels: List[str], title: str, outpath: Path):
    plt.figure(figsize=(max(8, 0.45 * len(labels)), max(6, 0.45 * len(labels))))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_energy_relationships(df: pd.DataFrame, outdir: Path, energy_col: str):
    variables = [
        "H1_sum_persistence", "H1_max_persistence", "H1_count",
        "H2_sum_persistence", "H2_max_persistence", "H2_count",
        "H1H2_sum_persistence", "n_interface_points",
    ]
    # energy_col can contain characters that are illegal/meaningful in file paths
    # (e.g. "dG_separated/dSASAx100" has a literal "/"); sanitize only for filenames.
    safe_energy_col = re.sub(r"[^A-Za-z0-9_.-]+", "_", energy_col)
    for var in variables:
        if var not in df.columns or energy_col not in df.columns:
            continue
        sub = df[[energy_col, var, "mutation"]].dropna()
        if len(sub) < 3:
            continue
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(sub[energy_col], sub[var], s=50)
        for _, r in sub.iterrows():
            plt.annotate(str(r["mutation"]), (r[energy_col], r[var]), fontsize=7, alpha=0.75)
        corr = sub[energy_col].corr(sub[var], method="spearman")
        plt.xlabel(energy_col + " (more negative = lower energy)")
        plt.ylabel(var)
        plt.title(f"{var} vs {energy_col} | Spearman={corr:.2f}")
        plt.tight_layout()
        plt.savefig(outdir / f"scatter_{var}_vs_{safe_energy_col}.png", dpi=220)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", default="./best_individuals/pdb", help="Folder with relaxed .pdb files")
    ap.add_argument("--score_dir", default=".", help="Folder with Rosetta .pdb.sc files")
    ap.add_argument("--face_file", default=None, help="Optional face/interface residue file: chain residue_number")
    ap.add_argument("--chain_mut", default="C", help="Mutated/design chain ID")
    ap.add_argument("--chain_partner", default="A", help="Partner chain ID")
    ap.add_argument("--cutoff", type=float, default=8.0, help="CA-CA distance cutoff in Angstrom")
    ap.add_argument("--atom_mode", choices=["sidechain", "ca"], default="sidechain",
                     help="'sidechain' (default): point cloud includes all heavy atoms of each selected "
                          "interface residue, so topology responds to which amino acid is present -- needed "
                          "for fixed-backbone datasets where the C-alpha trace never moves between mutants. "
                          "'ca': legacy C-alpha-only point cloud (matches the original thesis chapter's "
                          "method; only meaningful if the backbone actually moves between variants).")
    ap.add_argument("--maxdim", type=int, default=2)
    ap.add_argument("--thresh", type=float, default=None, help="Optional max filtration threshold for ripser")
    ap.add_argument("--energy_col", default="total_score", help="Rosetta score column used for energy plots")
    ap.add_argument("--top_n", type=int, default=20, help="Analyze only top N lowest-energy structures if scores exist; 0 = all")
    ap.add_argument("--n_clusters", type=int, default=3)
    ap.add_argument("--max_wasserstein_n", type=int, default=200,
                     help="Skip the full pairwise Wasserstein distance matrix if more than this many "
                          "structures are being analyzed (O(n^2) persim.wasserstein calls get very slow "
                          "for large n). Feature-based correlation/PCA/clustering still run on all of them.")
    ap.add_argument("--out", default="tda_results")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "persistence_diagrams").mkdir(exist_ok=True)

    scores = collect_scores(args.score_dir)
    if not scores.empty:
        scores.to_csv(outdir / "rosetta_scores_parsed.csv", index=False)
        print(f"Parsed {len(scores)} score files")
    else:
        print("No .pdb.sc score files found or parsed")

    pdb_files = sorted(glob.glob(str(Path(args.pdb_dir) / "**" / "*.pdb"), recursive=True))
    if not pdb_files:
        raise SystemExit(f"No .pdb files found in {args.pdb_dir}. Put relaxed PDBs there or change --pdb_dir.")

    pdb_df = pd.DataFrame({"pdb_file": pdb_files})
    # base_dir=args.pdb_dir keeps run/generation folder context in the key so
    # individuals from different run folders (e.g. run1/g0/g0_00.pdb vs.
    # run2/g0/g0_00.pdb) don't collide into the same "mutation" key.
    pdb_df["mutation"] = pdb_df["pdb_file"].map(lambda p: safe_name(p, base_dir=args.pdb_dir))

    if not scores.empty:
        use_cols = ["mutation"] + [c for c in scores.columns if c != "mutation"]
        pdb_df = pdb_df.merge(scores[use_cols], on="mutation", how="left")
        if args.energy_col in pdb_df.columns and args.top_n and args.top_n > 0:
            pdb_df = pdb_df.sort_values(args.energy_col, ascending=True).head(args.top_n).reset_index(drop=True)

    face_residues = load_face_file(args.face_file)
    if face_residues is not None:
        print(f"Loaded {len(face_residues)} face residues from {args.face_file}")
    else:
        print("No face file provided: using all CA atoms from mutated chain as interface seeds")

    diagrams: Dict[str, Sequence[np.ndarray]] = {}
    rows: List[Dict[str, float | str]] = []

    for _, row in pdb_df.iterrows():
        name = row["mutation"]
        pdb = row["pdb_file"]
        coords, meta = extract_interface_coords(
            pdb,
            chain_mut=args.chain_mut,
            chain_partner=args.chain_partner,
            cutoff=args.cutoff,
            face_residues=face_residues,
            atom_mode=args.atom_mode,
        )
        if coords.shape[0] < 4:
            print(f"Skipping {name}: only {coords.shape[0]} interface points")
            continue
        dgms = compute_persistence(coords, maxdim=args.maxdim, thresh=args.thresh)
        if dgms is None:
            continue
        diagrams[name] = dgms
        feat = topological_features(dgms)
        feat.update(meta)
        feat["mutation"] = name
        feat["pdb_file"] = pdb
        rows.append(feat)
        plot_pd(name, dgms, outdir / "persistence_diagrams")
        print(f"Processed {name}: {meta['n_interface_points']} interface points "
              f"({meta['n_mut_chain_residues']} mut-chain + {meta['n_partner_residues']} partner residues, "
              f"atom_mode={args.atom_mode})")

    features = pd.DataFrame(rows)
    if features.empty:
        raise SystemExit("No valid PDBs were processed. Check chains, face_file, and cutoff.")

    # Merge scores + features.
    result = features.merge(scores, on="mutation", how="left", suffixes=("", "_score")) if not scores.empty else features
    result.to_csv(outdir / "tda_features_with_scores.csv", index=False)

    labels = list(features["mutation"])
    n = len(labels)

    # Wasserstein matrices by dimension.
    if HAS_PERSIM and n >= 2 and n > args.max_wasserstein_n:
        print(f"Skipping pairwise Wasserstein matrices: {n} structures > --max_wasserstein_n="
              f"{args.max_wasserstein_n} (O(n^2) would be too slow). "
              f"Feature-based correlation/PCA/clustering below are unaffected.")
    if HAS_PERSIM and n >= 2 and n <= args.max_wasserstein_n:
        for dim in [1, 2]:
            D = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    di = finite_intervals(diagrams[labels[i]][dim])
                    dj = finite_intervals(diagrams[labels[j]][dim])
                    try:
                        val = float(wasserstein(di, dj, matching=False))
                    except Exception:
                        val = np.nan
                    D[i, j] = D[j, i] = val
            pd.DataFrame(D, index=labels, columns=labels).to_csv(outdir / f"wasserstein_H{dim}_matrix.csv")
            plot_heatmap(D, labels, f"Wasserstein distance matrix H{dim}", outdir / f"heatmap_wasserstein_H{dim}.png")

    # Feature-based clustering/PCA.
    feature_cols = [c for c in features.columns if re.match(r"H[012]_", c) or c in ["H1H2_sum_persistence", "H1H2_count", "n_interface_points"]]
    X = features[feature_cols].fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42).fit_transform(Xs)
    features["PC1"] = pca[:, 0]
    features["PC2"] = pca[:, 1]
    k = min(args.n_clusters, n)
    if k >= 2:
        features["cluster_kmeans"] = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xs)
    else:
        features["cluster_kmeans"] = 0
    features[["mutation", "PC1", "PC2", "cluster_kmeans"] + feature_cols].to_csv(outdir / "tda_clustering_pca.csv", index=False)

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(features["PC1"], features["PC2"], c=features["cluster_kmeans"], s=60)
    for _, r in features.iterrows():
        plt.annotate(r["mutation"], (r["PC1"], r["PC2"]), fontsize=7, alpha=0.8)
    plt.xlabel("PC1 (topological features)")
    plt.ylabel("PC2 (topological features)")
    plt.title("Mutants grouped by persistent-homology features")
    plt.tight_layout()
    plt.savefig(outdir / "pca_topological_features.png", dpi=220)
    plt.close()

    if args.energy_col in result.columns:
        plot_energy_relationships(result, outdir, args.energy_col)
        numeric_cols = [args.energy_col] + feature_cols
        corr = result[numeric_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
        corr.to_csv(outdir / "spearman_correlations_energy_topology.csv")

    print("\nDONE. Main outputs:")
    print(f"  {outdir / 'tda_features_with_scores.csv'}")
    print(f"  {outdir / 'persistence_diagrams'}")
    print(f"  {outdir / 'pca_topological_features.png'}")
    print(f"  {outdir / 'spearman_correlations_energy_topology.csv'}")


if __name__ == "__main__":
    main()
