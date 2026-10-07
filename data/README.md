# Datos de resultados (sin archivos .pdb)

Esta carpeta contiene únicamente las tablas de resultados y valores numéricos usados en la tesis. Los archivos de coordenadas (`.pdb`) no se incluyen por su tamaño.

| Carpeta | Contenido |
|---|---|
| `single_objective/` | Experimento de un objetivo (1000 estructuras): descriptores topológicos + energías de Rosetta (`tda_features_with_scores.csv`), PCA y grupos k-means (`tda_clustering_pca.csv`), correlaciones de Spearman (`spearman_correlations_energy_topology.csv`), energías parseadas (`rosetta_scores_parsed.csv`), índice de calidad (`topo_quality_single_objective*.csv`) y registro completo de `score` (`single_objective_all.csv`). |
| `biobjective_run12/` | Experimento biobjetivo, corrida `run12` (750 estructuras): mismos archivos que arriba, con el índice de calidad por rango de Pareto (`topo_quality_biobj_run12*.csv`). |
| `biobjective_all_runs/` | Registro de `score` y `ll` de los 12 000 individuos de las 16 corridas (`all_individuals.csv`), versión deduplicada con marca de frente de Pareto (`bi_obj_unique_with_pareto.csv`) y mejor individuo por generación. Es la fuente del frente de Pareto de la tesis. |
| `pilot_20/` | Conjunto piloto de 20 estructuras: descriptores, PCA, Spearman y matrices de distancia de Wasserstein (`wasserstein_H1_matrix.csv`, `wasserstein_H2_matrix.csv`). |
| `wasserstein_wt_vs_mutant/` | Resultados de `wasserstein_wt_vs_mutant.py` (silvestre vs. mejor mutante) para ambos experimentos. |
| `faceC.txt` | Definición de la interfaz de unión (38 residuos de la cadena C). |

`ll` es el *log-likelihood* de la secuencia según un modelo de lenguaje de proteínas (mayor = secuencia más probable).
