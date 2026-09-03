# Análisis topológico (TDA) de efectos de mutación en interacciones proteína-proteína

Código utilizado en la tesis de maestría *"Análisis de los efectos de mutación en modelos
de interacción proteína-proteína utilizando Topological Data Analysis (TDA)"*
(Maestría en Ciencia de Datos, Yachay Tech). El pipeline usa homología persistente
(complejos de Vietoris-Rips) para caracterizar la topología de la interfaz de unión de la
Receptor Binding Protein (RBP) del bacteriófago φRs551 a lo largo de series evolutivas de
mutaciones acumuladas, y relaciona los descriptores topológicos resultantes con la energía
de unión estimada por Rosetta.

## Contenido

| Script | Descripción |
|---|---|
| `tda_mutation_analysis.py` | Pipeline original de homología persistente (representación por carbono alfa, C$\alpha$), usado en la fase de desarrollo y verificación sobre el conjunto inicial de estructuras. |
| `tda_mutation_analysis_v2.py` | Versión extendida del pipeline, con el parámetro `--atom_mode` (`ca` / `sidechain`), usada para los dos experimentos evolutivos principales (backbone fijo, ver Sección 4.5.2 de la tesis). |
| `topo_quality_index_v2.py` | Cálculo del índice de calidad de la mutación (rango de `score` para un objetivo; rango de no-dominancia de Pareto para el biobjetivo) y su cruce con los descriptores topológicos. |
| `wasserstein_wt_vs_mutant.py` | Comparación dirigida entre dos estructuras puntuales (p. ej., el individuo silvestre y el mejor mutante): calcula la distancia de Wasserstein entre sus diagramas de persistencia $H_1$/$H_2$ y exporta una tabla comparativa de descriptores para el análisis interpretativo físico/biológico. |
| `verify_rerun.py` | Script de verificación end-to-end: compara, columna por columna, los resultados de una reejecución completa del pipeline contra los resultados originales reportados en la tesis. |

## Requisitos

```bash
pip install numpy pandas matplotlib biopython ripser persim scikit-learn scipy
```

## Uso básico

```bash
# Pipeline extendido sobre un conjunto de estructuras .pdb
python tda_mutation_analysis_v2.py \
    --pdb_dir ruta/a/estructuras \
    --score_dir ruta/a/puntuaciones_rosetta \
    --face_file faceC.txt \
    --chain_mut C --chain_partner A --cutoff 8.0 \
    --atom_mode sidechain \
    --out resultados_tda

# Distancia de Wasserstein entre el individuo silvestre y un mutante puntual
python wasserstein_wt_vs_mutant.py \
    --wt ruta/g0_00.pdb \
    --mut ruta/g19_24.pdb \
    --face_file faceC.txt \
    --out resultados_wasserstein
```

Cada script acepta `--help` para ver todos los parámetros disponibles.

## Datos

Este repositorio contiene únicamente el código de análisis, no las estructuras `.pdb` ni
los archivos de puntuación de Rosetta, que provienen del algoritmo evolutivo descrito en
Armas et al. (2026) y no se distribuyen aquí.

## Cita

Si este código es de utilidad, por favor cita la tesis correspondiente:

> Garzón Pardo, A. M. (2026). *Análisis de los efectos de mutación en modelos de
> interacción proteína-proteína utilizando Topology Data Analysis (TDA)*. Tesis de
> Maestría en Ciencia de Datos, Universidad Yachay Tech.

## Licencia

MIT License (ver `LICENSE`).
