# Multimodal Mental Health Analysis

This repository contains the analysis notebooks and implementation for the manuscript titled `When words and voice disagree: a quantum-inspired measure of semantic–affective discordance for multimodal depression screening` using the three datasets.

## Repository Structure

- `e_daic/` — Analysis and modelling using E-DAIC.
- `e_risk/` — Analysis and modelling using eRisk.
- `iemocap/` — Analysis and modelling using IEMOCAP.
- `notebooks/` (within each dataset folder) — Notebooks for preprocessing, data splitting, feature analysis, and evaluation.
- Python scripts — Model definitions and training/testing pipelines.

## Data Availability

**Datasets and processed data files are not included.** E-DAIC, eRisk, and IEMOCAP must be obtained separately from their respective sources, subject to their access conditions and licensing requirements.

Processed files, including train/test PKL files, are excluded. Local dataset paths and machine-specific configurations are also omitted.

## Reproducibility

The notebooks document the analysis workflow, while the Python scripts provide the implementation for model training, testing, and evaluation. Execution requires access to the relevant dataset and appropriate environment configuration.

## Scope

This repository supports research and reproducibility. The implemented models are intended for research-based mental health risk analysis, not clinical diagnosis.
