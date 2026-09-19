# Audited Results Summary

> **Scope:** Provisional research results. The universe is supported by the PA-001 Yahoo-based
> foundation, not by licensed point-in-time S&P/GICS reconciliation. These artifacts summarize
> existing audited outputs and do not introduce new inference or refit any model.

## Core findings

- **H1 — not supported.** The mean hierarchical-minus-GICS dependence-gap difference is
  -0.005590; its paired circular block-bootstrap 95% interval
  is [-0.018669, 0.006569]. The interval is not strictly
  positive.
- **H2 — no support.** 0 of
  6 Holm-adjusted matched DM tests favour the vine.
- **H3 — supported under the frozen ranking rule.** `M4` is the unique
  best eligible core model, with average primary-score rank
  1.333. This is a relative forecast-loss ranking, not a
  claim that vines dominate Gaussian copulas in the H2 pairwise tests.

## Exploratory ML extension

Spectral clustering and PCA-plus-k-means do not change the core conclusions. Across the frozen
24-test family, 0
comparisons favour an ML grouping and 0 favour a
baseline grouping after Benjamini-Hochberg adjustment.

## Robustness boundary

- `full_vine_tree_10`: no adjusted loss difference; core conclusions revised = false.
- `ml_groupings`: no fdr adjusted loss difference; core conclusions revised = false.
- `group_balanced_portfolios`: no holm adjusted difference; core conclusions revised = false.
- `wba_dap_valuation`: all frozen conclusions stable; core conclusions revised = false.
- `current_composition_historical_simulation`: partial realised history favoured; core conclusions revised = false.

## Artifact guide

- `tables/core_model_performance.csv` and `core_model_score_ranks.svg` report the frozen M0-M4
  loss ranking.
- `tables/core_dm_tests.csv` and `tables/core_calibration.csv` preserve the adjusted inferential
  results and full-period calibration diagnostics.
- `tables/clustering_diagnostics.csv` and `annual_dependence_gaps.svg` report annual grouping
  separation for the core and exploratory clustering methods.
- `report_manifest.json` binds every table, figure and this summary to the exact audited inputs.
