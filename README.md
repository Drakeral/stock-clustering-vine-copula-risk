# FE5110 Stock-Clustering and Vine-Risk Project

[![Quality checks](https://github.com/Drakeral/stock-clustering-vine-copula-risk/actions/workflows/quality.yml/badge.svg)](https://github.com/Drakeral/stock-clustering-vine-copula-risk/actions/workflows/quality.yml)

This repository implements the reproducible research pipeline for the fixed,
point-in-time S&P 100 universe described in `FE5110_Project_Plan.md`.

## Current gate status

`foundation_v2` has a **provisional pass** under protocol amendment PA-001. The
independent market-input, strengthened data-construction, arithmetic, and method
gates pass. The universe is the frozen contemporaneous Wikipedia snapshot,
supported by a Yahoo metadata and 2 January 2020 price-presence cross-check.
The live, separated statuses are in `data/audit/current_gate_status.json`.

The exact MediaWiki revision payloads are archived and hashed under
`data/raw/universe_sources/`. Yahoo does not establish historical index
membership, point-in-time GICS, or permanent identifiers. Consequently,
modelling may proceed, but every downstream result must be labelled
`provisional_research_results`; neither `foundation_v2=pass` nor licensed or
confirmatory provenance may be claimed.

The exploratory ML grouping gate also passes. Annual spectral clustering and
PCA-plus-k-means assignments and return panels have been generated under the
same leakage-free windows and 11-group portfolio arithmetic as the core study.
M5-M8 then apply the same marginal, Gaussian/vine, simulation, and risk-scoring
protocol to those groupings. Their comparisons remain exploratory and cannot
revise the core H1-H3 conclusions or model ranking.

The frozen WBA DAP valuation sensitivity also passes. Valuing the right at zero
or its `$3` cap leaves the primary H2/H3 decisions, M4's first-place rank, all
95%/97.5%/99% exception counts, and the exploratory ML evidence classification
unchanged. H1 is deliberately not retested because annual assignments are held
fixed. Exact scenario diagnostics and hashes are in
`data/audit/wba_dap_sensitivity.json`.

The machine-readable authorization and its narrow conditions are frozen in
`config/provenance_amendment.json`. A licensed point-in-time extract can still
supersede the amendment and obtain the strict provenance pass later.

## Rebuild from a clean clone

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required. The
checked-in `.python-version` selects Python 3.12 for the canonical local run,
while CI exercises both supported minor versions.
The repeatable lint, test and repository-hygiene checkpoint is documented in
`docs/code_quality.md`. The latest holistic research-code review, remediations,
and interpretation boundaries are recorded in `docs/quantitative_code_audit.md`.

```zsh
uv sync --frozen
mkdir -p data/raw/universe_sources data/raw/licensed
```

The tracked public revision payloads can be independently re-fetched with the
following commands. They are secondary sources, not licensed corroboration:

```zsh
curl --fail --silent --show-error --get \
  'https://en.wikipedia.org/w/api.php' \
  --data-urlencode 'action=query' \
  --data-urlencode 'prop=revisions' \
  --data-urlencode 'revids=929329274' \
  --data-urlencode 'rvprop=ids|timestamp|content' \
  --data-urlencode 'rvslots=main' \
  --data-urlencode 'format=json' \
  --data-urlencode 'formatversion=2' \
  --output data/raw/universe_sources/sp100_oldid_929329274.json

curl --fail --silent --show-error --get \
  'https://en.wikipedia.org/w/api.php' \
  --data-urlencode 'action=query' \
  --data-urlencode 'prop=revisions' \
  --data-urlencode 'revids=933762580' \
  --data-urlencode 'rvprop=ids|timestamp|content' \
  --data-urlencode 'rvslots=main' \
  --data-urlencode 'format=json' \
  --data-urlencode 'formatversion=2' \
  --output data/raw/universe_sources/sp500_oldid_933762580.json
```

The following licensed route is optional under PA-001, but remains the route to
a strict, non-provisional provenance pass. Place the authorised CSV at
`data/raw/licensed/universe_sp100_2020-01-02.csv`, using the exact header in
`config/licensed_universe_template.csv`. Keep the raw file untracked and retain
the licence or entitlement reference outside Git. The reference passed below is
hashed before it enters the public manifest; use a non-confidential local label,
not a credential. The approved sourcing routes and exact access-request template
are documented in `docs/licensed_universe_acquisition.md`. Then prepare and
reconcile the universe:

To populate the normalized file safely, first create the ignored working sheet:

```zsh
uv run python scripts/populate_licensed_universe.py create
```

Open `data/interim/licensed/universe_population_worksheet.csv`. The five
`reference_*` columns are lookup/reconciliation aids from the archived public
snapshot. Populate every `licensed_*` column exclusively from the authorized
point-in-time extract; in particular, use the provider's security, issuer, and
source-record identifiers and the provider's actual membership date. Do not copy
the synthetic `SP100-20200102:*`, `ISSUER:*`, or `wikipedia:*` identifiers.
After entry, validate and generate the exact gate input:

```zsh
uv run python scripts/populate_licensed_universe.py finalize
```

Finalization refuses incomplete rows, candidate identifiers, invalid dates,
duplicate identifiers, unsupported sectors, or any ticker/GICS/as-of mismatch.
The working sheet and final licensed CSV both remain ignored by Git.

```zsh
uv run python scripts/prepare_universe.py \
  --sp100-json data/raw/universe_sources/sp100_oldid_929329274.json \
  --sp500-json data/raw/universe_sources/sp500_oldid_933762580.json \
  --licensed-universe-csv data/raw/licensed/universe_sp100_2020-01-02.csv \
  --licensed-source-name '<authorised provider and dataset>' \
  --licensed-license-reference '<local entitlement or contract reference>' \
  --output data/raw/universe_sp100_2020-01-02.json \
  --metadata-output data/manifests/universe_source_manifest.json \
  --provenance-audit-output data/audit/universe_provenance.json
```

The public candidate remains in `data/raw/universe_sp100_2020-01-02.json` and
never receives vendor row values. Normalized licensed rows and full mismatch
details are written under ignored `data/interim/licensed/` paths.

Without licensed data, omit the three `--licensed-*` options. The command
reproduces the current secondary-source candidate and records a blocked
provenance gate rather than marking the candidate approved.

Build the required Yahoo-supported provisional reference:

```zsh
uv run python scripts/build_yahoo_universe_reference.py
```

This writes an ignored row-level comparison table to
`data/interim/provisional/yahoo_universe_reference.csv`, caches the raw Yahoo
responses under `data/raw/yahoo/`, and writes the sanitized, tracked manifest
`data/manifests/yahoo_universe_reference_manifest.json`. It does not populate or
replace `data/raw/licensed/universe_sp100_2020-01-02.csv`. Under PA-001 it can
support only the explicitly provisional modelling-readiness status.

Load credentials, download immutable provider data, and independently verify
every input against an XNYS session calendar:

```zsh
source scripts/load_credentials.zsh
uv run python scripts/download_market_data.py --workers 8
uv run python scripts/verify_foundation_inputs.py
uv run python scripts/build_return_panel.py
uv run python scripts/validate_portfolio_arithmetic.py
uv run python scripts/update_foundation_status.py --require-modelling-ready
uv run python scripts/build_historical_simulation.py
uv run python scripts/build_annual_groupings.py
uv run python scripts/build_marginal_models.py
uv run python scripts/build_gaussian_copula.py
uv run python scripts/build_vine_copula.py
uv run python scripts/evaluate_risk_models.py
uv run python scripts/build_vine_copula.py --truncation-level 10
uv run python scripts/evaluate_full_vine_robustness.py
uv run python scripts/build_ml_groupings.py
uv run python scripts/build_ml_marginal_models.py
uv run python scripts/build_ml_gaussian_copula.py
uv run python scripts/build_ml_vine_copula.py
uv run python scripts/evaluate_ml_risk_models.py
uv run python scripts/evaluate_wba_dap_sensitivity.py
uv run python scripts/build_group_balanced_portfolios.py
uv run python scripts/evaluate_group_balanced_robustness.py
uv run ruff check scripts tests
uv run ruff format --check scripts tests
uv run python -m unittest discover -s tests -v
```

The download is resumable. Existing daily and reference files are schema-
validated and hashed rather than overwritten. Use `--overwrite` only for an
intentional provider refresh. The verifier recomputes every hash and row count;
it does not infer sessions from the provider file list. The default verifier can
enrich this working copy's legacy operational manifest in memory by independently
inspecting every file; it does not mutate the ignored source manifest. A failed
verification updates the gate audit but cannot replace the last valid public
manifest. A fresh downloader run emits the strict v2 operational manifest.

Immediately before any clustering or risk-model command, require the aggregate
gate explicitly:

```zsh
uv run python scripts/update_foundation_status.py --require-modelling-ready
```

This accepts either strict `pass` or PA-001 `pass_provisional` modelling
readiness. Use `--require-pass` when a strict licensed-provenance pass is
required; it intentionally exits with status 2 while the licensed extract is
absent.

Finally capture the exact code/environment/configuration/input/output identity:

```zsh
uv run python scripts/generate_run_manifest.py --require-complete
uv run python scripts/verify_artifact_lineage.py
```

## Reproducibility and licensing boundary

- Track code, configuration, tests, audit statuses, and sanitised manifests in
  `data/manifests/`.
- Do not track credentials, `data/raw/licensed/`, `data/raw/market/`, or derived
  Parquet files. Whether normalized vendor fields may be submitted must be
  checked against the applicable licence.
- The historical access audit remains unchanged. New gates supersede its stale
  `next_gate` text through `data/audit/current_gate_status.json`.
- Under PA-001, modelling may proceed with reporting scope
  `provisional_research_results`. All tables, plots, narrative findings, and run
  manifests must preserve that label and must not claim licensed or
  confirmatory universe provenance.
- Detailed provider-event reconciliation is written only to the ignored
  `data/audit/data_quality_report.local.json`; its public report contains counts
  and a content hash, not licensed event rows.
- Build a submission bundle with `git archive HEAD`, never by zipping the live
  working folder, which contains ignored licensed files and credentials.

## License

No open-source license is granted at this time. This repository is public for
portfolio and review purposes; all rights remain reserved unless a license is
added later.

## Main outputs

- `data/manifests/universe_source_manifest.json`: source revisions, hashes,
  licences, and licensed reconciliation status;
- `data/manifests/market_input_manifest.json`: verified daily/reference hashes,
  schemas, row counts, and XNYS coverage;
- `data/processed/security_daily.parquet`: long-form prices, actions, and returns;
- `data/processed/daily_simple_total_returns.parquet` and
  `daily_total_returns.parquet`: simple- and log-return stock panels;
- `data/processed/portfolio_constituent_simple_returns.parquet` and
  `portfolio_constituent_returns.parquet`: daily-rebalanced modelling inputs;
- `data/processed/historical_simulation_risk_forecasts.parquet`: daily M0 rolling
  historical-simulation VaR, ES, realised return, and loss;
- `data/processed/historical_simulation_windows.parquet`: exact leakage-free M0
  training bounds and observation counts for every forecast date;
- `data/processed/annual_group_assignments.json`: leakage-free annual GICS and
  hierarchical-cluster membership for 2020–2025;
- `data/processed/annual_group_returns.parquet`: simple and log group returns,
  group sizes, and portfolio weights across both training and evaluation windows
  for the security-level primary universe and GOOG/GOOGL issuer-deduplicated
  robustness variant;
- `data/processed/ml_annual_group_assignments.json`: leakage-free annual
  spectral-cluster and PCA-plus-k-means memberships for 2020–2025;
- `data/processed/ml_annual_group_returns.parquet`: training and evaluation
  group returns for both exploratory ML groupings, using the same annual active
  sets, 11-group rule, and daily-rebalanced group-size arithmetic as the core
  analysis;
- `data/processed/ml_marginal_refits.parquet`,
  `ml_marginal_daily_forecasts.parquet`, and
  `ml_monthly_copula_training_pits.parquet`: isolated M5-M8 marginal fits,
  daily states, and monthly copula-training PITs;
- `data/processed/ml_gaussian_copula_refits.parquet` and
  `ml_gaussian_risk_forecasts.parquet`: monthly M5/M7 Gaussian dependence fits
  and daily risk forecasts using the primary common-random-number manifest;
- `data/processed/ml_vine_copula_refits.parquet` and
  `ml_vine_risk_forecasts.parquet`: monthly M6/M8 tree-3 vine fits and daily
  forecasts with explicit pair and whole-vine fallback records;
- `data/processed/ml_risk_evaluation_daily.parquet`: matched exploratory M5-M8
  quantile losses, FZ0 scores, VaR exceptions, and diagnostics;
- `data/processed/wba_dap_sensitivity/{dap_zero,dap_cap}/`: isolated M0-M8
  daily scores and marginal, Gaussian, and tree-3 vine refits after valuing the
  WBA DAP right at zero or its `$3` cap; the `$0.53` primary artifacts are never
  overwritten;
- `data/processed/marginal_refits.parquet`: one record per primary-universe
  group-month, including its leakage-free training bounds, selected marginal
  specification, parameters, convergence diagnostics, and complete fallback log;
- `data/processed/marginal_daily_forecasts.parquet`: daily one-step conditional
  group means, variances, standardized residuals, and clipped out-of-sample
  probability integral transforms for forecast diagnostics;
- `data/processed/monthly_copula_training_pits.parquet`: aligned, leakage-free
  in-sample PIT matrices for each monthly GICS and hierarchical copula refit; and
- `data/processed/gaussian_copula_refits.parquet`: fitted monthly M1/M3 Gaussian
  dependence matrices and numerical diagnostics;
- `data/processed/gaussian_risk_forecasts.parquet`: daily M1/M3 portfolio VaR,
  ES, realised loss, and matched copula log scores;
- `data/processed/vine_copula_refits.parquet`: fitted monthly M2/M4 truncated
  R-vine structures, pair-family selections, parameters, fit diagnostics, and
  explicit pair/whole-vine fallback records;
- `data/processed/vine_risk_forecasts.parquet`: daily M2/M4 portfolio VaR, ES,
  realised loss, matched vine log scores, and fallback flags;
- `data/processed/risk_evaluation_daily.parquet`: matched M0-M4 daily quantile
  losses, FZ0 scores, VaR exceptions, realised losses, and model diagnostics;
- `data/processed/full_vine_copula_refits.parquet` and
  `full_vine_risk_forecasts.parquet`: isolated tree-10 M2/M4 robustness fits and
  daily forecasts using the same monthly information sets and random uniforms as
  the primary tree-3 vines;
- `data/processed/full_vine_robustness_daily.parquet`: matched tree-3/tree-10
  scores, exceptions, and diagnostics for all four vine variants;
- `data/processed/group_balanced_group_returns.parquet` and
  `group_balanced_portfolio_returns.parquet`: annually reset GICS-balanced and
  hierarchical-cluster-balanced group and portfolio returns, with pre-return
  weights drifting between annual rebalances;
- `data/processed/group_balanced_robustness/`: isolated historical-simulation,
  marginal, Gaussian, tree-3 vine, forecast, and daily-score artifacts for the
  two group-balanced portfolio targets;
- `data/manifests/simulation_seed_manifest.json`: reproducible monthly common-
  random-number seeds and uniform-matrix hashes; and
- `data/manifests/inference_seed_manifest.json`: the H1 block-bootstrap seed,
  accepted-index hash, and deterministic degenerate-resample dispositions;
- `data/manifests/ml_clustering_seed_manifest.json`: deterministic annual
  k-means seeds, selected restarts, embedding hashes, and assignment hashes;
- `data/audit/clustering_diagnostics.json`: dependence gaps, ARI, NMI, pair
  counts, portfolio-identity errors, and hashes of the grouping artifacts;
- `data/audit/ml_clustering_diagnostics.json`: exploratory out-of-sample
  separation and stability diagnostics, dimensionality choices, portfolio
  identities, and bound ML-grouping output hashes;
- `data/audit/ml_marginal_model_quality.json`,
  `ml_gaussian_copula_quality.json`, and `ml_vine_copula_quality.json`: isolated
  M5-M8 fit, fallback, simulation, coverage, and bound-output checks;
- `data/audit/ml_model_evaluation.json`: the frozen 24-test exploratory DM
  family with Benjamini-Hochberg adjustment, descriptive calibration, matched
  log scores, and the explicit boundary that core H1-H3 are unchanged;
- `data/audit/wba_dap_sensitivity.json`: formula and one-cell mutation checks,
  four-group portfolio identities, model-quality diagnostics, primary-versus-
  scenario forecast changes, and conclusion-stability results for both WBA DAP
  values;
- `data/audit/historical_simulation_quality.json`: M0 window, forecast coverage,
  arithmetic, upstream lineage, and output-hash checks;
- `data/audit/marginal_model_quality.json`: marginal coverage, fallback incidence,
  PIT bounds, output hashes, and the frozen 1% EWMA quality gate;
- `data/audit/gaussian_copula_quality.json`: M1/M3 fit, simulation, forecast-
  coverage, portfolio-identity, and numerical-stability checks;
- `data/audit/vine_copula_quality.json`: M2/M4 truncation, family-selection,
  failed-pair, whole-vine fallback, forecast-coverage, and output-hash checks;
- `data/audit/model_evaluation.json`: model summaries, full-period and annual
  calibration diagnostics, six adjusted DM comparisons, H1 bootstrap inference,
  and the frozen H1-H3 decisions;
- `data/audit/full_vine_robustness_quality.json`: tree-10 fit, pair-family,
  fallback, forecast-coverage, and bound-output checks;
- `data/audit/full_vine_robustness_evaluation.json`: refit/forecast identity
  checks, complexity summaries, the frozen 16-test calibration family, six
  tree-10-minus-tree-3 DM comparisons, and matched descriptive log scores;
- `data/audit/group_balanced_portfolio_construction.json`: annual reset dates,
  drifting-weight bounds, exact stock-to-group portfolio identities, lineage,
  and construction output hashes;
- `data/audit/group_balanced_robustness.json`: six-model quality diagnostics,
  a 24-test calibration family, six within-portfolio vine-minus-Gaussian DM
  comparisons, and separate rankings for the two realised portfolio targets;
- `data/audit/`: separate provenance, input-integrity, construction, and current
  readiness reports.
