# Quantitative Code Audit

Audit date: 12 September 2026

## Conclusion

The implemented core pipeline is fit to continue as an academic research
project. Its principal quantitative invariants are explicit, tested, and
preserved across the generated artifacts. No evidence of look-ahead leakage,
loss-sign reversal, log-return portfolio averaging, inconsistent monthly random
draws, or silent programming-error fallback was found.

This conclusion is not a production-trading certification. All model outputs
remain `provisional_research_results` because the Yahoo-supported universe does
not satisfy the licensed point-in-time provenance requirement.

## Scope reviewed

The review covered universe construction and provenance, market-input
verification, lifecycle and corporate-action handling, return construction,
portfolio arithmetic, annual grouping construction, marginal filtering,
Gaussian and R-vine copulas, Monte Carlo risk forecasts, statistical inference,
artifact schemas and hashes, tests, CI, dependency locking, secret boundaries,
and repository hygiene.

## Quantitative invariants confirmed

- Stock log returns are converted to simple returns before within-group and
  portfolio aggregation. Direct stock-level and group-size-weighted daily
  portfolio returns agree to the frozen `1e-12` tolerance.
- Annual clustering uses only observations available before the evaluation
  year. Monthly marginal and copula refits use left-closed, right-open training
  windows ending before the forecast date.
- Marginal parameters remain fixed within a refit month while conditional state
  updates use only returns already observed.
- Gaussian and matched vine models use the same stored monthly uniform draws.
  Simulated group log returns are converted back to simple returns before
  portfolio loss aggregation.
- Realised loss is the negative of the daily-rebalanced equal-weight simple
  portfolio return. VaR forecasts use the positive-loss upper-tail convention.
- The historical-simulation estimator, FZ0 score, Newey-West DM comparison,
  Holm corrections, and within-year circular block bootstrap follow the frozen
  configuration.
- Missing sessions, reviewed halts, terminal cash positions, corporate-action
  dispositions, extreme returns, active-set changes, and the DOW lineage are
  enforced by fail-closed construction checks.

## Remediations made during this audit

- The GOOG/GOOGL issuer robustness portfolio now propagates a missing share-
  class return instead of silently reallocating the nominal 50/50 position to
  the observed class. Composite-name collisions are rejected.
- A marginal fit with non-finite likelihood, AIC, or BIC is now rejected.
  Non-finite diagnostics and parameters are represented as JSON `null` in the
  rejected-attempt record, allowing the deterministic fallback to continue
  without producing non-standard JSON.
- Cached Yahoo reference rebuilds preserve the original retrieval timestamp
  when neither the source payload nor the derived table changed. A genuinely
  fetched payload receives a new timestamp.
- The input verifier can safely enrich the legacy operational manifest from
  independently inspected files. A blocked verification no longer overwrites
  the last valid public input manifest.
- Foundation and modelling artifacts now share unique, atomic JSON, text, and
  Parquet writers. This prevents an interrupted build from replacing a valid
  output with a partial file while preserving the existing file formats.
- Evaluation loop variables were clarified to remove accidental reassignment
  without changing calculations.

## Verification evidence

The local checkpoint includes Ruff lint and format checks, source compilation,
the complete unit and artifact-bound test suite, a fresh return-panel build,
portfolio-arithmetic validation, and the aggregate modelling-readiness gate.
The return rebuild reproduced the tracked public construction audit without a
content change. The foundation remains `provisional_pass`; all implemented
construction and modelling quality gates pass.

Current production artifacts contain 1,508 evaluation dates and 7,540 matched
daily model records. Realised portfolio losses agree across all five models to
within `5.56e-17`. All 1,580 accepted GARCH-family refits have finite likelihood,
AIC, and BIC values. The four EWMA refits remain below the frozen one-percent
quality limit. All 144 primary vine refits completed without whole-vine
fallback, and all eight tabular output schemas match their generated columns.

## Interpretation and remaining boundaries

- Licensed point-in-time S&P 100 membership, historical GICS, and permanent
  identifiers remain absent. Yahoo cannot close this provenance gap.
- H1 is not supported: the confidence interval for the cluster-minus-GICS
  dependence gap includes zero. H2 has no confirmatory support: none of the six
  vine-minus-Gaussian loss comparisons is significant after Holm adjustment.
- M4 ranks first and passes the frozen H3 calibration restriction, but this is a
  provisional model-ranking result, not evidence that vines dominate Gaussian
  copulas or that clustering creates return predictability.
- With 1,508 evaluation dates, the 99% VaR analysis has only about 15 expected
  exceptions. Forecast losses and matched comparisons should carry more weight
  than annual 99% coverage p-values.
- The configured full ten-tree vine robustness run, secondary group-balanced
  portfolios, spectral clustering, and PCA-plus-k-means extension remain future
  stages. They must not be described as implemented results.
- Several mature ingestion and orchestration functions remain long and complex.
  Their behavior is well covered at important boundaries, but further
  decomposition should be incremental and paired with characterization tests to
  avoid changing the frozen research design.

## Release rule

Before presenting new results, rerun the checkpoint in `docs/code_quality.md`,
confirm a clean Git state and complete run manifest, and retain the provisional
label until an authorized universe extract passes strict reconciliation.
