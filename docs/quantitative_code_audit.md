# Quantitative Code Audit

Audit date: 15 September 2026

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
the full ten-tree vine robustness analysis, artifact schemas and hashes, tests,
CI, dependency locking, secret boundaries, and repository hygiene.

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

- Two stale transitive bindings were identified despite all existing
  production-artifact tests passing: the clustering audit referenced an older
  model configuration, and the model-evaluation audit referenced an older
  primary-vine audit. A recursive repository lineage verifier now checks every
  `path`/`sha256` record in audit and manifest JSON documents. Modelling entry
  points also reject stale hash records in each upstream audit before loading
  numerical inputs, and run-manifest completeness now requires a clean lineage
  result.
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
- Licensed-universe worksheets, validation copies, and final CSV gate inputs now
  use the same atomic-write discipline; an interrupted population step cannot
  partially replace a valid provenance input.
- CI now tests both Python 3.12 and 3.13, matching the support interval declared
  by the package metadata and lock file.
- Evaluation loop variables were clarified to remove accidental reassignment
  without changing calculations.
- Primary tree-3 and robustness tree-10 vine runs now resolve to distinct output
  paths. A fail-closed guard prevents a robustness invocation from overwriting
  any primary refit, forecast, or audit artifact.
- The tree-depth evaluator verifies identical monthly information sets, group
  order, marginal states, realised returns, dates, and common-random-number
  seeds before calculating any robustness comparison.

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
fallback. The full tree-10 run also completed all 144 refits and 3,016 forecasts
without failed pairs or whole-vine fallback. Its 7,920 pair positions and 6,032
matched robustness score rows pass their production artifact checks. The full
local suite contains 133 passing tests, including all eight artifact-bound
production classes. After rebuilding the affected audit chain, the annual
grouping, marginal, Gaussian, primary-vine, evaluation, full-vine, robustness,
and exploratory ML numerical artifacts all reproduced their pre-audit SHA-256
hashes exactly. The final lineage scan verifies 165 path-bound hash records
across 24 audit and manifest documents.

The post-audit exploratory ML-grouping checkpoint adds 12 deterministic annual
fits: spectral clustering and PCA-plus-k-means for each year from 2020 through
2025. Both methods produce exactly 11 non-empty groups per year and 132,704
long-form training/evaluation group-return rows. The largest independently
reconstructed stock-versus-group portfolio discrepancy is `6.94e-17`. A second
complete build reproduced the assignment, Parquet, seed-manifest, and audit
hashes byte for byte.

## Interpretation and remaining boundaries

- Licensed point-in-time S&P 100 membership, historical GICS, and permanent
  identifiers remain absent. Yahoo cannot close this provenance gap.
- H1 is not supported: the confidence interval for the cluster-minus-GICS
  dependence gap includes zero. H2 has no confirmatory support: none of the six
  vine-minus-Gaussian loss comparisons is significant after Holm adjustment.
- M4 ranks first and passes the frozen H3 calibration restriction, but this is a
  provisional model-ranking result, not evidence that vines dominate Gaussian
  copulas or that clustering creates return predictability.
- Tree 10 increases mean fitted parameters from 40.51 to 66.22. Its mean
  out-of-sample copula log score is descriptively higher within both GICS and
  hierarchical groupings, but none of the six tree-10-minus-tree-3 VaR/ES loss
  comparisons is significant after Holm adjustment. All four tree-depth
  variants avoid adjusted Kupiec or Christoffersen independence rejections.
  The robustness run therefore supplies no adjusted forecast-loss evidence that
  the additional trees improve or worsen the primary portfolio-risk forecasts,
  and it does not alter H1-H3.
- With 1,508 evaluation dates, the 99% VaR analysis has only about 15 expected
  exceptions. Forecast losses and matched comparisons should carry more weight
  than annual 99% coverage p-values.
- M0 deliberately estimates historical simulation from the realised primary
  portfolio series, using the annual active set that applied on each historical
  date. M1-M4 instead apply the evaluation year's active set and grouping to
  their trailing estimation window. This is the frozen implemented design, not
  a coding error, but a current-composition historical-simulation backcast would
  be a useful explicitly labelled robustness check before final submission.
- Secondary group-balanced portfolios and the matched M5-M8 Gaussian/vine risk
  forecasts remain future stages. The spectral and PCA-plus-k-means grouping
  implementation is an exploratory extension; its assignments and diagnostics
  must not be described as completed VaR/ES evidence.
- Descriptively, spectral grouping's mean annual out-of-sample pair-weighted
  dependence gap is 0.0273 above GICS and 0.0329 above hierarchical grouping.
  PCA-plus-k-means is 0.0051 and 0.0107 above those baselines, respectively.
  These are separation diagnostics, not adjusted inference, return-prediction
  evidence, or a risk-model ranking; the core H1-H3 conclusions are unchanged.
- Several mature ingestion and orchestration functions remain long and complex.
  Their behavior is well covered at important boundaries, but further
  decomposition should be incremental and paired with characterization tests to
  avoid changing the frozen research design.

## Release rule

Before presenting new results, rerun the checkpoint in `docs/code_quality.md`,
confirm a clean Git state and complete run manifest, and retain the provisional
label until an authorized universe extract passes strict reconciliation.
