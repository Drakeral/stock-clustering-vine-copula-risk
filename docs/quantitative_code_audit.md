# Quantitative Code Audit

Audit date: 20 September 2026

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
the full ten-tree vine robustness analysis, the spectral and PCA-plus-k-means
M5-M8 risk extension, the annual group-balanced portfolio robustness, the
current-composition historical-simulation robustness, artifact schemas and
hashes, deterministic report tables and figures, tests, CI, dependency locking,
secret boundaries, and repository hygiene.

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
- The annual group-balanced portfolio constructor was decomposed into focused
  validation, weighting, return, and state-transition functions. New checks
  reject invalid tolerances, duplicate or non-string security columns, blank
  labels, inconsistent active sets, missing return columns, and misaligned
  annual rebalance dates. Rebuilding the production artifacts after this
  refactor reproduced all three output hashes exactly.
- Credential-bearing Massive requests and Yahoo reference requests now require
  an exact allowlisted HTTPS origin before network access. Provider pagination
  is revalidated on every page, malformed page structures fail closed, and CI
  runs an explicit security-rule scan. Git executable discovery in the run
  manifest no longer relies on a partial executable path.
- Massive S3 credentials now have the same fail-closed origin protection as the
  REST key: the configured endpoint is validated and canonicalised to
  `https://files.massive.com` before either credential is read or passed to the
  SDK. Provider object keys are resolved beneath the raw-data directory and
  reject wrong prefixes, traversal components, and symlink escapes.
- Repository lineage records can no longer bind or inspect paths outside the
  project root. Relative traversal, absolute external paths, and symlink escapes
  are reported as `external_path` failures before any file is hashed.
- The maintained-module complexity gate now covers the deterministic report
  generator, provider boundary, and shared filesystem/lineage helpers in
  addition to evaluation and group-balanced orchestration. The remaining
  complexity exceptions are older, explicitly bounded ingestion and modelling
  orchestration functions with characterization tests.

## Verification evidence

The local checkpoint includes locked-environment reconstruction, Ruff lint,
security, maintained-module complexity and format checks, source compilation,
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
local suite contains 182 passing tests, including all thirteen artifact-bound
production classes. After rebuilding the affected audit chain, the annual
grouping, marginal, Gaussian, primary-vine, evaluation, full-vine, robustness,
and exploratory ML numerical artifacts all reproduced their pre-audit SHA-256
hashes exactly. The final lineage scan verifies 360 path-bound hash records
across 33 audit, manifest, and report-lineage documents.

An independent read-only reconciliation of the production Parquet files
reconstructed all quantile and FZ0 scores to a maximum numerical difference of
`8.53e-14`, matched every model summary exactly, and found maximum realised-loss
disagreement of `5.55e-17` across M0-M4. It independently confirmed 1,584
marginal refits with four EWMA fallbacks (`0.253%`), 144 Gaussian and 144 primary
vine refits, zero failed vine pairs, zero whole-vine fallbacks, at least 750
observations in every historical-simulation window, and a maximum GICS-versus-
hierarchical portfolio identity error of `2.78e-17`.

The deterministic reporting checkpoint adds seven CSV tables, three canonical
SVG figures, one audit-derived narrative summary, and a report manifest. A
repeat build reproduced all eleven generated output hashes; visual inspection
confirmed the final layouts. The generator rejects failed or stale source
audits, preserves the provisional reporting scope, and performs neither model
refits nor new inference.

The post-audit exploratory ML-grouping checkpoint adds 12 deterministic annual
fits: spectral clustering and PCA-plus-k-means for each year from 2020 through
2025. Both methods produce exactly 11 non-empty groups per year and 132,704
long-form training/evaluation group-return rows. The largest independently
reconstructed stock-versus-group portfolio discrepancy is `6.94e-17`. A second
complete build reproduced the assignment, Parquet, seed-manifest, and audit
hashes byte for byte.

The downstream ML-risk checkpoint adds 1,584 monthly marginal refits, 144
Gaussian refits, 144 truncated-vine refits, and 6,032 M5-M8 daily evaluation
records. All marginal fits were accepted by a GARCH-family specification, all
pair-copula fits succeeded, no whole-vine fallback was used, and the Gaussian
stage reproduced the primary 72-record common-random-number manifest exactly.
The frozen 24-test exploratory DM family produced no loss difference significant
after Benjamini-Hochberg adjustment at 5%. Within each ML grouping, vine mean
copula log scores are descriptively higher than the matched Gaussian scores;
these log-score comparisons are not cross-grouping inference. A second complete
M5-M8 marginal, Gaussian, vine, and evaluation build reproduced all 12 derived
Parquet and audit SHA-256 hashes byte for byte.

The WBA DAP valuation checkpoint reruns M0-M8 with the right valued at zero and
at its `$3` cap, without overwriting the `$0.53` primary artifacts. The scenario
WBA returns are -4.4240% and +20.6177%, respectively. Each scenario changes
exactly one stock-return cell, holds every annual assignment fixed, and
reconstructs the direct equal-stock-weighted event-date portfolio from all four
groupings with maximum error below `6.51e-19`. Both runs complete 3,168 marginal,
288 Gaussian, and 288 tree-3 vine refits. They have no failed pair fits or
whole-vine fallbacks; their five and two EWMA fits remain well below the one-
percent limit. All 95%, 97.5%, and 99% exception counts remain unchanged. M4
remains uniquely first, H2 remains unsupported, H3 remains supported under its
frozen operational rule, and none of the 24 exploratory ML comparisons is
significant after BH adjustment. H1 is unchanged and not retested in this
fixed-assignment corporate-action valuation sensitivity.

The portfolio-weighting checkpoint constructs two distinct annual buy-and-hold
targets: equal GICS-sector weights and equal hierarchical-cluster weights. It
adds 132,704 group-return rows and 12,064 portfolio-return rows. Both targets
contain 1,508 evaluation dates, and their direct stock-versus-group return
identities hold within `2.78e-17`. Evaluation-period group weights drift from
3.71% to 18.35%, confirming that the portfolios are not being silently reset
each day.

The matched risk run adds 1,584 marginal refits, 144 Gaussian refits, 144 tree-3
vine refits, and 9,048 daily score rows. Fourteen marginal fits use a fallback
level, of which two use EWMA (`0.126%`, below the 1% gate). No Gaussian
correlation repair, failed pair-copula fit, or whole-vine fallback occurs. The
Gaussian model ranks first and the vine second within each portfolio; historical
simulation ranks third. None of the six vine-minus-Gaussian loss comparisons is
significant after Holm correction—the smallest raw p-value is `0.0461`, but its
adjusted p-value is `0.2766`. Historical simulation has adjusted exception-
independence rejections, while neither Gaussian nor vine does. Rankings remain
strictly within portfolio because the two realised loss series differ. This
exploratory robustness therefore does not alter H1-H3.

The current-composition historical-simulation checkpoint adds 1,508 `M0_CC`
forecasts and 3,016 matched daily score rows. It freezes each evaluation year's
active set across that year's rolling three-year backcast and reproduces M0's
realised return exactly on every date. Risk forecasts differ on 710 dates, but
both variants retain 58, 28, and 16 exceptions at 95%, 97.5%, and 99%. M0 has
lower mean loss on all three primary scores. Only the 99% quantile-loss
difference is significant after the separate three-test Holm correction
(`M0_CC - M0` mean loss `1.66e-6`, adjusted p-value `3.86e-25`); the 95% and FZ0
differences are not adjusted-significant. This is partial exploratory evidence
favouring realised-history M0, not a revision of H1-H3. A second complete build
reproduced the forecast, window, daily-score, and audit hashes byte for byte.

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
  their trailing estimation window. The completed `M0_CC` robustness check
  indicates that the original M0 choice has lower 99% quantile loss after Holm
  correction, while its other two primary-score advantages are not
  adjusted-significant.
- Secondary group-balanced portfolios are implemented as an exploratory
  weighting robustness. Gaussian models rank first within both targets, but no
  adjusted vine-versus-Gaussian loss difference is detected. This is not a
  cross-portfolio ranking and does not revise H1-H3 or the core M0-M4 ranking.
- The WBA sensitivity brackets the DAP at zero and its contractual cap; it does
  not model a stochastic payoff distribution or dependence between that payoff
  and market returns. Its conclusion is narrowly that these two endpoint
  valuations do not change the implemented risk-model conclusions.
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
