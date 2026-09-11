# FE5110 Data Dictionary

## Frozen-universe dataset

File: `data/raw/universe_sp100_2020-01-02.json`

| Field | Type | Definition |
|---|---|---|
| `security_id` | string | Stable security identifier in the normalized point-in-time universe. |
| `issuer_id` | string | Stable issuer identifier; shared by GOOG and GOOGL. |
| `ticker` | string | Exchange ticker shown in the point-in-time membership table. |
| `share_class` | string | Normalized share-class label. |
| `company_name` | string | Company/security name in the membership snapshot. |
| `gics_sector` | string | Contemporaneous GICS sector from the S&P 500 snapshot. |
| `gics_sub_industry` | string | Contemporaneous GICS sub-industry from the S&P 500 snapshot. |
| `membership_as_of` | date string | Research-universe reference date: 2020-01-02. |
| `membership_table_date` | date string | Date printed on the underlying membership table: 2019-11-21. |
| `initial_universe_status` | categorical string | Candidate status until the full 2017-2019 coverage screen is complete. |
| `known_ticker_event_review` | string | Flag for known symbol, merger, or terminal-return continuity work. |
| `source_record_id` | string | Identifier of the source record used for reconciliation. |

The universe contains securities, not unique issuers. Multiple listed share classes are retained because they are separate return series. The tracked Wikipedia revisions and Yahoo cross-check are secondary sources only. Under protocol amendment PA-001 they support provisional modelling, but `licensed_universe_provenance` remains false until an authorised extract with the normalized fields above reconciles without membership or GICS differences.

## Massive daily aggregate input

Source prefix: `us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`

| Field | Type | Definition |
|---|---|---|
| `ticker` | string | Point-in-time exchange ticker in Massive symbology. |
| `volume` | number | Daily reported share volume. |
| `open` | number | Unadjusted daily opening price. |
| `close` | number | Unadjusted daily closing price. |
| `high` | number | Unadjusted daily high price. |
| `low` | number | Unadjusted daily low price. |
| `window_start` | integer | UTC Unix timestamp in nanoseconds for the aggregate window. |
| `transactions` | integer | Number of qualifying transactions in the aggregate. |

Flat-file prices are not split- or dividend-adjusted. The processed return pipeline must therefore combine these files with Massive split and dividend reference data.

## Processed return definition

The primary series is the close-to-close log total return after applying point-in-time stock-split factors and cash distributions on their ex-dividend dates. Corporate actions, ticker changes, mergers, and delistings are joined using the versioned security master and explicit action terms. Raw provider files remain unchanged.

For an ordinary day, gross total return is `(close_t + cash_dividend_t) / close_(t-1)`. A provider factor is applied as a share multiplier only for an event classified as `literal_split` or `stock_dividend`. Events are typed as `literal_split`, `stock_dividend`, `spin_off`, `merger_exchange`, or `cash_acquisition`. For a merger or spin-off, gross return is total closing consideration per old share—successor shares, cash, contingent value, and distributed-share value—divided by the previous close. The explicit action terms and primary-source URLs are recorded in `config/manual_corporate_actions.json`.

Provider P-prefixed continuity factors associated with reconstructed spin-offs are retained only in `provider_continuity_factor`. They are not described as parent shares received and are never applied in addition to an explicit distribution value.

## Processed datasets

### `data/processed/security_daily.parquet`

One row per research security and provider trading date. Important derived fields are:

| Field | Type | Definition |
|---|---|---|
| `research_ticker` | string | Frozen-universe identifier used across ticker segments. |
| `ticker` | string | Provider ticker observed on the date. |
| `observation_status` | categorical string | `observed`, `verified_halt`, `synthetic_terminal_consideration`, or `unexplained_missing`. |
| `lifecycle_classification` | categorical string | Executable lifecycle classification attached to the materialized row. |
| `is_synthetic` | boolean | Whether the row was inserted by a documented lifecycle or terminal-consideration rule. |
| `split_factor` | number | Applied factor for a provider event classified as a literal split; 1 otherwise. |
| `cash_dividend` | number | Cash distribution per share on the ex-dividend date. |
| `effective_share_multiplier` | number | Split or merger exchange ratio used for the current close. |
| `manual_cash_per_old_share` | number | Explicit cash consideration per old share. |
| `manual_contingent_value_per_old_share` | number | Frozen primary value of contingent consideration, including WBA's DAP right. |
| `stock_distribution_value` | number | Event-date closing value of securities distributed per old share. |
| `total_consideration_per_old_share` | number | Current security value plus cash and distributed-security value. |
| `total_return` | number | Simple close-to-close total return. |
| `log_total_return` | number | Natural logarithm of gross total return. |
| `manual_action_id` | nullable string | Identifier joining the row to the manual action configuration. |
| `manual_action_type` | nullable categorical string | Explicit typed-event classification for a manual action. |
| `provider_event_id` | nullable string | Provider event identifier reconciled to the action. |
| `provider_continuity_factor` | nullable number | Provider adjustment factor retained for audit only when explicit action arithmetic replaces it. |
| `quality_flag` | string | Semicolon-separated audit flags; blank when none apply. |

### `data/processed/daily_total_returns.parquet`

Wide stock log-total-return matrix. A security is missing after it ceases trading; no fictitious stock returns are created.

### `data/processed/daily_simple_total_returns.parquet`

The matching wide stock simple-total-return matrix. Cross-sectional portfolio and group arithmetic uses this scale.

### `data/processed/portfolio_constituent_returns.parquet`

Wide modelling matrix. After a documented terminal event, proceeds receive a zero log return until the next annual rebalance, when the security becomes inactive. The separate `data/processed/active_universe_by_year.json` file identifies the annual active set.

### `data/processed/portfolio_constituent_simple_returns.parquet`

The matching simple-return modelling matrix. The daily-rebalanced primary portfolio is the mean of each year's active security columns. Group returns are security means within the annual grouping and are aggregated with annual group-size weights. `data/audit/portfolio_arithmetic.json` verifies this identity to `1e-12` on every defined return date.

### `data/processed/final_universe.json`

Coverage-screen result based only on 3 January 2017 through 31 December 2019. It preserves all candidates and records inclusion status, price coverage, return coverage, and exclusion reason. Its top-level status is `provisional_pending_universe_provenance` while the licensed-universe gate is blocked.

### `data/processed/annual_group_assignments.json`

Annual 2020–2025 GICS and hierarchical-cluster assignments. Each year records
the active securities, rebalance date, left-closed/right-open three-calendar-year
training window, deterministic average-linkage merge history, and final group
labels. Hierarchical labels use only returns strictly before the rebalance date.

### `data/processed/annual_group_returns.parquet`

Long-form daily group-return panel used by the later marginal and copula models.

| Field | Type | Definition |
|---|---|---|
| `date` | date | Trading date within the row's training or evaluation window. |
| `year` | integer | Grouping/model year whose annual labels apply; dates can precede this year for training rows. |
| `universe_variant` | categorical string | `security_primary` or the 50/50 Alphabet `issuer_deduplicated_robustness`. |
| `sample_role` | categorical string | `training` for the preceding three-year window or `evaluation` for the grouping year. |
| `grouping_id` | categorical string | `gics_sector` or `hierarchical_cluster`. |
| `group_id` | string | Frozen sector or annual deterministic cluster label. |
| `group_size` | integer | Number of annual active securities in the group. |
| `portfolio_weight` | number | `group_size / annual_active_security_count`. |
| `simple_return` | number | Equal-weight mean of constituent simple returns. |
| `log_return` | number | `log1p(simple_return)`, used for marginal fitting. |

`data/audit/clustering_diagnostics.json` binds these artifacts to their inputs
and reports annual dependence gaps, pair counts, ARI, NMI, and the direct-versus-
grouped primary-portfolio arithmetic error.

## Historical-simulation benchmark outputs

`data/processed/historical_simulation_risk_forecasts.parquet` stores the M0
benchmark using the common forecast schema. It contains one record per 2020–2025
evaluation date and uses the canonical daily-rebalanced equal-weight portfolio,
not either grouping representation. Because M0 is deterministic and has neither
marginal nor copula estimation, `seed_components` and `copula_log_score` are null,
the fallback fields remain zero/false, and `grouping_id` is `none`.

`data/processed/historical_simulation_windows.parquet` binds each daily M0
`refit_id` to its requested three-calendar-year start, first and last observations,
observation count, forecast date, and current annual active-security count. Every
training end date is strictly before its forecast date and every window has at
least 700 observations. Its schema is
`config/schemas/historical_simulation_window_record.schema.json`.

`data/audit/historical_simulation_quality.json` binds both M0 outputs to the
simple-return panel, annual active schedule, passed portfolio-arithmetic audit,
foundation status, and frozen model configuration by SHA-256.

### `data/processed/marginal_refits.parquet`

One row per year, month, grouping system, and group in the primary security
universe. Each record contains the exact three-calendar-year training bounds,
training/evaluation counts, fitted parameters in the documented 100-times return
scale, convergence information, the selected method, fallback level, and a JSON
log of every attempted fit. The schema is
`config/schemas/marginal_refit_record.schema.json`.

### `data/processed/marginal_daily_forecasts.parquet`

One row per evaluation date and group. Conditional means, volatilities, and
variances are stored in decimal log-return units. The realized group log return,
standardized residual, clipped PIT, monthly `refit_id`, method, and fallback status
are retained for out-of-sample forecast diagnostics. The schema is
`config/schemas/marginal_daily_record.schema.json`.

### `data/processed/monthly_copula_training_pits.parquet`

Long-form, aligned in-sample PIT matrices used to estimate monthly copulas. Every
`copula_refit_id` contains exactly 11 groups on each retained training date. A
date is retained only when every group in the matched GICS or hierarchical
representation has a finite standardized residual and clipped PIT. All dates are
strictly earlier than the associated `refit_date`; AR(1) initialization can remove
the first raw training observation. At least 700 aligned dates are required for
every monthly block. The schema is
`config/schemas/marginal_training_pit_record.schema.json`.

`data/audit/marginal_model_quality.json` binds all three outputs to the annual
grouping panel, grouping audit, foundation status, and frozen model configuration.
It fails for duplicate or incomplete forecasts, a missing monthly refit, an
incomplete or incorrectly dimensioned training PIT matrix, look-ahead, non-finite
output, an out-of-bound PIT, or an EWMA share above 1% of group-month fits.

## Vine-copula model outputs

`data/processed/vine_copula_refits.parquet` stores one M2 or M4 record per
grouping-month. Each record binds the source marginal PIT block, matched Gaussian
fallback, ordered variables, truncation level, selected R-vine structure and pair
copulas, AIC/log-likelihood diagnostics, common-random-number hash, and all pair or
whole-vine fallback dispositions. The primary output is truncated after tree 3.

`data/processed/vine_risk_forecasts.parquet` uses the common forecast-record
schema. It contains daily M2/M4 VaR and ES forecasts, realised simple portfolio
return and loss, copula log score, monthly seed components, marginal fallback
count, and whole-vine fallback flag.

`data/audit/vine_copula_quality.json` binds both outputs and every upstream input
by SHA-256. It separately reports computational validity and whether the vine is
eligible to be declared best under the frozen one-percent whole-vine fallback
limit. Fallback date counts and fractions are reported for each model; the maximum
model-specific fraction governs eligibility. The audit also retains the union of
dates on which either model fell back as a conservative descriptive diagnostic.

## Model-evaluation outputs

`data/processed/risk_evaluation_daily.parquet` contains one matched row per model
and evaluation date. Its 7,540 rows cover M0-M4 on the same 1,508 realised
portfolio losses and store 95% and 99% quantile losses, the 97.5% FZ0 score,
strict VaR exception indicators, and the source fallback/log-score diagnostics.
Its schema is `config/schemas/risk_evaluation_daily_record.schema.json`.

`data/manifests/inference_seed_manifest.json` records the H1 PCG64DXSM seed,
block length, candidate and accepted replication counts, accepted-index hash,
and replicate-statistic hash. A paired candidate that makes any retained stock
constant is discarded in full and redrawn by continuing the same random stream.

`data/audit/model_evaluation.json` binds every forecast, assignment, stock-return,
configuration, upstream audit, daily score, and inference-seed artifact by
SHA-256. It reports the 20-test Holm calibration family, annual descriptive
diagnostics, six matched vine-minus-Gaussian DM tests, equal-year H1 bootstrap,
model rankings, and the preregistered H1-H3 decisions. A hypothesis not being
supported is a research result and does not make this computational quality gate
fail.

## Lifecycle policy

Every XNYS session between a security's explicit listing and removal dates is classified as `active_price`, `verified_halt`, `terminal_cash`, `removed`, or `unexplained_missing`; pre-listing dates are counted as `pre_inception`. A verified halt with no corporate action receives a stale synthetic price and zero return, while the cumulative price move remains on resumption as a `gap_bridge_return`. A corporate action on a missing price date requires review and cannot be stale-filled automatically.

## Required quality checks and dispositions

- duplicate ticker-date observation;
- missing expected trading day and segment gap;
- non-positive price;
- extreme unadjusted return around a split;
- dividend without a matching price observation;
- ticker continuity event;
- terminal/delisting event; and
- initial training coverage below 95%.

Every raw split or dividend record receives exactly one disposition: `applied`, `duplicate_aggregated`, `out_of_segment`, or `unmatched_price`. Every absolute simple return above 40% requires a versioned `validated` disposition in `config/observation_reviews.json`; validated genuine observations are retained without winsorisation.

The executable `data_construction_v2` gate fails on any duplicate, unexplained active-session absence, non-positive or non-finite required value, invalid OHLC ordering, negative volume/transaction count, unmatched or unclassified reference event, missing manual action, unresolved lifecycle/exit review, or unreviewed extreme. The aggregate `foundation_v2` status additionally requires universe provenance, market-input integrity, portfolio arithmetic, and the method freeze to pass.
