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

The universe contains securities, not unique issuers. Multiple listed share classes are retained because they are separate return series. The tracked Wikipedia revisions are reproducible secondary sources only. `provenance_status` remains blocked until an authorised extract with the normalized fields above reconciles without membership or GICS differences.

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
