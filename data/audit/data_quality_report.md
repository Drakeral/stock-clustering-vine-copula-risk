# FE5110 Data Quality Report

## Data gate

**Status: PASS**

- Gate: data_construction_v2
- Output scope: provisional_pending_universe_provenance
- Universe provenance: blocked
- Provider daily files: 2262
- Market trading dates: 2262
- Candidate securities: 101
- Included after the initial-window coverage screen: 100
- Excluded: 1
- Duplicate security-date rows: 0
- Non-positive closes: 0
- Unexplained lifecycle issues: 0
- Unmatched reference events: 0
- Extreme-review issues: 0

## Exclusions

| Ticker | Company | Price coverage | Return coverage | Reason |
|---|---|---:|---:|---|
| DOW | Dow Inc. | 25.199% | 25.100% | initial_training_coverage_below_threshold |

## Quality flags

| Flag | Count |
|---|---:|
| extreme_total_return | 2 |
| extreme_unadjusted_return_without_split | 2 |
| gap_bridge_return | 2 |
| manual_corporate_action | 13 |
| synthetic_terminal_consideration | 1 |
| ticker_segment_transition | 6 |
| verified_halt_zero_return | 2 |

## Corporate-action reviews still requiring validation

- None. All prespecified complex events have a documented treatment.

## Manual corporate-action reconciliations

- Expected actions: 13
- Applied actions: 13
- Missing actions: 0

## Terminal-series audit

- Early-ending series: 3
- Documented exit treatments: 3
- Exit-treatment issues: 0

## Lifecycle completion

- Verified halts applied: 2
- Unexplained missing active dates: 0
- Terminal-cash security dates: 250

## Reference-event reconciliation

- applied: 3138
- duplicate_aggregated: 14
- out_of_segment: 23

## Extreme observations

- Reviewed extremes: 2
- Review issues: 0

The coverage decision uses only the configured 2017–2019 initial training window. Later availability is not used to select the frozen universe.
