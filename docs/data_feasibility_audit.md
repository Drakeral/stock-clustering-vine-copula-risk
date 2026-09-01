# FE5110 Data Feasibility Audit

> Historical access-feasibility record only. Its conditional pass is preserved,
> but it is superseded for research readiness by
> `data/audit/current_gate_status.json`. It does not authorize clustering,
> out-of-sample fitting, or confirmatory result inspection.

Audit date: 15 August 2026

## Outcome

**Conditional pass.** The current Massive credentials support the revised 2017-2025 sample through stock daily-aggregate flat files, and the REST API supplies the split and dividend reference data needed to construct total returns. The original 2015 start date is not available under the current entitlement.

The research sample is therefore amended to:

- initial training period: 3 January 2017 to 31 December 2019;
- point-in-time universe date: 2 January 2020; and
- out-of-sample period: 2 January 2020 to 31 December 2025.

## Access tests

| Test | Result | Interpretation |
|---|---|---|
| REST daily bars, AAPL, January 2025 | HTTP 200; five rows returned | REST credential is valid. |
| REST daily bars, AAPL, January 2015 | HTTP 403 `NOT_AUTHORIZED` | REST plan excludes that historical timeframe. |
| Flat-file day-aggregate HEAD, 2015-2016 | HTTP 403 | Those years are outside the current flat-file entitlement. |
| Flat-file day-aggregate HEAD, 2017-2025 | HTTP 200 | Revised sample is downloadable. |
| Flat-file download, 3 January 2017 | HTTP 200; 7,980 rows | Earliest entitled year and schema validated. |
| Flat-file download, 2 January 2025 | HTTP 200; 10,870 rows | Recent schema is consistent with the 2017 file. |
| REST AAPL splits, 2017-2025 | HTTP 200; one event | Split endpoint is accessible. |
| REST AAPL dividends, 2017-2025 | HTTP 200; 36 events | Dividend endpoint is accessible. |

The daily files share the fields `ticker`, `volume`, `open`, `close`, `high`, `low`, `window_start`, and `transactions`. Spot checks found legacy symbols such as `FB`, `RTN`, and `UTX` in 2017 and successor symbols such as `META` and `RTX` in 2025, confirming that ticker-continuity logic is required.

## Universe-source audit

- S&P 100 membership revision: `929329274`, timestamped 5 December 2019, containing a component table dated 21 November 2019.
- S&P 500/GICS revision: `933762580`, timestamped 2 January 2020.
- Parsed securities: 101.
- Unmatched GICS mappings: 0.
- GICS sectors represented: 11.

These are contemporaneous secondary-source snapshots suitable for provisional pipeline development. The original final provenance gate requires an authorised point-in-time S&P/Compustat/vendor extract and cannot be waived silently. Protocol amendment PA-001 explicitly permits Yahoo-supported provisional modelling while preserving that licensed gate as not passed.

## Data-construction risks

1. Flat-file prices are unadjusted, while REST aggregate bars are split-adjusted but not dividend-adjusted.
2. Massive does not automatically concatenate ticker changes, so stable-identifier mapping is required.
3. Mergers and delistings require a terminal-return rule; they must not be treated as ordinary missing data.
4. The frozen universe is still a candidate universe until every security passes the 95% initial-window coverage test.
5. Licensed raw files should not be committed or redistributed; only reproducible download code, derived results, and permitted supporting data should be submitted.

## Source documentation

- Massive custom bars: https://massive.com/docs/rest/stocks/aggregates/custom-bars
- Massive stock flat-file overview: https://massive.com/docs/flat-files/stocks/overview
- Massive day aggregates: https://massive.com/docs/flat-files/stocks/day-aggregates
- Massive adjustment policy: https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends
- S&P 100 snapshot: https://en.wikipedia.org/w/index.php?title=S%26P_100&oldid=929329274
- S&P 500/GICS snapshot: https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=933762580

## Historical next-phase conclusion (superseded)

**Passed on 15 August 2026.** The implementation downloaded and validated 2,262 daily files, collected 210 ticker/event-type reference responses, resolved ticker and complex corporate-action continuity, and applied the 95% screen using only 2017-2019 data. The frozen research universe contains 100 securities; DOW is excluded because its initial-window price coverage is 47.480%.

The executable construction audit is stored in `data/audit/data_quality_report.md`. Subsequent remediation separated access, provenance, market-input, construction, arithmetic, and method-freeze gates. The current aggregate gate—not this historical conclusion—controls whether modelling may begin.
