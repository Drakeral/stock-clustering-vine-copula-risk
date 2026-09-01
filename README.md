# FE5110 Stock-Clustering and Vine-Risk Project

This repository implements the reproducible research pipeline for the fixed,
point-in-time S&P 100 universe described in `FE5110_Project_Plan.md`.

## Current gate status

`foundation_v2` is intentionally **blocked only at the licensed-universe gate**.
The independent market-input and strengthened data-construction gates pass. The
live, separated statuses are in `data/audit/current_gate_status.json`.

The exact MediaWiki revision payloads are archived and hashed under
`data/raw/universe_sources/`. One external input still has to be supplied by an
authorised user: a licensed point-in-time S&P/Compustat/vendor universe and GICS
extract for 2 January 2020.

The repository never fabricates or silently passes either requirement.

## Rebuild from a clean clone

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

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

Place the authorised CSV at
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

Load credentials, download immutable provider data, and independently verify
every input against an XNYS session calendar:

```zsh
source scripts/load_credentials.zsh
uv run python scripts/download_market_data.py --workers 8
uv run python scripts/verify_foundation_inputs.py
uv run python scripts/build_return_panel.py
uv run python scripts/validate_portfolio_arithmetic.py
uv run python scripts/update_foundation_status.py
uv run python -m unittest discover -s tests -v
```

The download is resumable. Existing daily and reference files are schema-
validated and hashed rather than overwritten. Use `--overwrite` only for an
intentional provider refresh. The verifier recomputes every hash and row count;
it does not infer sessions from the provider file list. For the legacy inputs in
this working copy, an offline re-verification after a wrapper-format migration
can use `--manifest data/manifests/market_input_manifest.json`; a fresh downloader
run emits the strict operational manifest used by the default command.

Immediately before any clustering or risk-model command, require the aggregate
gate explicitly:

```zsh
uv run python scripts/update_foundation_status.py --require-pass
```

An exit status of 2 is expected while the licensed-universe input is absent.

Finally capture the exact code/environment/configuration/input/output identity:

```zsh
uv run python scripts/generate_run_manifest.py
```

## Reproducibility and licensing boundary

- Track code, configuration, tests, audit statuses, and sanitised manifests in
  `data/manifests/`.
- Do not track credentials, `data/raw/licensed/`, `data/raw/market/`, or derived
  Parquet files. Whether normalized vendor fields may be submitted must be
  checked against the applicable licence.
- The historical access audit remains unchanged. New gates supersede its stale
  `next_gate` text through `data/audit/current_gate_status.json`.
- A run may be computed for diagnostics while provenance is blocked, but all
  such results must be labelled provisional and must not enter confirmatory
  modelling.
- Detailed provider-event reconciliation is written only to the ignored
  `data/audit/data_quality_report.local.json`; its public report contains counts
  and a content hash, not licensed event rows.
- Build a submission bundle with `git archive HEAD`, never by zipping the live
  working folder, which contains ignored licensed files and credentials.

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
  and
- `data/audit/`: separate provenance, input-integrity, construction, and current
  readiness reports.
