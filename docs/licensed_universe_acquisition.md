# Licensed S&P 100 Universe Acquisition

## Required one-time extract

Obtain an authorized, point-in-time constituent extract for:

- index: S&P 100 (OEX);
- effective/as-of date: 2 January 2020;
- expected result: 101 securities, including both Alphabet share classes; and
- purpose: academic research for NUS FE5110, with licensed raw data retained
  locally and excluded from Git and publication.

Request these source fields wherever available:

| Required normalized field | Preferred source value |
| --- | --- |
| `security_id` | Permanent vendor security identifier, such as Bloomberg FIGI/security ID or Compustat `GVKEY.IID` |
| `issuer_id` | Permanent vendor company identifier, such as Bloomberg company ID or Compustat `GVKEY` |
| `ticker` | Point-in-time trading ticker on 2 January 2020 |
| `share_class` | Point-in-time share-class description or code |
| `company_name` | Point-in-time issuer/security name |
| `gics_sector` | Point-in-time GICS sector name |
| `gics_sub_industry` | Point-in-time GICS sub-industry name |
| `membership_date` | Index membership effective-from date supplied by the source |
| `as_of_date` | `2020-01-02` |
| `source_record_id` | Unique vendor row/file record identifier; if none exists, a documented composite of immutable vendor keys and the as-of date |

Also retain the index identifier, constituent effective-through date, GICS
codes, export timestamp, source dataset name, and entitlement reference as local
audit fields if the source supplies them.

## Preferred authorized sources

### 1. NUS-licensed Bloomberg terminal

NUS Business School's Financial Database lists Bloomberg, Refinitiv Workspace,
WRDS, CRSP, and Compustat among its subscriptions. It states that access is for
current NUS Business School users and that students need faculty consent before
contacting the database team:

https://bizfaculty.nus.edu.sg/financial-database/

Ask for a Bloomberg terminal export of historical members of `OEX Index` on
2 January 2020, together with permanent security/company identifiers and the
GICS sector and sub-industry classifications effective on that date. On the
terminal, `OEX Index` followed by `MEMB <GO>` is the member-weightings workflow;
the historical date and entitlement must be confirmed on the licensed terminal.
Do not substitute current GICS values for point-in-time classifications.

### 2. S&P Dow Jones Indices SPICE or Data Services

This is the direct-source route. S&P DJI states that its subscription-based
SPICE platform includes current and historical constituent data and supports
custom constituent-level downloads:

https://www.spglobal.com/spdji/en/landing/topic/spice/

S&P DJI also states that its authenticated Data Services APIs provide historical
constituents, corporate actions/index events, and fundamental data. An S&P DJI
Data Services account is required:

https://www.spglobal.com/spdji/en/landing/topic/api-data-solutions/

### Compustat caveat

Do not assume that a current WRDS Compustat subscription still contains the S&P
100 membership file. An SMU Libraries review records that S&P constituent-name
data was removed from Compustat in July 2020 because of direct S&P DJI licensing:

https://library.smu.edu.sg/topics-insights/notes-and-thoughts-retrieving-historical-members-sp-500-wrds

WRDS/Compustat remains useful for permanent identifiers and historical GICS
after an authorized S&P 100 membership extract has been obtained. WRDS documents
historical ticker/CUSIP tables and a historical-GICS method here:

https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/using-compustat-historical-identifier-notebook/

https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/wrds-macro-indclass/

## Access request template

Subject: FE5110 request for one-time historical S&P 100 constituent extract

> I am an NUS MSc Financial Engineering student conducting an FE5110 academic
> research project on stock clustering and portfolio tail-risk modelling. I
> request authorized access, or a staff-assisted one-time export, for the S&P 100
> (OEX) constituents effective on 2 January 2020. The required fields are the
> permanent security and issuer identifiers, point-in-time ticker/share class and
> company name, membership effective-from/effective-through dates, and the GICS
> sector and sub-industry effective on that date. The expected result is 101
> securities because both Alphabet share classes are represented. The raw export
> will remain within the licensed research environment, will not be committed to
> Git or redistributed, and only sanitized hashes and reconciliation counts will
> appear in the project. My faculty supervisor's approval can be provided.

For the NUS Business School Financial Database, the published contact is
`bizfdb@nus.edu.sg`. Send the request only after obtaining the required faculty
consent.

## Local handoff

Save the authorized export under `data/interim/licensed/` without renaming or
editing the original. Record the provider/dataset name and a non-confidential
entitlement label. The project helper will map it into
`data/interim/licensed/universe_population_worksheet.csv`, validate all 101
rows, and generate the ignored gate input at
`data/raw/licensed/universe_sp100_2020-01-02.csv`.
