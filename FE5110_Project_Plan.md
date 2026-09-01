# FE5110 Research Protocol

## Working title

**Data-Driven Stock Clustering and Vine-Copula Portfolio Tail-Risk Modelling**

## 1. Research objective

This project studies whether data-driven stock groupings and flexible dependence models improve portfolio tail-risk measurement. It does not study earnings announcements, implied-volatility crush, NLP, or earnings forecasting.

### Primary research question

For a fixed, point-in-time universe of large-cap US equities, does grouping stocks with rolling hierarchical correlation clustering and modelling the resulting group returns with a regular vine (R-vine) copula improve one-day-ahead out-of-sample Value-at-Risk (VaR) and Expected Shortfall (ES) forecasts relative to GICS sector groupings and Gaussian dependence models?

### Secondary question

Do cluster-balanced portfolios exhibit better realised diversification than comparable sector-balanced portfolios?

## 2. Hypotheses

- **H1 - Cluster information:** Data-driven clusters formed in each training window will have a larger out-of-sample within-group versus between-group dependence gap than GICS sectors.
- **H2 - Copula specification:** For a fixed grouping and identical marginal models, the R-vine copula will produce lower out-of-sample VaR and joint VaR-ES loss than the Gaussian copula.
- **H3 - Integrated model:** The data-driven-cluster plus R-vine specification will provide the best overall VaR/ES calibration among the preregistered models.

The secondary portfolio comparison is exploratory and will not be treated as evidence of expected-return predictability.

## 3. Dataset and sample design

### Universe

- Common equities in the **S&P 100 as at 2 January 2020**.
- The constituent list and GICS sector labels will be frozen before examining out-of-sample results.
- A security must have at least 95% of observations in the initial training period to enter the fixed research universe.
- A security that subsequently ceases trading will be retained through its final valid return, including a terminal delisting return when available, and removed at the next annual rebalance. No later index additions will be introduced.
- The final universe file, inclusion rules, and exclusions will be archived with the project.
- If a reliable point-in-time constituent list and sector mapping cannot be obtained, the universe will not be silently replaced; the protocol and resulting survivorship limitation must first be documented.

### Market data

- Daily unadjusted OHLCV flat files plus split and dividend reference data from Massive.
- Sample period: **3 January 2017 to 31 December 2025**.
- Primary variable: close-to-close split- and dividend-adjusted daily log total return.
- Obvious data errors, duplicate observations, non-trading days, missing values, and corporate actions will be handled using documented, reproducible rules.

### Estimation and test windows

- Initial training window: 2017-2019.
- Out-of-sample evaluation: 2020-2025.
- Clusters are re-estimated annually using the preceding three years only.
- Marginal and copula risk models are re-estimated monthly using the latest three years; fitted volatility states are updated daily.
- All transformations, model choices, weights, and thresholds for a forecast must use information available before that forecast date.

## 4. Portfolio and clustering definitions

### Data-driven grouping

1. Estimate the Spearman rank-correlation matrix from the rolling training window.
2. Convert correlations to distances using
   `d(i,j) = sqrt((1 - rho(i,j)) / 2)`.
3. Apply average-linkage hierarchical clustering.
4. Set the number of clusters equal to the number of non-empty GICS sectors in the frozen universe. This keeps the data-driven and sector models at the same aggregation dimension.

### Gated unsupervised-learning extension

After the complete hierarchical-clustering/R-vine pipeline runs reproducibly, the clustering study will be expanded with two prespecified unsupervised alternatives:

- spectral clustering on a rolling correlation-based affinity graph; and
- k-means applied to principal-component embeddings of stocks' rolling correlation profiles.

All feature construction, principal-component fitting, affinity calibration, and clustering will use the training window only. Each method will use the same number of groups as the GICS and hierarchical specifications. The alternatives will be compared using out-of-sample dependence separation, cluster stability, and downstream VaR/ES forecast loss. Hierarchical clustering remains the primary preregistered method; the additional algorithms form a secondary ML robustness study and will be implemented only after the core models and evaluation pipeline pass their tests.

### Group returns and primary portfolio

- Stock log returns are converted to simple returns before cross-sectional aggregation. Each group simple return is the equal-weight mean of its constituent stock simple returns.
- Group weights are proportional to group size, so both grouping representations exactly reconstruct the same daily-rebalanced equal-weight stock portfolio. Membership and group labels change only at the annual rebalance.
- Marginal models use the log of each group gross return, but simulated group returns are converted back to simple returns before portfolio aggregation.
- This common target portfolio isolates risk-model performance from changes in underlying stock exposure.

### Secondary portfolios

- Cluster-balanced portfolio: equal capital across data-driven clusters and equal weight within each cluster.
- Sector-balanced portfolio: equal capital across GICS sectors and equal weight within each sector.
- These portfolios are initialised at equal group/stock weights annually and allowed to drift until the next annual rebalance. Their performance comparison is secondary.

## 5. Risk models and baselines

### Common marginal model

Each group return will use an AR(1)-GARCH(1,1) model with Student-t innovations. Standardised residuals will be transformed to pseudo-observations for copula estimation. AR is removed only after a failed or unstable fit, never solely because its p-value is insignificant. The deterministic fallback sequence and numerical acceptance criteria are frozen in `config/model_config.toml` and documented in `docs/methodology_freeze.md`.

### Preregistered model comparison

| Model | Grouping | Dependence model |
|---|---|---|
| M1 | GICS sectors | Gaussian copula |
| M2 | GICS sectors | R-vine copula |
| M3 | Data-driven clusters | Gaussian copula |
| M4 - proposed | Data-driven clusters | R-vine copula |
| M0 - benchmark | None | Rolling historical simulation of portfolio returns |

For the R-vine, pair-copula families will be restricted to independence, Gaussian, Student-t, Frank, Clayton, Gumbel, and relevant rotations. The primary vine is truncated after tree 3, with a full ten-tree robustness fit. Vine structure and pair families will be selected within each training window using the frozen Dißmann/AIC rules. Gaussian and R-vine models share marginal forecasts, 100,000 simulations, and common random numbers.

The models will generate daily one-step-ahead forecasts for:

- 95% VaR;
- 97.5% VaR, required for joint scoring with ES;
- 99% VaR; and
- 97.5% ES.

## 6. Evaluation metrics

### Primary risk-forecast evaluation

- VaR exception rate versus nominal coverage;
- Kupiec unconditional-coverage test;
- Christoffersen independence and conditional-coverage tests;
- quantile loss for 95% and 99% VaR;
- the frozen FZ0 joint 97.5% VaR-ES loss;
- average out-of-sample copula log score; and
- Diebold-Mariano comparisons of forecast losses, with multiplicity adjustment where appropriate.

The main conclusion will be based on forecast loss and full-period calibration collectively, not on a single test or favourable metric. Annual 99% coverage tests are descriptive only.
The primary model ranking will be the average rank across 95% VaR loss, 99% VaR loss, and 97.5% joint VaR-ES loss; a model with systematic coverage failure cannot be selected as best.

### Clustering evaluation

- out-of-sample within-group minus between-group Spearman dependence, with equal weight per unordered security pair and explicit singleton handling;
- year-to-year cluster stability using Adjusted Rand Index;
- similarity to GICS using Normalised Mutual Information; and
- cluster membership and economic interpretability.

### Secondary portfolio evaluation

- annualised return and volatility;
- maximum drawdown;
- downside deviation;
- diversification ratio; and
- turnover.

## 7. Decision rules

- H1 is supported if the cluster dependence gap is positive out of sample and the 95% paired circular-block-bootstrap confidence interval for `(cluster gap - GICS gap)` excludes zero under the frozen 20-day, 10,000-replication procedure.
- H2 receives full support only if all six matched R-vine-minus-Gaussian loss comparisons favour the vine and remain significant at 5% after Holm adjustment; mixed results are partial or no support.
- H3 is supported only if M4 ranks first under the primary ranking rule and its VaR forecasts do not show systematic coverage or independence failures.
- Results that are mixed, economically small, or unstable across years will be reported as such rather than classified as success.

## 8. Scope boundaries

### In scope

- daily US large-cap equity returns;
- hierarchical clustering as the primary unsupervised-learning method;
- a gated comparison with spectral clustering and PCA-plus-k-means;
- GICS comparison;
- Gaussian and R-vine dependence models;
- univariate conditional-volatility margins;
- one-day VaR/ES forecasting;
- rolling out-of-sample testing; and
- a limited secondary diversification comparison.

### Out of scope

- earnings-announcement or IV-crush forecasting;
- options data;
- NLP or company-text embeddings;
- full replication of the contextual time-series embedding paper;
- deep learning;
- intraday or high-frequency data;
- expected-return forecasting or claims of trading alpha;
- unconstrained portfolio optimisation;
- dynamic/regime-switching vine copulas;
- transaction-cost modelling beyond a simple turnover sensitivity; and
- non-equity asset classes.

## 9. Planned original contributions

1. A controlled comparison of data-driven and sector groupings while preserving the same underlying equal-weight stock portfolio.
2. A matched Gaussian-versus-R-vine comparison using identical marginal models and genuinely out-of-sample forecasts.
3. Evidence on whether clustering adds information to tail-risk modelling beyond conventional industry classification.
4. A controlled secondary comparison of alternative unsupervised stock-clustering algorithms under identical information and risk-model conditions.

## 10. Reproducibility requirement

The final project will include a frozen universe file, data dictionary, configuration file, documented preprocessing, one main executable pipeline, fixed random seeds, model diagnostics, saved result tables, and code that regenerates every reported figure and metric. API credentials will not be stored in submitted code or data files.

The detailed pre-results definitions are frozen in `config/model_config.toml` and `docs/methodology_freeze.md`. The universe-provenance, data-construction, and modelling-readiness gates must pass separately before confirmatory results are generated.
