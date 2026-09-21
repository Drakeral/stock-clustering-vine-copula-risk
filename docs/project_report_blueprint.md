# FE5110 Project Report Blueprint

Status: writing scaffold based on the frozen methodology and audited results as
at 21 September 2026. This document does not add models, recompute inference,
or change the research protocol.

## 1. Recommended title and central argument

### Recommended title

**Do Data-Driven Stock Clusters and Vine Copulas Improve Portfolio Tail-Risk
Forecasts? Evidence from a Provisional S&P 100 Research Universe**

The word `provisional` may be removed from the title only if an authorised
point-in-time constituent and GICS extract later passes strict reconciliation.
It must otherwise remain prominent in the abstract, data section, limitations,
and conclusion.

### Exact research question

For a fixed large-cap US equity universe, does replacing GICS sectors with
rolling data-driven stock clusters and replacing Gaussian dependence with a
regular vine copula improve one-day-ahead out-of-sample VaR and ES forecasts?

### Central answer

The integrated hierarchical-cluster plus truncated-vine model, M4, has the best
average rank across the three prespecified loss scores and passes the frozen
calibration-eligibility rule. That ranking does not establish either component
claim separately. Hierarchical clusters do not achieve a reliably larger
out-of-sample dependence gap than GICS, and none of the six matched
vine-versus-Gaussian forecast-loss comparisons is significant after Holm
adjustment. The defensible conclusion is therefore that M4 is the best eligible
model in this experiment, not that clustering or vine copulas are universally
superior.

### Argument spine

The report should move through four linked questions:

1. Can data-driven groups separate equity dependence more effectively than
   sectors without changing the underlying portfolio?
2. Conditional on the same grouping and marginal models, does flexible vine
   dependence improve tail-risk forecasts relative to a Gaussian copula?
3. Which complete model gives the best prespecified combination of forecast
   loss and calibration?
4. Are the conclusions stable to alternative cluster algorithms, vine depth,
   portfolio weighting, constituent-history treatment, and uncertain WBA
   acquisition consideration?

## 2. FE5110 brief requirements

The module brief does not state a report word limit. This blueprint therefore
uses an internal target of **8,000-9,000 words of main text**, excluding the
cover page, contents, references, and appendices. Confirm with the lecturer if a
separate limit is later announced.

The report must visibly address the five assessment dimensions:

| Assessment dimension | How this project should demonstrate it |
|---|---|
| Clarity | Use one common portfolio target, a five-model comparison table, a pipeline diagram, and explicit hypothesis decisions. |
| Accuracy and relevance | Preserve the frozen loss convention, simple-return portfolio arithmetic, no-look-ahead windows, multiplicity corrections, and risk-management interpretation. |
| Depth of understanding | Explain why margins and copulas are separated, how R-vines work, why VaR and ES need loss functions as well as coverage tests, and why non-significance differs from model rank. |
| Insight and innovation | Emphasise the controlled grouping comparison, matched Gaussian-vine design with common random numbers, and identical downstream evaluation of hierarchical, spectral, and PCA-k-means groupings. |
| Labour involved | Document the actual hours spent on data work, literature, programming, validation, writing, and other tasks on the cover page. Do not estimate retrospectively without records. |

The brief also requires:

- a cover page with project title, name, student ID, itemised workload, total
  workload, and original contributions;
- objectives, methodology, results, conclusions, strengths, weaknesses,
  dataset applicability, possible improvements, and extensions;
- original writing suitable for Turnitin review;
- reproducible source code and an explanation of which program produces each
  result; and
- submission of the main report and supporting appendix material through the
  designated Canvas locations.

## 3. Cover-page template

Populate the following with the student's actual details and recorded hours.
Do not invent workload figures.

| Field | Content |
|---|---|
| Project title | Do Data-Driven Stock Clusters and Vine Copulas Improve Portfolio Tail-Risk Forecasts? |
| Name | `[FULL NAME]` |
| Student ID | `[STUDENT ID]` |
| Data collection and validation | `[ACTUAL HOURS]` |
| Literature review | `[ACTUAL HOURS]` |
| Programming and testing | `[ACTUAL HOURS]` |
| Statistical analysis | `[ACTUAL HOURS]` |
| Report writing and presentation | `[ACTUAL HOURS]` |
| Other workload | `[DESCRIPTION AND ACTUAL HOURS]` |
| Total workload | `[SUM OF ACTUAL HOURS]` |

Candidate original-contribution wording, to be edited so it accurately reflects
the student's own work:

1. A controlled comparison of GICS and data-driven stock groupings that holds
   the daily equal-weight stock portfolio exactly constant across
   representations.
2. A leakage-free Gaussian-versus-R-vine tail-risk comparison using identical
   marginal models, monthly information sets, Monte Carlo draws, and
   prespecified loss functions.
3. A substantial unsupervised-learning extension comparing hierarchical,
   spectral, and PCA-plus-k-means groupings under identical windows, group
   counts, portfolio arithmetic, and risk-model evaluation.
4. A reproducible, fail-closed data and model audit covering security
   lifecycles, corporate actions, missing sessions, extreme returns, numerical
   fallbacks, seeded simulation, and artifact lineage.

## 4. Recommended word allocation

| Component | Target words | Primary purpose |
|---|---:|---|
| Abstract | 250 | Question, design, sample, principal findings, limitation |
| 1. Introduction | 700 | Motivation, research gap, questions, contribution |
| 2. Literature review | 1,200 | Position clustering, conditional volatility, copulas, and risk evaluation |
| 3. Data and research design | 900 | Universe, returns, lifecycle handling, windows, portfolio target |
| 4. Methodology | 1,700 | Clustering, margins, Gaussian/vine dependence, simulation |
| 5. Forecast evaluation and inference | 800 | VaR/ES scores, tests, multiplicity, decision rules |
| 6. Core empirical results | 1,600 | H1-H3 evidence and economic interpretation |
| 7. ML extension and robustness | 750 | Alternative groupings and sensitivity analyses |
| 8. Discussion, limitations, and extensions | 700 | Meaning, strengths, weaknesses, applicability |
| 9. Conclusion | 300 | Direct answer and restrained implications |
| **Main-text target** | **8,900** | Internal planning target, not an official limit |

Write the introduction and abstract last. Draft the data, methods, and results
first because they are already fixed by code and audit records.

## 5. Detailed section plan

### Abstract - approximately 250 words

Include, in this order:

1. the tail-risk motivation;
2. the exact controlled comparison;
3. the provisional universe and 2017-2025 sample, with 2020-2025 out-of-sample;
4. the five core models and daily VaR/ES forecasts;
5. the H1-H3 results; and
6. the provenance and statistical-power limitations.

Required numerical anchors:

- 1,508 out-of-sample dates;
- H1 difference `-0.005590`, 95% block-bootstrap interval
  `[-0.018669, 0.006569]`;
- zero of six Holm-adjusted DM comparisons favouring the vine;
- M4 average primary-score rank `1.333`, uniquely first and calibration
  eligible.

Do not claim that M4's rank proves vine superiority.

### 1. Introduction - approximately 700 words

#### Purpose

Establish why portfolio tail risk depends on both conditional marginal dynamics
and cross-sectional dependence, and why industry classifications may not be the
best statistical grouping for risk aggregation.

#### Paragraph sequence

1. VaR and ES are widely used but sensitive to volatility and dependence,
   especially in stressed markets.
2. High-dimensional stock dependence is difficult to estimate directly;
   grouping provides a controlled dimension-reduction device.
3. GICS is economically interpretable but static, whereas correlation-based
   clustering may capture dependence that cuts across sectors.
4. Gaussian copulas are parsimonious but impose restrictive dependence;
   R-vines allow heterogeneous pair families and tail asymmetry.
5. Existing comparisons often change several components simultaneously. This
   study isolates grouping and copula effects while preserving the same
   underlying portfolio, margins, windows, and random draws.
6. State the research question, H1-H3, principal contributions, and report
   structure.

#### Contribution statement

The strongest contribution is experimental control, not the discovery that one
complex model always wins. The design separates:

- grouping information: hierarchical clusters versus GICS;
- dependence specification: vine versus Gaussian within a grouping; and
- integrated model selection: M0-M4 under common loss and calibration rules.

### 2. Literature review - approximately 1,200 words

Organise by problem rather than by author.

#### 2.1 Dependence-based stock grouping

Explain correlation distance, hierarchical clustering, financial-market
taxonomies, and the distinction between economic sectors and statistical
communities. Discuss the risk of interpreting visually attractive clusters as
out-of-sample improvement.

Citation targets:

- Mantegna (1999), hierarchical structure in financial markets;
- correlation-network and econophysics literature on stock taxonomies;
- clustering validation concepts, including ARI and NMI; and
- a source explaining spectral clustering and one explaining PCA-based
  dimension reduction.

#### 2.2 Conditional volatility and marginal filtering

Explain why copula estimation requires credible marginal dynamics and why
Student-t innovations are useful for equity-return tails.

Citation targets:

- Engle (1982), ARCH;
- Bollerslev (1986), GARCH;
- Student-t conditional-return modelling; and
- probability integral transforms for copula construction.

#### 2.3 Copulas and vine dependence

Cover Sklar's theorem, Gaussian-copula restrictions, pair-copula construction,
regular vines, Dißmann structure selection, family selection, and truncation.
Explain that greater in-sample flexibility can create estimation variance and
need not improve portfolio risk forecasts.

Citation targets:

- Sklar (1959);
- Bedford and Cooke on vines;
- Aas, Czado, Frigessi and Bakken (2009) on pair-copula constructions;
- Dißmann et al. (2013) on sequential R-vine selection; and
- empirical financial applications comparing Gaussian and flexible copulas.

#### 2.4 VaR/ES evaluation

Explain why exception counts alone are insufficient, why ES is not elicitable
alone but VaR and ES are jointly elicitable, and why forecast-loss comparison
must account for serial dependence and multiple testing.

Citation targets:

- Artzner et al. on coherent risk measures;
- Kupiec (1995) and Christoffersen (1998);
- Fissler and Ziegel (2016) on joint VaR-ES scoring;
- Diebold and Mariano (1995);
- Newey and West (1987);
- Holm (1979); and
- Benjamini and Hochberg (1995).

End the literature review with the gap this experiment addresses: few studies
jointly control the grouping representation, portfolio exposure, marginal
filters, simulation randomness, and out-of-sample inference.

### 3. Data and research design - approximately 900 words

#### 3.1 Universe and provenance

State precisely:

- frozen S&P 100 research universe as at 2 January 2020;
- provisional contemporaneous Wikipedia membership/GICS source supported by
  Yahoo metadata and price-presence checks;
- no claim of licensed point-in-time membership, historical GICS, or permanent
  identifiers;
- initial 95% coverage screen over 2017-2019; and
- no later index additions.

Avoid calling Yahoo an authoritative constituent-history source.

#### 3.2 Market data and total returns

Describe unadjusted OHLCV plus dividends and corporate actions, expected XNYS
sessions, simple and log total returns, and the treatment of:

- verified BIIB halts;
- DOW legacy/new-security lineage separation;
- spin-offs as typed events rather than generic splits;
- AGN and WBA terminal consideration and cash carry; and
- reviewed OXY and BIIB extreme returns retained without winsorisation.

Use these examples to demonstrate data-engineering depth, but keep event-level
details in an appendix if space is tight.

#### 3.3 Sample split and no-look-ahead design

- market-data span: 3 January 2017 to 31 December 2025;
- initial estimation period: 2017-2019;
- out-of-sample period: 2020-2025;
- annual grouping refits using information through the prior year-end; and
- monthly marginal/copula refits using information through the preceding
  trading date.

#### 3.4 Portfolio arithmetic

Include:

\[
r_{i,t}=\exp(\ell_{i,t})-1,
\qquad
r_{g,t}=\frac{1}{n_{g,y}}\sum_{i\in g_y}r_{i,t},
\]

\[
R_{p,t}=\sum_g\frac{n_{g,y}}{N_y}r_{g,t}
=\frac{1}{N_y}\sum_i r_{i,t}.
\]

Explain that this is a daily-rebalanced equal-stock portfolio. Group-size
weights ensure GICS and cluster representations reconstruct the same realised
portfolio, so forecast differences arise from the risk models rather than
different stock exposure. Realised loss is `L_t=-R_{p,t}`.

### 4. Methodology - approximately 1,700 words

#### 4.1 GICS and hierarchical grouping

For each year, estimate training-window Spearman correlation and transform it
to:

\[
d_{ij}=\sqrt{\frac{1-\rho_{ij}}{2}}.
\]

Apply deterministic average-linkage clustering with the number of clusters set
equal to the number of non-empty GICS sectors. Explain the 80% pairwise-data
requirement and deterministic tie-breaking.

Define the pair-weighted dependence gap:

\[
G_y=\overline{\rho}_{\text{within},y}
-\overline{\rho}_{\text{between},y}.
\]

Singletons contribute no within pair but retain all between pairs.

#### 4.2 Marginal model

Present the AR(1)-GARCH(1,1)-Student-t structure compactly:

\[
x_t=\mu+\phi x_{t-1}+\varepsilon_t,
\qquad
\varepsilon_t=\sigma_t z_t,
\]

\[
\sigma_t^2=\omega+\alpha\varepsilon_{t-1}^2
+\beta\sigma_{t-1}^2.
\]

Describe the deterministic acceptance and fallback sequence without listing
every optimizer detail in the main text. Put numerical thresholds and starting
values in an appendix. Mention that only four of 1,584 core group-month fits use
EWMA, below the frozen 1% quality gate.

#### 4.3 Gaussian copula

Explain PIT transformation, normal-score transformation, sample correlation,
positive-definiteness repair, Cholesky simulation, and the matched copula log
score. State that log scores are compared only within the same grouping.

#### 4.4 Truncated R-vine

Explain:

- 11 dimensions;
- Dißmann maximum-spanning-tree selection on absolute Kendall tau;
- sequential maximum likelihood;
- AIC selection among independence, Gaussian, Student-t, Frank, and rotated
  Clayton/Gumbel families;
- primary truncation after tree 3; and
- pair-level independence and whole-vine Gaussian fallback rules.

Do not spend report space reproducing implementation internals. Focus on the
economic role of flexible conditional dependence and the bias-variance tradeoff
created by additional pair copulas.

#### 4.5 Monte Carlo forecasting

Each monthly fit uses 100,000 scenarios from `PCG64DXSM` with
`SeedSequence([5110, year, month])`. Gaussian and vine models use the same
stored independent uniforms. Explain that common random numbers reduce
simulation noise in the matched comparison; they do not make the models
statistically identical.

Simulated group log returns are transformed to simple returns before portfolio
aggregation. Forecasts are one-day-ahead 95%, 97.5%, and 99% VaR and 97.5% ES.

#### 4.6 Core models

| Model | Grouping | Dependence |
|---|---|---|
| M0 | None | Three-calendar-year rolling historical simulation |
| M1 | GICS | Gaussian copula |
| M2 | GICS | Tree-3 R-vine |
| M3 | Hierarchical clusters | Gaussian copula |
| M4 | Hierarchical clusters | Tree-3 R-vine |

This table should appear before any results.

### 5. Forecast evaluation and inference - approximately 800 words

#### 5.1 Loss functions

At confidence `c`, use quantile loss:

\[
Q_c(L,q)=(c-\mathbf{1}\{L<q\})(L-q).
\]

At `c=0.975`, `p=0.025`, use FZ0:

\[
S_{FZ0}(L,q,e)=
\frac{\mathbf{1}\{L>q\}(L-q)}{pe}
+\frac{q}{e}+\log(e)-1.
\]

State the enforced conditions `e >= q` and `e > 0`.

#### 5.2 Calibration

Report Kupiec unconditional coverage and Christoffersen independence for the
95% and 99% primary levels, with a 20-test Holm family. Treat conditional
coverage and annual 99% results as descriptive companions. With 1,508 dates,
only about 15 exceptions are expected at 99%, so annual tests have low power.

#### 5.3 Matched forecast comparison

DM loss differentials are `vine - Gaussian`, with two-sided inference,
Newey-West/Bartlett HAC lag 7, and Holm correction across six comparisons.
Direction is interpreted only after adjusted significance.

#### 5.4 Hypothesis rules

- H1: hierarchical-minus-GICS dependence gap must be positive and its paired
  circular-block-bootstrap 95% interval must exclude zero.
- H2: all six adjusted matched comparisons must significantly favour the vine
  for full support.
- H3: M4 must be uniquely first on average rank across 95% quantile loss, 99%
  quantile loss, and 97.5% FZ0, remain vine-quality eligible, and avoid adjusted
  primary calibration rejections.

Emphasise that H2 and H3 ask different questions.

### 6. Core empirical results - approximately 1,600 words

#### 6.1 Clustering diagnostics and H1

Use `reports/figures/annual_dependence_gaps.svg` as the main figure and a compact
extract from `reports/tables/clustering_diagnostics.csv`.

Required interpretation:

- hierarchical clustering outperforms GICS in some years and underperforms in
  others;
- the mean hierarchical-minus-GICS difference is `-0.005590`;
- the paired block-bootstrap interval is
  `[-0.018669, 0.006569]`; and
- H1 is not supported because the interval includes zero and the point estimate
  is not positive.

Do not say that hierarchical clusters are useless. The correct statement is
that the prespecified evidence does not establish better dependence separation
than GICS.

#### 6.2 Model performance

Use `reports/figures/core_model_score_ranks.svg` and a condensed version of
`reports/tables/core_model_performance.csv`.

Core ranking:

| Rank | Model | Average primary-score rank |
|---:|---|---:|
| 1 | M4: hierarchical + vine | 1.333 |
| 2 | M3: hierarchical + Gaussian | 1.667 |
| 3 | M2: GICS + vine | 3.333 |
| 4 | M1: GICS + Gaussian | 3.667 |
| 5 | M0: historical simulation | 5.000 |

Explain that M4's advantage is rank-based across three loss functions; it is not
uniformly lowest on every individual loss.

#### 6.3 Vine versus Gaussian and H2

Use `reports/tables/core_dm_tests.csv`.

Key result: zero of six comparisons is significant after Holm adjustment. Some
raw mean differences favour vines and others favour Gaussian copulas, but none
survives the family-wise correction. H2 therefore has no support.

Do not use unadjusted p-values or descriptive copula log scores to reverse this
decision.

#### 6.4 Calibration and H3

Use `reports/figures/core_var_exception_rates.svg` and a compact version of
`reports/tables/core_calibration.csv`.

Explain:

- M0 has adjusted Christoffersen independence rejections and is classified as
  a systematic calibration failure under the frozen rule;
- M1-M4 have no adjusted Kupiec or independence rejection at the primary
  levels; and
- M4 is uniquely first, quality eligible, and not systematically miscalibrated,
  so H3 is supported under the operational ranking rule.

Immediately qualify H3: it is a within-experiment integrated ranking, not a
general proof of vine dominance.

#### 6.5 Hypothesis summary

Include `reports/tables/hypothesis_conclusions.csv` as a three-row table near
the end of the core-results section.

### 7. ML extension and robustness - approximately 750 words

Use `reports/tables/ml_model_performance.csv` and
`reports/tables/robustness_summary.csv`. Keep this section shorter than the core
results so exploratory work does not obscure the preregistered questions.

#### 7.1 Alternative unsupervised groupings

- Spectral and PCA-plus-k-means each produce 11 non-empty groups annually.
- Spectral clustering has descriptively stronger average dependence separation
  than GICS and hierarchical grouping.
- None of the 24 downstream risk-loss comparisons is significant after
  Benjamini-Hochberg adjustment.
- These results do not revise H1-H3.

#### 7.2 Vine depth

Tree 10 increases mean fitted parameters from `40.51` to `66.22`. Its copula log
score is descriptively higher within both groupings, but no adjusted VaR/ES loss
difference is significant. Interpret this as no forecast-loss evidence that the
additional complexity is worthwhile in this sample.

#### 7.3 Portfolio weighting

Equal-group portfolios are different realised portfolios and therefore are
ranked only within target. Gaussian models rank first for both targets; no
adjusted vine-Gaussian difference is significant.

#### 7.4 Constituent-history robustness

`M0_CC` changes 710 of 1,508 forecasts but no exception count. The realised-
history M0 has lower mean loss on all three scores; only the 99% quantile-loss
difference remains significant after the separate Holm correction.

#### 7.5 WBA DAP valuation

The `$0` and `$3` endpoint valuations do not alter H2, H3, exception counts, or
the exploratory ML conclusions. State that this is an endpoint sensitivity, not
a stochastic valuation model for the DAP right.

### 8. Discussion, limitations, and extensions - approximately 700 words

#### Interpretation

The likely economic interpretation is that hierarchical grouping contributes
enough representation change for M3 and M4 to rank well, but the direct H1 test
does not show superior generic dependence separation. Likewise, the vine may
improve some loss dimensions descriptively, but its extra flexibility does not
produce statistically reliable matched gains over Gaussian dependence after
multiplicity control.

Discuss the bias-variance tradeoff: flexible dependence can fit richer joint
behaviour but must estimate more parameters from rolling windows. The tree-10
robustness supports caution about complexity.

#### Strengths

- genuinely out-of-sample annual and monthly windows;
- identical realised portfolio across primary grouping representations;
- matched margins and common random numbers;
- explicit fallback and numerical-quality gates;
- multiple-testing control and loss-based inference;
- substantial alternative-clustering and robustness work; and
- reproducible artifacts with hashes and tests.

#### Limitations

1. The universe/GICS provenance is provisional and not licensed point-in-time
   evidence.
2. The initial screen and fixed-universe design may introduce selection and
   survivorship limitations despite explicit lifecycle handling.
3. There are only 1,508 out-of-sample dates and roughly 15 expected 99% VaR
   exceptions.
4. Group aggregation may hide security-level tail dependence.
5. Copulas and marginal parameters are piecewise fixed between scheduled
   refits; no explicit regime-switching dependence is modelled.
6. The primary portfolio is stylised, equal weighted, and daily rebalanced with
   no transaction costs.
7. The WBA DAP sensitivity brackets endpoints rather than modelling the right's
   payoff distribution.
8. Results concern one US large-cap universe and should not be generalised to
   other markets, asset classes, or trading profitability.

#### Extensions

Prioritise extensions that address a limitation:

1. licensed point-in-time constituent/GICS reconciliation;
2. a broader multi-index or international sample;
3. regime-dependent or time-varying copulas under a separately frozen protocol;
4. security-level or higher-dimensional sparse vine alternatives;
5. transaction-cost-aware rebalancing frequencies; and
6. economic-capital or stress-testing applications beyond one-day VaR/ES.

Do not promise deep learning or additional methods merely to increase scope.

### 9. Conclusion - approximately 300 words

Answer the research question directly:

- hierarchical clustering did not significantly improve dependence separation
  over GICS under H1;
- truncated vines did not significantly beat matched Gaussian copulas under H2;
- M4 nevertheless ranked first and satisfied the frozen H3 eligibility rule;
- ML and robustness exercises did not overturn the core conclusions; and
- all claims remain provisional because the universe provenance is not
  licensed point-in-time evidence.

End with the methodological insight: sophisticated integrated models can rank
well even when neither individual modelling component demonstrates universal
superiority. Controlled design and disciplined inference matter as much as
model flexibility.

## 6. Table and figure plan

| Order | Item | Source | Main text or appendix | Purpose |
|---:|---|---|---|---|
| Figure 1 | Research pipeline and information timeline | Create descriptively from the frozen protocol; no new inference | Main text | Make annual/monthly/daily timing and no-look-ahead design immediately clear |
| Table 1 | Universe, sample, and portfolio design | `docs/data_dictionary.md`, construction audits | Main text | Define the data and provenance boundary |
| Table 2 | M0-M4 model definitions | Frozen protocol | Main text | Establish controlled comparisons before results |
| Figure 2 | Annual dependence gaps | `reports/figures/annual_dependence_gaps.svg` | Main text | Show H1 variation over time |
| Table 3 | H1 estimate and interval | `reports/tables/hypothesis_conclusions.csv` | Main text | Report the prespecified decision |
| Figure 3 | Core model score ranks | `reports/figures/core_model_score_ranks.svg` | Main text | Show integrated ranking |
| Table 4 | Core performance summary | `reports/tables/core_model_performance.csv` | Main text | Provide losses, exceptions, and rank |
| Table 5 | Matched vine-Gaussian DM tests | `reports/tables/core_dm_tests.csv` | Main text, possibly condensed | Support H2 |
| Figure 4 | VaR exception rates | `reports/figures/core_var_exception_rates.svg` | Main text | Compare observed and nominal rates |
| Table 6 | Calibration tests | `reports/tables/core_calibration.csv` | Main text condensed; full table in appendix | Support H3 eligibility |
| Table 7 | H1-H3 decisions | `reports/tables/hypothesis_conclusions.csv` | Main text | Close core results clearly |
| Table 8 | ML model performance | `reports/tables/ml_model_performance.csv` | Appendix or compact main-text extract | Present exploratory extension |
| Table 9 | Robustness summary | `reports/tables/robustness_summary.csv` | Main text | Demonstrate conclusion stability |

Every caption should state whether the evidence is confirmatory,
confirmatory-methodological, exploratory, or robustness-only. Do not copy raw
CSV tables into the report without rounding and explanatory captions.

## 7. Evidence ledger

Use this ledger while drafting to prevent unsupported claims.

| Claim | Authoritative project source |
|---|---|
| Reporting scope is provisional | `data/audit/current_gate_status.json` |
| Universe and return construction passed the implemented gates | `data/audit/data_quality_report.json` and `data/audit/portfolio_arithmetic.json` |
| H1-H3 decisions | `reports/tables/hypothesis_conclusions.csv` |
| Core loss values and ranks | `reports/tables/core_model_performance.csv` |
| Matched H2 inference | `reports/tables/core_dm_tests.csv` |
| Calibration evidence | `reports/tables/core_calibration.csv` |
| Annual grouping diagnostics | `reports/tables/clustering_diagnostics.csv` |
| ML results | `reports/tables/ml_model_performance.csv` |
| Robustness conclusions | `reports/tables/robustness_summary.csv` |
| Exact frozen formulas and rules | `docs/methodology_freeze.md` |
| Numerical and code audit | `docs/quantitative_code_audit.md` |
| Output/source hashes | `reports/report_manifest.json` and `data/manifests/run_manifest.json` |

If prose and a generated table ever disagree, stop and resolve the discrepancy
against the passed audit JSON rather than editing the number manually.

## 8. Interpretation guardrails

The report must not make any of the following claims:

- “Hierarchical clustering is better than GICS.” H1 is not supported.
- “Vine copulas outperform Gaussian copulas.” H2 has no adjusted support.
- “M4 proves H1 and H2.” H3 is a separate integrated ranking decision.
- “The sample is an authoritatively verified historical S&P 100 universe.” The
  current universe is provisional.
- “The models predict stock returns or generate alpha.” The project forecasts
  portfolio tail risk, not expected returns.
- “A non-significant test proves equality.” It shows insufficient evidence of a
  difference under this design and sample.
- “The 99% annual coverage results are decisive.” The exception count is too
  small for strong annual inference.
- “Copula log scores compare GICS and cluster representations.” They are valid
  only within matched groupings.
- “The WBA endpoints value the DAP.” They are valuation sensitivities around
  the primary `$0.53` assumption.

## 9. Literature-review working list

The following are citation targets, not a final bibliography. Verify exact
titles, editions, page numbers, DOIs, and the required citation style before
submission.

| Topic | Starting sources |
|---|---|
| Financial clustering | Mantegna (1999); subsequent correlation-network and market-taxonomy studies |
| Cluster agreement | Hubert and Arabie on adjusted Rand index; normalized mutual-information references |
| Spectral clustering | Ng, Jordan and Weiss (2002) |
| PCA and k-means++ | Jolliffe on PCA; Arthur and Vassilvitskii (2007) |
| Conditional volatility | Engle (1982); Bollerslev (1986) |
| Copula foundations | Sklar (1959); financial copula applications such as Patton |
| Vine copulas | Bedford and Cooke; Aas et al. (2009); Dißmann et al. (2013) |
| Risk measures | Artzner et al.; McNeil, Frey and Embrechts |
| VaR backtesting | Kupiec (1995); Christoffersen (1998) |
| Joint VaR-ES scoring | Fissler and Ziegel (2016) |
| Forecast comparison | Diebold and Mariano (1995); Newey and West (1987) |
| Multiple testing | Holm (1979); Benjamini and Hochberg (1995) |

For each source, write two notes before drafting prose:

1. What precise methodological claim does the source support?
2. How does this project adopt, modify, test, or challenge that claim?

This prevents the literature review from becoming a list of summaries.

## 10. Reproducibility appendix plan

The appendix should contain:

1. repository URL and release commit or tag;
2. software versions and `uv sync --frozen` setup;
3. a concise pipeline map showing which command produces each reported output;
4. frozen configuration files and seed rules;
5. data dictionary and lifecycle/corporate-action policies;
6. full calibration, DM, ML, and robustness tables omitted from the main text;
7. numerical fallback and quality-gate summaries;
8. code-quality and test evidence; and
9. a data-availability statement distinguishing redistributable public
   material from provider/licence-restricted raw data.

The assignment asks for reproducible code and data. This does not override
provider licences. Submit permitted code, manifests, derived summaries, and
clear retrieval/rebuild instructions; confirm separately whether any raw market
or vendor-derived data may be distributed through Canvas.

## 11. Final writing and submission checklist

### Content

- [ ] Research question and H1-H3 appear verbatim and before results.
- [ ] Original contributions are explicit and truthful.
- [ ] The common-portfolio control is explained clearly.
- [ ] Every equation defines its symbols and sign convention.
- [ ] Every model comparison is matched to its correct grouping and portfolio.
- [ ] Adjusted rather than raw p-values drive decisions.
- [ ] Strengths, weaknesses, dataset applicability, and extensions are present.
- [ ] Provisional provenance appears in the abstract, data section, limitations,
      conclusion, and relevant captions.
- [ ] All writing has been reviewed and rewritten in the student's own voice.

### Figures and tables

- [ ] Figures remain legible in grayscale and at final page width.
- [ ] Axes, confidence levels, units, and sample sizes are labelled.
- [ ] Captions state the analysis role and adjustment method.
- [ ] Main-text tables are rounded consistently without altering stored values.
- [ ] Full-precision reproducible CSV files remain in the appendix/repository.

### Reproducibility

- [ ] The final report cites the GitHub repository and an immutable commit/tag.
- [ ] A clean clone passes the portable test suite.
- [ ] The local artifact-complete checkpoint and lineage verifier pass.
- [ ] No credentials, `.env` files, or restricted raw data are submitted.
- [ ] Data-access and licence restrictions are disclosed.

### FE5110 submission administration

- [ ] Cover page contains name, student ID, itemised actual workload, total
      workload, and original contributions.
- [ ] Main PDF filename follows `[Full Name] ([Student ID]) FE5110 Term 2610.pdf`.
- [ ] Main report is below the Turnitin 40 MB limit.
- [ ] Supporting code/data appendix is submitted to the designated Canvas
      appendix location.
- [ ] Final submission is completed by **Friday, 6 November 2026**.
- [ ] The Canvas/Turnitin digital receipt is retained.

## 12. Recommended drafting order

1. Data and research design.
2. Methodology.
3. Forecast evaluation and inference.
4. Core results.
5. ML extension and robustness.
6. Discussion and limitations.
7. Literature review, integrated around the completed methods.
8. Introduction and conclusion.
9. Abstract.
10. Cover page, appendices, cross-references, and final compliance review.

The next concrete writing deliverable should be Sections 3-5. They are the most
objective parts of the report and can be drafted directly from the frozen
methodology before stylistic work begins on the introduction.
