# Do Data-Driven Stock Clusters and Vine Copulas Improve Portfolio Tail-Risk Forecasts?

## Evidence from a Provisional S&P 100 Research Universe

> **Draft status:** Sections 3-5 are drafted from the frozen methodology and
> audited project artifacts. Sections 1, 2, and 6-9 remain to be written. This
> source document deliberately omits personal identifiers; the student's name,
> student ID, recorded workload, and final filename will be inserted only into
> the local submission documents.

## Abstract

*To be drafted after the results, discussion, and conclusion are complete.*

## 1. Introduction

*To be drafted after Sections 3-9 are stable.*

## 2. Literature Review

*To be drafted as an integrated review of financial clustering, conditional
volatility, copula dependence, and VaR/ES forecast evaluation.*

## 3. Data and Research Design

### 3.1 Research universe and provenance

The study uses a frozen large-cap US equity research universe intended to
represent the securities in the S&P 100 on 2 January 2020. The membership and
GICS sector labels were fixed before the out-of-sample model comparisons. No
later index additions were introduced. This design avoids changing the research
question in response to subsequent index composition, while annual active sets
allow securities with documented terminal events to leave the portfolio at the
next calendar-year rebalance.

The provenance of the universe requires an explicit qualification. Membership
is based on an archived contemporaneous Wikipedia revision dated 21 November
2019, while Yahoo metadata and price availability on 2 January 2020 provide a
secondary cross-check. These sources do not constitute licensed point-in-time
evidence of index membership, historical GICS classification, or permanent
security identity. Protocol amendment PA-001 permits modelling on this basis,
but the universe gate remains `provisional_authorized_yahoo` and every empirical
result is therefore reported as a provisional research result. The analysis
does not describe Yahoo as an authoritative historical constituent source.

The initial source table contained 101 candidate securities. A security entered
the fixed research universe only if both price and return coverage reached 95%
over the initial period from 3 January 2017 to 31 December 2019. One candidate,
Dow Inc. (DOW), was excluded because the new company's price history begins in
April 2019 and covers only approximately 25.2% of the initial price window. The
legacy Dow Chemical series was not joined to the new Dow Inc. series; doing so
would create an economically invalid return across two different security
lineages. The final modelling universe consequently contains 100 securities
across 11 non-empty GICS sectors. The active set contains 100 securities in
2020 and 98 from the 2021 annual rebalance onward following documented terminal
events. GOOG and GOOGL remain separate listed securities in the primary
analysis. A robustness construction replaces them with one equally weighted
Alphabet issuer position.

### 3.2 Market data and total-return construction

The market-data sample runs from 3 January 2017 through 31 December 2025 and
contains 2,262 expected New York Stock Exchange sessions. The raw inputs are
daily unadjusted open, high, low, close, and volume observations together with
dividend and corporate-action reference records. The use of unadjusted prices
keeps the economic components of total return explicit rather than relying on
an undocumented vendor-adjusted close.

For security \(i\) on date \(t\), the close-to-close simple total return is
constructed from the price change and applicable cash or security
distributions. The corresponding log total return is

\[
\ell_{i,t}=\log(1+r_{i,t}).
\]

Cross-sectional portfolio calculations use simple returns, not averages of log
returns. Log returns are retained for time-series marginal modelling only. This
distinction is important: an arithmetic mean of stock log returns represents a
geometric aggregation and does not equal the simple return on an equal-weight
portfolio.

An independently generated exchange-session grid is combined with explicit
listing and terminal dates. Each absent security-date observation must be
classified as pre-inception, a verified halt, terminal cash, removed, or an
unexplained active-period absence. Unexplained active-period absences fail the
construction gate. Two missing BIIB sessions, 6 November 2020 and 9 June 2023,
were verified as trading halts. Each receives a synthetic stale-price return of
zero, while the cumulative market movement remains in the observed return when
trading resumes. This treatment avoids both deleting an active portfolio
position and spreading the resumption movement over a day when the position
could not trade.

Corporate actions are reconciled to typed economic events rather than treated
as interchangeable vendor split factors. The classifications distinguish
literal splits, stock dividends, spin-offs, merger exchanges, and cash
acquisitions. Spin-offs use documented child-share ratios and values; a provider
continuity factor cannot be applied a second time. Allergan's terminal value is
booked as USD 120.30 in cash plus 0.866 times AbbVie's event-date close. The
primary Walgreens Boots Alliance terminal value on 28 August 2025 is USD 11.45
cash plus a USD 0.53 market-implied value for the non-transferable DAP right.
After terminal consideration is booked, the proceeds earn a zero return until
the next annual rebalance. Endpoint sensitivities value the DAP at zero and at
its contractual USD 3 cap, but do not alter the primary data construction.

Extreme returns are retained only after an observation-level disposition. The
OXY total return of -50.48% on 9 March 2020 was reconciled to the oil-price shock,
raw OHLC observations, and the applicable dividend. The BIIB return of +43.97%
on 4 November 2020 was reconciled to the release of FDA briefing materials.
Both are treated as genuine economic observations and are not winsorised. This
is especially important for a tail-risk study, where mechanically suppressing
large valid returns would bias both marginal tail fitting and dependence
estimation.

The final construction audit reports zero duplicate security-date observations,
zero non-positive closing prices, no unexplained active-period gaps, and no
unmatched or unclassified reference events. All 3,175 dividend and action
records are classified as applied, duplicate-aggregated, or out-of-segment.
These checks support the computational integrity of the return panel but do not
remove the separate point-in-time universe-provenance limitation.

### 3.3 Estimation and evaluation windows

The first three years, 2017-2019, form the initial estimation history. The
out-of-sample evaluation period is 2 January 2020 through 31 December 2025,
comprising 1,508 trading dates. Group definitions are re-estimated on the first
trading day of each year using the left-closed, right-open interval beginning
three calendar years earlier and ending immediately before the rebalance date.
Marginal and copula models are refitted on the first trading day of each month
using information ending on the preceding trading date. Between monthly
refits, marginal parameters remain fixed while conditional mean and volatility
states update daily using only previously observed returns.

This timing convention separates estimation from evaluation at every stage. A
2020 cluster, for example, uses no 2020 return, and a forecast for date (t)
cannot use the realised return or any transformation of information first
observed on (t). Historical simulation uses an analogous three-calendar-year
window, \([t-3\text{ years},t)\), with at least 700 valid observations.

### 3.4 Common portfolio target

Let \(r_{i,t}=\exp(\ell_{i,t})-1\) be the simple total return of security \(i\).
For group \(g\) in calendar year \(y\), the equal-weight group return is

\[
r_{g,t}=\frac{1}{n_{g,y}}\sum_{i\in g_y}r_{i,t},
\]

where \(n_{g,y}\) is the number of active securities in the group. The primary
portfolio is rebalanced to equal security weights every day. Giving each group
the weight \(n_{g,y}/N_y\) yields

\[
R_{p,t}
=\sum_g\frac{n_{g,y}}{N_y}r_{g,t}
=\frac{1}{N_y}\sum_i r_{i,t}.
\]

Thus the GICS and data-driven representations reconstruct the same realised
equal-weight stock portfolio. This identity isolates the effects of grouping
and dependence modelling: M1-M4 cannot obtain a different realised portfolio
merely by assigning the same stocks to different groups. Verified-halt and
terminal-cash positions remain in \(n_{g,y}\) with a zero return until annual
removal. Across 2,261 defined return dates, direct stock-level and group-size-
weighted portfolio returns agree to a maximum error of
\(5.55\times10^{-17}\), well inside the frozen \(10^{-12}\) tolerance.

The realised loss used throughout the evaluation is

\[
L_t=-R_{p,t},
\]

so positive values denote portfolio losses. The secondary equal-group
portfolios are distinct robustness exercises. They reset to equal group weights
annually and drift between annual rebalances, and are never ranked directly
against the primary portfolio because their realised return series differ.

## 4. Methodology

### 4.1 Sector and hierarchical representations

GICS sectors provide the economically interpretable benchmark grouping. The
data-driven alternative is estimated separately for each evaluation year from
the preceding three-calendar-year stock-return window. Spearman rank
correlations are used because the grouping objective concerns monotonic
dependence and should be less sensitive than Pearson correlation to the exact
magnitude of a small number of extreme returns. Each pair must have valid joint
observations on at least 80% of the annual training dates; the procedure fails
rather than silently deleting an ineligible pair.

Following the correlation-distance construction used in financial clustering
(Mantegna, 1999), the annual correlation matrix is transformed as

\[
d_{ij}=\sqrt{\frac{1-\rho_{ij}}{2}}.
\]

This maps perfect positive correlation to zero distance and perfect negative
correlation to one. Average-linkage agglomerative clustering is then applied.
The Lance-Williams update is weighted by cluster size, so every original
cross-cluster security pair receives equal weight. Exact distance ties and the
final cluster labels are resolved by lexicographically ordered member tuples,
making the annual assignment deterministic.

The number of hierarchical clusters equals the number of non-empty GICS
sectors, which is 11 in every evaluation year. Holding dimension constant
prevents the hierarchical method from improving fit mechanically through a
larger number of groups. Annual assignments use only information available
before the rebalance date and remain fixed through that calendar year.

The primary grouping statistic is an out-of-sample within-minus-between
dependence gap. For grouping method \(m\) and evaluation year \(y\),

\[
G_{m,y}=
\overline{\rho}^{\,\mathrm{OOS}}_{\mathrm{within},m,y}
-\overline{\rho}^{\,\mathrm{OOS}}_{\mathrm{between},m,y}.
\]

Every unordered security pair receives equal weight. Singleton groups
contribute no within-group pair and are not inserted as zero-dependence groups;
their cross-group pairs remain in the between-group mean. Reports also retain
pair counts, the number of groups contributing within pairs, consecutive-year
Adjusted Rand Index on intersected active sets, and annual normalised mutual
information relative to GICS.

### 4.2 Marginal return dynamics

Copulas describe dependence after marginal dynamics have been filtered. For
each of the 11 group return series, the modelling input is
\(x_{g,t}=\log(1+r_{g,t})\). The primary marginal model is an
AR(1)-GARCH(1,1) process with Student-t innovations (Engle, 1982; Bollerslev,
1986):

\[
x_{g,t}=\mu_g+\phi_g x_{g,t-1}+\varepsilon_{g,t},
\qquad
\varepsilon_{g,t}=\sigma_{g,t}z_{g,t},
\]

\[
\sigma_{g,t}^{2}=\omega_g
+\alpha_g\varepsilon_{g,t-1}^{2}
+\beta_g\sigma_{g,t-1}^{2}.
\]

The Student-t innovation \(z_{g,t}\) is standardised to unit variance. Returns
are multiplied by 100 during numerical estimation and converted back to decimal
units in stored forecasts. An insignificant AR coefficient is not a reason to
discard the AR term, because model selection based on repeated in-sample
significance testing would make the forecast procedure data dependent in a way
not specified by the protocol.

Each fit must satisfy finite likelihood and information criteria,
\(|\phi|<0.98\), non-negative variance parameters,
\(\alpha+\beta<0.999\), Student-t degrees of freedom above 2.1, and a positive
finite forecast variance. Failure triggers a deterministic sequence: a second
AR-GARCH-t attempt with controlled starting values, a constant-mean GARCH-t
fit, and finally EWMA with decay parameter \(\lambda=0.94\) and empirical
standardised innovations. A monthly EWMA fraction above 1% fails the modelling
quality gate. Numerical fallbacks are therefore observable model outcomes, not
silent exception handling.

For an accepted Student-t model, standardised residuals are mapped through the
fitted unit-variance Student-t distribution. For EWMA, they are mapped through
the frozen training-sample empirical distribution using midranks. Probability
integral transforms (PITs) are clipped to
\([10^{-6},1-10^{-6}]\) to avoid infinite inverse-CDF values. The monthly
copula training panel retains only dates on which all 11 group PITs are
available and strictly precede the refit date.

### 4.3 Gaussian dependence benchmark

By Sklar's theorem, a multivariate distribution can be represented through its
marginal distributions and a copula that contains the remaining dependence.
The Gaussian benchmark transforms each aligned PIT \(u_{g,t}\) to a normal
score \(z_{g,t}=\Phi^{-1}(u_{g,t})\) and estimates the sample Pearson
correlation matrix of those scores.

The estimated matrix is symmetrised before simulation. If its minimum
eigenvalue is below \(10^{-8}\), eigenvalues are floored at \(10^{-8}\), the
matrix is reconstructed, and the diagonal is rescaled to one. Any adjustment
larger than \(10^{-12}\) is reported, and a repaired condition number above
\(10^{10}\) fails the Gaussian quality gate. Cholesky factorisation generates
correlated Gaussian scores, which are transformed to uniforms and then through
the inverse conditional marginal distributions. The Gaussian copula density
also supplies a daily dependence log score, but that score is compared only
with a vine fitted to the same grouping representation.

### 4.4 Regular-vine dependence model

A regular vine decomposes an 11-dimensional copula density into a sequence of
conditional bivariate copulas. This pair-copula construction allows different
pairs to exhibit different forms of symmetry and tail dependence (Aas et al.,
2009). The primary model is truncated after tree 3 to balance flexibility
against the estimation burden of a full ten-tree vine.

Structure selection follows the sequential Dißmann procedure (Dißmann et al.,
2013). Tree 1 is the maximum-spanning tree under absolute empirical Kendall
tau. Subsequent trees obey the vine proximity condition and are selected from
conditional-dependence estimates produced by the preceding trees. For each
edge, sequential maximum likelihood and AIC select among independence,
Gaussian, Student-t, Frank, and the relevant rotations of Clayton and Gumbel
copulas. Copula-t degrees of freedom are restricted to \([2.1,50]\).

Failure of an individual pair fit replaces that pair with independence and is
recorded. If more than 10% of a month's pair fits fail, or the monthly structure
cannot be estimated, the corresponding Gaussian copula supplies that month's
forecasts. If whole-vine fallback covers more than 1% of out-of-sample dates,
the vine is ineligible to be declared the best model. A full ten-tree vine is
estimated separately as an exploratory robustness model using identical
monthly information sets and simulation seeds; it does not overwrite the
primary tree-3 artifacts.

### 4.5 Simulation and daily forecasting

Each monthly copula fit uses 100,000 Monte Carlo scenarios generated by NumPy's
`PCG64DXSM` bit generator from
`SeedSequence([5110, year, month])`. Gaussian and vine specifications use the
same stored independent-uniform matrix within each month. These common random
numbers reduce simulation noise in matched model differences without changing
the distribution implied by either copula.

The simulated uniforms are inverted through each group's conditional marginal
distribution. Student-t innovations are multiplied by
\(\sqrt{(\nu-2)/\nu}\) to impose unit variance; an EWMA marginal uses the
inverted empirical CDF of its complete frozen innovation sample. The resulting
group log returns are converted to simple returns using `expm1` before they are
aggregated with annual group-size weights. This reproduces the same economic
portfolio convention used for realised returns.

Every model supplies one-day-ahead 95%, 97.5%, and 99% VaR forecasts and 97.5%
Expected Shortfall. Parameters and dependence structures remain fixed within a
month, but conditional marginal states are updated each day from information
available through the preceding session.

### 4.6 Core model comparison

The empirical design contains one non-parametric benchmark and four matched
copula specifications:

| Model | Grouping | Margins and dependence |
|---|---|---|
| M0 | None | Rolling historical simulation of the primary portfolio |
| M1 | GICS sectors | AR-GARCH-t margins and Gaussian copula |
| M2 | GICS sectors | AR-GARCH-t margins and tree-3 R-vine |
| M3 | Hierarchical clusters | AR-GARCH-t margins and Gaussian copula |
| M4 | Hierarchical clusters | AR-GARCH-t margins and tree-3 R-vine |

M0 uses the canonical realised daily-rebalanced portfolio series over the
three-calendar-year window. Its VaR is the empirical order statistic at
\(\lceil nc\rceil\) for confidence level \(c\). ES integrates the empirical
quantile function over the upper loss tail and fractionally weights the boundary
observation when \(n(1-c)\) is not an integer. M0 has no grouping, marginal
filter, copula, simulation seed, or copula log score.

The comparisons are intentionally factorial. M2 versus M1 isolates dependence
specification within GICS; M4 versus M3 does the same within hierarchical
clusters. Comparisons across M1/M2 and M3/M4 combine representation and
dependence effects and therefore inform the integrated ranking rather than a
single component claim.

## 5. Forecast Evaluation and Inference

### 5.1 Loss and exception conventions

All forecasts use the positive-loss convention. At confidence level \(c\), the
VaR forecast \(q_{c,t}\) is an upper quantile of \(L_t\), and an exception occurs
when \(L_t>q_{c,t}\). VaR accuracy is evaluated using the quantile loss

\[
Q_c(L_t,q_{c,t})
=\left(c-\mathbf{1}\{L_t<q_{c,t}\}\right)
\left(L_t-q_{c,t}\right).
\]

Lower average loss indicates a more accurate quantile forecast. The core
ranking uses quantile loss at 95% and 99%.

Expected Shortfall cannot be ranked in isolation by a strictly consistent
point-forecast score, but VaR and ES are jointly elicitable under regularity
conditions (Fissler and Ziegel, 2016). At confidence
\(c=0.975\), tail probability \(p=0.025\), VaR \(q_t\), and positive-loss ES
\(e_t\), the project uses the frozen FZ0 score (Patton, Ziegel and Chen, 2019)

\[
S_{FZ0}(L_t,q_t,e_t)=
\frac{\mathbf{1}\{L_t>q_t\}(L_t-q_t)}{p e_t}
+\frac{q_t}{e_t}+\log(e_t)-1,
\]

subject to \(e_t\ge q_t\) and \(e_t>0\). Lower mean FZ0 is better. The three
primary losses are therefore 95% quantile loss, 99% quantile loss, and 97.5%
FZ0.

### 5.2 VaR calibration

The Kupiec test evaluates whether the full-period exception frequency equals
the nominal tail probability (Kupiec, 1995). Frequency alone does not detect
clusters of exceptions, so the Christoffersen test evaluates first-order
independence in the binary exception sequence (Christoffersen, 1998). The
conditional-coverage statistic is reported as an unadjusted descriptive
companion rather than an additional confirmatory test.

The confirmatory calibration family contains 20 tests: five models, two primary
VaR levels (95% and 99%), and the Kupiec and Christoffersen-independence tests.
Holm's step-down procedure controls family-wise error at 5% (Holm, 1979).
Annual coverage
results at 95%, 97.5%, and 99% are descriptive only. With 1,508 evaluation
dates, the expected number of 99% exceptions is approximately 15; subdividing
that count by year produces tests with little power. Forecast losses and
full-period matched comparisons consequently carry more evidential weight.

### 5.3 Matched predictive-accuracy tests

For a given grouping and score (S), the daily loss differential is

\[
d_t=S_{\mathrm{vine},t}-S_{\mathrm{Gaussian},t}.
\]

A negative mean favours the vine. The Diebold-Mariano test evaluates equal
predictive accuracy using a two-sided alternative (Diebold and Mariano, 1995).
Because daily loss differentials may be serially correlated, the variance of
their sample mean uses the Newey-West estimator with Bartlett weights and lag
7 (Newey and West, 1987). Direction is examined only after the two-sided result
has been evaluated.

The H2 family contains six comparisons: M2 versus M1 and M4 versus M3 for each
of the three primary losses. Holm adjustment controls family-wise error at 5%.
Copula log scores are descriptive and are compared only within a matched
grouping because GICS and hierarchical models have different multivariate group
return representations.

### 5.4 Clustering inference

H1 concerns the annual difference

\[
\Delta_y=G_{\mathrm{hierarchical},y}-G_{\mathrm{GICS},y}.
\]

The point estimate is the equally weighted mean of the six annual differences.
Uncertainty is estimated with 10,000 paired circular moving-block bootstrap
replications using blocks of 20 trading days. Entire daily cross-sectional
return vectors are resampled so that contemporaneous dependence is preserved.
Blocks wrap within an evaluation year but never cross year boundaries, and the
same resampled indices are applied to both grouping methods. If a resample
makes a retained security constant and produces undefined Spearman
correlations, the whole paired replication is discarded and the deterministic
`PCG64DXSM` stream continues. H1 requires both a positive point estimate and a
strictly positive two-sided 95% percentile interval.

### 5.5 Prespecified decisions and integrated ranking

H2 receives full support only if all six Holm-adjusted matched comparisons are
significant and have lower vine loss. It receives partial support if at least
one adjusted comparison favours the vine and none favours the Gaussian model;
all other patterns are classified as no support. This demanding rule prevents
selective emphasis on one confidence level or scoring function.

The overall model ranking gives equal weight to each model's rank on the three
primary mean losses. H3 is supported only if M4 is uniquely first, passes the
vine-quality gate, and has no Holm-adjusted Kupiec or Christoffersen-
independence rejection at either primary VaR level. Any such adjusted rejection
is the operational definition of systematic calibration failure.

H2 and H3 answer different questions. H2 tests whether vines consistently
improve matched forecast loss after multiplicity adjustment. H3 asks whether
the complete hierarchical-vine specification is the best eligible model under
the prespecified combined ranking and calibration rule. A favourable H3 result
cannot be used to retroactively declare H1 or H2 supported.

## 6. Core Empirical Results

*To be drafted from the deterministic report tables after Sections 3-5 are
reviewed.*

## 7. ML Extension and Robustness

*To be drafted after the core-results narrative is stable.*

## 8. Discussion, Limitations, and Extensions

*To be drafted after the results and robustness sections are complete.*

## 9. Conclusion

*To be drafted last, before the abstract.*

## Working References for Sections 3-5

The final bibliography style remains to be selected. These entries are the
verified working sources cited in the current draft.

- Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009). Pair-copula
  constructions of multiple dependence. *Insurance: Mathematics and
  Economics*, 44(2), 182-198.
  <https://doi.org/10.1016/j.insmatheco.2007.02.001>
- Bollerslev, T. (1986). Generalized autoregressive conditional
  heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
  <https://doi.org/10.1016/0304-4076(86)90063-1>
- Christoffersen, P.F. (1998). Evaluating interval forecasts.
  *International Economic Review*, 39(4), 841-862.
  <https://doi.org/10.2307/2527341>
- Diebold, F.X. and Mariano, R.S. (1995). Comparing predictive accuracy.
  *Journal of Business & Economic Statistics*, 13(3), 253-263.
  <https://doi.org/10.1080/07350015.1995.10524599>
- Dißmann, J., Brechmann, E.C., Czado, C. and Kurowicka, D. (2013). Selecting
  and estimating regular vine copulae and application to financial returns.
  *Computational Statistics & Data Analysis*, 59, 52-69.
  <https://doi.org/10.1016/j.csda.2012.08.010>
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation. *Econometrica*,
  50(4), 987-1007.
  <https://doi.org/10.2307/1912773>
- Fissler, T. and Ziegel, J.F. (2016). Higher order elicitability and Osband's
  principle. *The Annals of Statistics*, 44(4), 1680-1707.
  <https://doi.org/10.1214/16-AOS1439>
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65-70.
  <https://doi.org/10.2307/4615733>
- Kupiec, P.H. (1995). Techniques for verifying the accuracy of risk
  measurement models. *The Journal of Derivatives*, 3(2), 73-84.
  <https://doi.org/10.3905/jod.1995.407942>
- Mantegna, R.N. (1999). Hierarchical structure in financial markets.
  *The European Physical Journal B*, 11, 193-197.
  <https://doi.org/10.1007/s100510050929>
- Newey, W.K. and West, K.D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica*, 55(3), 703-708.
  <https://doi.org/10.2307/1913610>
- Patton, A.J., Ziegel, J.F. and Chen, R. (2019). Dynamic semiparametric models
  for expected shortfall (and Value-at-Risk). *Journal of Econometrics*,
  211(2), 388-413.
  <https://doi.org/10.1016/j.jeconom.2018.10.008>
- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
  *Publications de l'Institut de Statistique de l'Université de Paris*, 8,
  229-231.
