# Frozen Pre-Modelling Methodology

Status: frozen before any out-of-sample model fitting or comparison.

The machine-readable source of truth is `config/model_config.toml`. The tested
implementations of the return arithmetic, forecast scores, historical simulation,
dependence gap, resampling, and multiplicity corrections are in
`scripts/research_methods.py`.

## Protocol amendment PA-001: provenance scope

On 2 September 2026, the project user authorized the frozen contemporaneous
Wikipedia membership/GICS snapshot, supported by the Yahoo reference manifest,
as the provisional project foundation. This amendment permits model fitting and
the pre-specified comparisons below, but does not convert Yahoo into a licensed
or authoritative point-in-time constituent/GICS source. Until licensed
reconciliation passes, every result has reporting scope
`provisional_research_results`; references below to confirmatory tests describe
their pre-specified statistical role, not a licensed-provenance claim. The exact
authorization, accepted Yahoo gaps, and bound hashes are in
`config/provenance_amendment.json`.

## Return and portfolio convention

Stock log total returns are converted to simple returns before cross-sectional
aggregation:

\[
r_{i,t}=\exp(\ell_{i,t})-1,
\qquad
r_{g,t}=\frac{1}{n_{g,y}}\sum_{i\in g_y}r_{i,t}.
\]

The primary portfolio is rebalanced to equal stock weights every day. Membership
and group labels are fixed within each calendar year. Consequently,

\[
R_{p,t}=\sum_g\frac{n_{g,y}}{N_y}r_{g,t}
=\frac{1}{N_y}\sum_i r_{i,t}.
\]

Verified halt and terminal-cash positions remain in the relevant group with a zero
simple return until the next annual rebalance. Marginal models use
`log1p(group_simple_return)`; simulations are converted back with `expm1` before
portfolio aggregation. Realised loss is `L = -R_p`, so a larger positive number is
a larger loss.

The secondary group-balanced portfolios start each year with equal capital per
group and equal capital per stock within a group. Their stock weights drift until
the next annual rebalance.

## Clustering statistics

For each 2020–2025 annual rebalance, Spearman correlations use the left-closed,
right-open interval from three calendar years before the rebalance date up to,
but excluding, that date. Correlations are converted to
`sqrt((1-rho)/2)` distances. Average linkage uses the UPGMA/Lance-Williams
size-weighted update, which gives every original cross-cluster security pair
equal weight. Exact distance ties and final cluster IDs are resolved by
lexicographically ordered sorted member tuples. The annual cluster count equals
the number of non-empty GICS sectors.

The primary within-minus-between dependence gap gives equal weight to every
unordered security pair. A singleton group supplies no within pair and is not
entered as a zero; all of its cross-group pairs remain in the between mean. Each
annual pair must have at least 80% paired observations. Reports include within and
between pair counts and the number of groups contributing within pairs.

The descriptive group-balanced robustness statistic averages non-singleton group
within-means equally and averages the cross-correlation mean of every unordered
group pair equally. ARI is calculated on the intersection of consecutive annual
active sets. NMI uses the active set for that year and mutual information divided
by the arithmetic mean of the two partition entropies.

The main security-level analysis retains GOOG and GOOGL. The issuer robustness
analysis replaces them with one position whose simple return is their 50/50 mean;
`log1p` is applied only after that averaging.

## Forecasts and scores

Every model supplies 95%, 97.5%, and 99% loss VaR and 97.5% loss ES. At confidence
`c`, VaR is the upper `c` quantile and an exception is `L > VaR`. Quantile loss is

\[
Q_c(L,q)=(c-\mathbf{1}\{L<q\})(L-q).
\]

The joint score uses FZ0 at `c=0.975`, tail probability `p=0.025`:

\[
S_{FZ0}(L,q,e)=
\frac{\mathbf{1}\{L>q\}(L-q)}{p e}
+\frac{q}{e}+\log(e)-1,
\]

with `e >= q` and `e > 0`.

Historical simulation uses the interval from three calendar years before the
forecast date up to, but excluding, the forecast date, with at least 700 valid
observations. VaR is empirical order statistic `ceil(n*c)`. ES integrates the
empirical quantile function over the upper tail, using fractional weight on the
boundary order statistic when `n*(1-c)` is non-integral.

## Margins, copulas, and simulation

The first marginal attempt is AR(1)-GARCH(1,1) with Student-t innovations. An
insignificant AR p-value is not a reason to change the model. A fit is rejected for
non-finite output, `abs(phi) >= 0.98`, invalid nonnegative variance parameters,
`alpha + beta >= 0.999`, Student-t degrees of freedom `<= 2.1`, or nonpositive
forecast variance. The deterministic fallback sequence is another AR-GARCH-t
attempt, constant-mean GARCH-t, then EWMA (`lambda=0.94`) with empirical
standardised innovations. An EWMA share over 1% of group-month fits fails the
modelling-quality gate.

For numerical estimation, group log returns are multiplied by 100 and converted
back to decimal units in stored daily forecasts. The initial optimizer receives
at most 1,000 iterations. The deterministic AR-GARCH-t retry receives at most
3,000 iterations and starts from an OLS AR(1) estimate clipped to `[-0.95,0.95]`,
`alpha=0.05`, `beta=0.90`, Student-t degrees of freedom 8, and
`omega=0.05` times the residual variance. The constant-mean GARCH-t fallback uses
the analogous sample-mean start. The frozen `alpha + beta < 0.999` admissible
region is imposed as an optimizer constraint on every GARCH attempt and is
checked again on the returned parameters and first forecast variance. Parameters
are fixed within each month while
the conditional mean and variance states update daily from returns observed up
to the preceding date.

The EWMA fallback uses the training-sample mean, variance with `ddof=1`, and
`lambda=0.94`. Its innovation distribution is the frozen training-sample
empirical distribution; PITs use a midrank empirical CDF and are clipped to
`[1e-6,1-1e-6]`. Student-t PITs use the standardized, unit-variance Student-t
distribution fitted by the marginal model and the same clipping bounds.

For each monthly copula refit, in-sample standardized residuals are transformed
with that group's selected Student-t or empirical marginal CDF. The copula
training panel retains only dates present for all 11 groups in the matched annual
representation. Every retained date is strictly before the refit date; the first
training observation may be absent because an AR(1) fit requires one lag. These
aligned training PITs are distinct from the daily out-of-sample PITs used for
forecast diagnostics.

The primary 11-dimensional R-vine is truncated after tree 3; a full ten-tree vine
is a robustness model. Structure selection uses the Dißmann sequential
maximum-spanning-tree procedure on absolute empirical Kendall tau. Pair families
are selected by AIC after sequential MLE from independence, Gaussian, Student-t,
Frank, and all relevant rotations of Clayton and Gumbel. PITs are clipped to
`[1e-6, 1-1e-6]`, and copula-t degrees of freedom are constrained to `[2.1, 50]`.

A failed pair becomes independence and is recorded. If over 10% of a month's pairs
fail, or structure fitting fails, the matched Gaussian copula supplies that month.
If whole-vine fallback covers over 1% of out-of-sample forecast dates, the vine
cannot be declared the best model.

The matched Gaussian copula is estimated by applying the standard-normal inverse
CDF to the aligned training PITs and taking their sample Pearson correlation.
The matrix is symmetrised before use. If its minimum eigenvalue is below `1e-8`,
all eigenvalues are floored at `1e-8`, the matrix is reconstructed, and its
diagonal is rescaled to one. Adjustments larger than `1e-12` are flagged, the
raw and repaired minimum eigenvalues and maximum elementwise adjustment are
retained, and a repaired condition number above `1e10` fails the quality gate.
Simulation uses a Cholesky factor and the Gaussian copula density supplies the
daily log score.

Each monthly fit uses 100,000 simulations generated by NumPy `PCG64DXSM` from
`SeedSequence([5110, year, month])`. The same independent-uniform matrix is passed
to all four copula specifications as common random numbers. Cluster definitions are
refitted on the first trading day of each year using information through the prior
year-end. Margins and copulas are refitted on the first trading day of each month
using information through the preceding trading date.

Student-t innovations are standardized to unit variance by multiplying ordinary
t quantiles by `sqrt((nu-2)/nu)`. For an EWMA fallback, simulation inverts the
complete frozen training innovation sample using the inverted empirical CDF.

Copula log scores are comparable only between Gaussian and vine models using the
same grouping.

## Inference and interpretation

Confirmatory Diebold-Mariano loss differences are `vine - Gaussian`. Tests are
two-sided with a direction check and Newey-West/Bartlett HAC lag 7. Holm correction
at 5% is applied across the six matched tests: two grouping comparisons times the
95% VaR, 99% VaR, and 97.5% FZ0 losses. Full support for H2 requires all six
adjusted results to favour the vine; other patterns are explicitly partial or no
support.

H1 uses 10,000 paired circular moving-block replications with block length 20
trading days. Entire daily cross-sectional vectors are resampled, and blocks wrap
within but never cross evaluation years.

A separate Holm family contains 20 full-period calibration tests: five models,
two primary VaR levels, and the Kupiec coverage and Christoffersen independence
tests. Annual 99% counts and tests are descriptive only. Forecast loss and paired
DM inference have greater evidential weight than annual coverage p-values.

The later spectral-clustering and PCA-plus-k-means comparisons are exploratory and
use Benjamini-Hochberg FDR at 5%.
