# 0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing

<!-- @document-metadata
  @title: 0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing
  @type: academic-paper
  @core-question: We study realized payoffs of S&P500 zero-days-to-expiration (0DTE) options and standard multi-leg structures from...
  @core-answer: Unconditional 0DTE is weak after costs; conditional OOS rules for selected strategies deliver SR 1.0-1.3
  @keywords: 0DTE, tactical overlay, tail risk, out-of-sample timing, option implementation, volatility trading
  @datasets: Cboe 30-min SPXW bars (2016-2026), ThetaData 1-min SPX/VIX bars
  @key-equations: (eq:is)-(eq:vrp)
  @key-tables: 15 tables (tab:0dte_strat_ret, tab:0dte_stratret2022_2023, tab:0dte_stratret2024_2026, tab:cond_feature_dictionary, tab:cond_model_zoo...)
  @key-figures: 13 figures (fig:cond_oos_investment_ts, fig:optprices, fig:optret, fig:optret-1300, fig:optret-1500...)
-->

<!-- @section-type: introduction
  @key-claim: Trading in zero-days-to-expiration options has become one of the most visible recent developments in the U.S. index-option market. For investors, derivatives desks, overlay managers, and risk...
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

## Introduction

Trading in zero-days-to-expiration options has become one of the most visible recent developments in
the U.S. index-option market. For investors, derivatives desks, overlay managers, and risk officers,
the practical question is no longer whether 0DTE activity is large, but whether it is usable. Do
same-day SPX option positions deliver a repeatable net edge after realistic execution costs, or do
they mainly load portfolios with concentrated intraday tail risk? And if payoffs vary materially
with market state, can that variation be turned into disciplined real-time decisions rather than
retrospective stories? Those are the questions that matter for portfolio use, mandate design, and
risk governance.

We study these questions using SPXW 0DTE calls, puts, and standard multi-leg structures from 09/2016
to 01/2026. The main text focuses on positions opened at 10:00 ET and held to the 16:00 ET close,
which matches a simple real-time implementation problem: choose a structure at a fixed decision
time, trade at observed quotes, and carry the position to settlement. The Appendix reports analogous
evidence for entries at 16:00 ET on the previous day and at 13:00 ET and 15:00 ET on the same day.
At each entry time, we interpolate the observed 0DTE cross-section over moneyness from 0.98 to 1.02
in steps of 0.001 and use these standardized instruments to construct comparable option- and
strategy-level payoffs. The strategy set covers straddles and strangles, iron butterflies and iron
condors, bull call and bear put spreads, call and put ratio spreads, and risk reversals. The goal is
to evaluate these structures the way a practitioner would: on a common scale, net of implementable
frictions, and with explicit attention to downside capital usage rather than average PNL alone.

Three findings stand out. First, a positive 0DTE variance risk premium exists, but at same-day
horizons its economic magnitude is small. Second, strategy-level payoff distributions are wide,
tail-heavy, and unstable across regimes; for many structures, downside risk is large relative to
average carry. Third, conditional timing works better when formulated as directional classification
than when posed as direct return prediction. In the conditional out-of-sample (OOS) implementation,
put ratio spreads reach a net Sharpe ratio of 1.26, iron butterfly structures reach 0.82, and
diversified equal-weight baskets reach net Sharpe ratios between 1.01 and 1.27.

For practitioners, these findings imply a fairly narrow use case. Unconditional 0DTE exposure is
difficult to justify as a standing allocation once one accounts for realistic execution and downside
capital usage. The more credible use is as a tactical sleeve: positions should be sized against
tail-risk measures rather than mean PNL alone, signals should rely on transparent real-time
variables rather than complex in-sample fit, and diversification across selected strategy families
is more defensible than reliance on a single structure.

We contribute to several adjacent strands of research. Relative to the emerging 0DTE market-impact
literature, including the merged draft of (AdamsDimErakerFontaineOrnthanalaiVilkov2025)[^1] and the
related evidence in (BrogaardHanWon2026, AmayaGarciaAresPearsonVasquez2025), our object is not the
causal effect of dealer hedging on aggregate volatility, but the realized payoff distribution of
standardized strategy templates under implementable construction rules. Relative to the ultra-short
option-pricing literature (e.g., (Bandi23, AlmeidaFreireHizmeri2025)), we take a reduced-form route
and document how feasible 0DTE strategy PNL maps into implied and realized moments. Relative to the
broader short-dated option-return literature (e.g., (Beckmeyer23, bryzgalova2022,
BogousslavskyMuravyev2025)), our evidence supports the view that gross option stories weaken
materially once one accounts for execution costs and tail risk. Finally, relative to the option-
return and event-state literature (e.g., (BuechnerKelly2022, LondonoSamadi2025,
KnoxLondonoSamadiVissingJorgensen2025)), we study unhedged same-day strategy payoffs, restrict
predictors to information available by 10:00 ET, and impose a strict out-of-sample protocol for
conditional rules.

The rest of the paper proceeds as follows. Section [sect:data-prep] describes the data and the
construction of the option-, strategy-, and state variables used throughout. Section [sect:uncond]
documents unconditional payoffs of individual options and strategy templates. Section [sect:drivers]
links strategy PNL to implied and realized moments and tightens inference for these links. Section
[sect:cond-rules] studies conditional trading rules. Section [sect:conclusion] concludes. Appendix
[sect:app-extratabs-figs] contains supporting tables and figures.

[^1]: The 2025 merged draft consolidates and extends two predecessor papers,
(AdamsFontaineOrnthanalai2025) and (DimErakerVilkov2024SSRN4692190).

<!-- @section-type: data
  @key-claim: 
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

## Data, Implementation, and Variable Construction


<!-- @section-type: data
  @key-claim: This section describes the data sources and the construction choices used throughout the paper. We use Cboe 30-minute option bars with NBBO quotes and sizes, OHLC prices, trade volume, and underlying...
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

#### Data on Options and Underlying Markets.

This section describes the data sources and the construction choices used throughout the paper. We
use Cboe 30-minute option bars with NBBO quotes and sizes, OHLC prices, trade volume, and underlying
levels, focusing on SPX Weeklys (root `SPXW`). These contracts are European and cash settled against
the SPX close at 16:00 ET. The sample runs from 09/2016 to 01/2026. SPXW had three weekly
expirations from late August 2016; two additional weekdays were introduced on April 18, 2022 and May
11, 2022, giving daily weekday 0DTE expirations.  For underlying prices and realized-moment
construction, we use *ThetaData* one-minute bars (SPX and VIX where needed) and ThetaData daily
closes.

<!-- @section-type: drivers
  @key-claim: We compute implied variance ($IV$) to expiration at each intraday bar-end $t$ on expiration day using the VIX methodology (CboeVIXMeth2023) applied to SPXW 0DTE options. The conditioning variable at...
  @importance: core
  @data-source: none
  @equations: eq:iv, eq:is, eq:rv_dte0, eq:rs, eq:vrp, eq:srp
  @tables: none
  @figures: none
-->

#### Implied and Realized Moment Measures.

We compute implied variance ($IV$) to expiration at each intraday bar-end $t$ on expiration day
using the VIX methodology (CboeVIXMeth2023) applied to SPXW 0DTE options. The conditioning variable
at 10:00 ET is $IV_{10:00}$, i.e., integrated implied variance from 10:00 ET to 16:00 ET:

$$
IV_{t} = 2 e^{rT} \sum_i \frac{\Delta K_i}{K_i^2} Q(K_i) - [F/K_0-1]^2,
$$

where $K_i$ is strike, $K_0$ is the first strike at or immediately below option-implied forward $F$,
$Q(K_i)$ is the OTM mid quote ($i\neq0$), $Q(K_0)$ is the average of call and put quotes at $K_0$,
$r$ is the risk-free rate (1-month Treasury bill from FRED), and $T$ is time to expiration in years.
We also define up and down semivariances, $IV^{up}_t$ and $IV^{dn}_t$, by applying (eq:iv) to weakly
OTM calls and puts, respectively, and scaling the subtracted term to $0.5 [F/K_0-1]^2$ so that the
two components sum to total $IV_t$. The difference between up and down semivariance is a common
implied-skewness proxy (e.g., (FeunouJahanParvarOkou2017, KilicShaliastovich2017)), and we use: [^1]

$$
IS_t = IV^{up}_t - IV^{dn}_t.
$$

We compute forward-looking realized variance ($RV$) at the end of each bar $t$ for periods matching
each computed $IV$ as the sum of squared one-minute log returns from the end of a bar to the end of
the day:

$$
\tag{eq:rv_dte0}
RV_{t} = \sum_t^{T-1} r_{t,t+1}^2,
$$

where $r_{t,t+1}$ is the one-minute close-to-close log return from minute $t$ to $t+1$, summed
through $T{=}16{:}00$ on the same day. Analogous to implied semivariances, realized semivariances
$RV^{up}_t$ and $RV^{dn}_t$ are computed by multiplying each $r_{t,t+1}^2$ term in (eq:rv_dte0) by
$\mathbf{1}_{r_{t,t+1}>0}$ or $\mathbf{1}_{r_{t,t+1}<0}$, respectively. We define realized skewness
proxy as

$$
RS_t = RV^{up}_t - RV^{dn}_t.
$$

We define *ex post* variance risk premium $VRP$ at bar $t$ as the payoff on a short variance swap
from $t$ to expiration, i.e., implied minus realized variance over that window:

$$
VRP_{t} = IV_{t} - RV_{t}.
$$

Semivariance premia $VRP^{up}_{t}$ and $VRP^{dn}_{t}$ are defined analogously.[^2]  Skewness premium
is then defined in two equivalent ways:

$$
SRP_{t} = IS_t - RS_t = VRP^{up}_{t} - VRP^{dn}_{t}.
$$

For options expiring the same day, observed at an intraday time (for example 10:00 ET), we compute
moneyness as strike divided by spot and retain options in $[0.98,1.02]$ (within $\pm2%$ of ATM). We
scale mid prices and bid-ask spreads by spot, then interpolate key variables (mid, spread, implied
volatility, Greeks, and leg-level flow/liquidity inputs used later) over moneyness with step 0.001,
separately for calls and puts.[^3]  For each interpolated option, we compute the payoff:

$$
Payoff_t (M, Type) =

(S_T/S_t-M)^{+} if Type=Call, (M-S_T/S_t)^{+} if Type=Put,

$$

where $M$ is moneyness and $S_T/S_t$ is gross index return from $t$ to $T{=}16{:}00$.  For each
option we also compute intrinsic value (immediate-exercise payoff) and time value (mid minus
intrinsic). If time value is negative because of mid-quote noise, we set time value to zero and
intrinsic value to mid. Since moneyness is strike scaled by spot, payoff, intrinsic, and time value
are already in spot-relative units.  Option PNL relative to spot is:

$$
PNL_t(M, Type) = Payoff_t (M, Type) - Mid_t(M,Type),
$$

where $Mid_t(M,Type)$ is the mid price at time $t$ for option $(M,Type)$. Realized return to
expiration is:

$$
Ret_t(M, Type) = \frac{PNL_t(M, Type)}{Mid_t(M,Type)} = \frac{Payoff_t (M, Type)}{Mid_t(M,Type)} - 1.
$$

We report both PNL and return in percentage terms (multiply by 100).  For multi-leg structures,
strategy PNL is the signed sum of leg-level PNL weighted by leg quantities; by construction it
remains in spot-relative units.

[^1]: Implied skewness can also be computed directly (e.g., (ba/ka/ma/03),
(KozhanNeubergerSchneider2013)). We use the semivariance-difference proxy because it is simple under
both risk-neutral and physical measures and maps naturally to skewness-premium definitions.

[^2]: We do not annualize these quantities; the goal is to keep magnitudes in realized same-day
payoff units.

[^3]: At SPX$=4000$, step 0.001 corresponds to roughly 4 index points. Interpolation standardizes
moneyness across dates and enables direct strategy construction. We use one-dimensional piecewise
cubic interpolation (Akima1970), with similar results under alternative interpolation schemes.
Interpolation is cross-sectional in moneyness at a fixed date/time and option type; we do not
interpolate across time.

<!-- @section-type: methodology
  @key-claim: For the conditional model section, we use the same SPXW option pipeline and then construct additional predictors at the strategy level. In preprocessing, we keep same-day-expiry SPXW quotes for...
  @importance: supporting
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

#### Construction of Conditional Features at 10:00 ET.

For the conditional model section, we use the same SPXW option pipeline and then construct
additional predictors at the strategy level. In preprocessing, we keep same-day-expiry SPXW quotes
for intraday bars and include next-day-expiry contracts only at 16:00 ET for close-to-close
alignment in intermediate steps; the conditional models themselves use the 10:00 ET cross-section of
same-day-expiry strategy observations only. Calls and puts are processed separately: interpolation
is run within option type on the moneyness grid $M \in \{0.98,0.981,\dots,1.02\}$, and strategy legs
are matched by $(datetime,option\_type,moneyness)$. In short, option-leg predictors are taken from
this interpolated 10:00 ET grid, while market controls and lagged realized variables are taken from
non-interpolated underlying time series.

Let $i=(s,m,t)$ index strategy type $s$, moneyness configuration $m$, and date-time $t$ (here,
$t=10{:}00$ ET), and let $\mathcal{L}_{s,m}$ denote legs in that structure, with signed leg quantity
$q_l$ ($q_l>0$ long, $q_l<0$ short). For each leg $l$, we collect the option inputs $bas_l$,
$mid_l$, $\Delta_l$, $\Gamma_l$, $\nu_l$, $Vol_l$, $OI_l$, $BidSz_l$, $AskSz_l$, and the active
underlying level $S_l$. If multiple rows map to the same $(datetime,option\_type,moneyness)$ key, we
aggregate by variable type before leg construction: price and Greek variables
$(bas,mid,\Delta,\Gamma,\nu,S,OI)$ are averaged, while traded and displayed depth variables
$(Vol,BidSz,AskSz)$ are summed. For compactness, we denote half-spread by $h$, displayed depth by
$d$, relative spread by $\rho$, traded volume by $v$, flow notional by $f$, and gamma exposure by
$g$.

We then define leg-level flow, gamma-exposure, and liquidity building blocks. In our notation,
$(h,d,\rho)$ are *liquidity* features. They proxy execution cost, displayed depth, and relative
tightness of quoted markets at the entry time:

$$
h_{i,l} = |q_l| \cdot \frac{bas_{l}}{2},
d_{i,l} = |q_l| \cdot (BidSz_l + AskSz_l),
\rho_{i,l} = |q_l| \cdot \frac{bas_l}{|mid_l|}.
$$

The variables $(v,f^\Delta,f^\Gamma,f^\nu)$ are *flow* features: they measure how much contract
volume, delta notional, gamma notional, and vega notional trade through the strikes used by the
strategy:

$$
v_{i,l} = |q_l| \cdot Vol_l,
f^\Delta_{i,l} = |q_l| \cdot Vol_l \cdot |\Delta_l| \cdot 100 \cdot S_l,
f^\Gamma_{i,l} = |q_l| \cdot Vol_l \cdot |\Gamma_l| \cdot 100 \cdot S_l^2,
f^\nu_{i,l} = |q_l| \cdot Vol_l \cdot |\nu_l| \cdot 100.
$$

The variables $(g^{OI,n},g^{OI,a},g^{\Gamma,n},g^{\Gamma,a})$ are *GEX-style* features. They
summarize signed and absolute convexity exposure, either weighted by open interest or coming
directly from the strategy-leg mix:

$$
g^{OI,n}_{i,l} = q_l \cdot OI_l \cdot \Gamma_l \cdot 100 \cdot S_l^2,
g^{OI,a}_{i,l} = |q_l| \cdot OI_l \cdot |\Gamma_l| \cdot 100 \cdot S_l^2,
g^{\Gamma,n}_{i,l} = q_l \cdot \Gamma_l,
g^{\Gamma,a}_{i,l} = |q_l| \cdot |\Gamma_l|.
$$

We derive these three families separately because they capture distinct economic channels:
implementation frictions, current trading pressure, and concentration of convexity.

The leg-level liquidity, flow, and GEX features defined above are aggregated to the strategy level
by summing over all legs in the strategy at 10:00 ET. In the current conditional models we use same-
day-expiry contracts only at 10:00 ET; next-day-expiry contracts are used only for 16:00 ET
alignment in preprocessing. We additionally use transformed ratios:

$$
B^\Gamma_i = \frac{g^{OI,n}_i}{|g^{OI,a}_i|+1},
R^\Gamma_i = \frac{f^\Gamma_i}{|g^{OI,a}_i|+1},
T_i = \frac{h_i}{d_i+1}.
$$

These transformed indicators remain in the same three families: $B^\Gamma$ is a normalized *GEX
balance* measure, $R^\Gamma$ is a *flow-pressure* measure scaled by the local GEX base, and $T$ is a
*liquidity tightness* measure. We derive them because raw flow and raw GEX are strongly scale-
dependent across strategies, strike widths, and premium sizes. These are *flow-style* and exposure-
style proxies based on traded volume, open interest, and leg Greeks; they are not a dealer-inventory
reconstruction. If a strategy observation has incomplete leg coverage at 10:00 ET, we set the
engineered leg-aggregated features to missing. For specifications that use date-wise cross-sectional
scaling, each strategy-level predictor $X_{i,t}$ is transformed as

$$
X^{CS}_{i,t}=\frac{X_{i,t}-\overline{X}_t}{std_t(X)}, \quad
\overline{X}_t=\frac{1}{N_t}\sum_{i=1}^{N_t}X_{i,t},
$$

where $N_t$ is the number of strategy observations at date $t$.

Baseline strategy-level predictors include time value, mid-price, $\Delta$, $\Gamma$, $\nu$,
moneyness-center, and lagged PNL terms. For a structure with leg moneyness values
$\{M_l\}_{l\in\mathcal{L}_{s,m}}$, we define

$$
M_{i,t} = \frac{1}{|\mathcal{L}_{s,m}|}\sum_{l\in\mathcal{L}_{s,m}} M_l.
$$

Lagged PNL features are

$$
P^{(1)}_{i,t} = PNL_{i,t-1}, \quad
\bar P^{(5)}_{i,t} = \frac{1}{5}\sum_{j=1}^5 PNL_{i,t-j}, \quad
\sigma^{(5)}_{P,i,t} = sd(PNL_{i,t-1},\dots,PNL_{i,t-5}),
$$

computed within each $(strategy,moneyness)$ series. We also include strategy fixed identifiers as
one-hot dummies $D_{s,i}\in\{0,1\}$. Market-state controls at 10:00 ET are

$$
IV_t = 10^5 \cdot \overline{VIX}_{t}^{(SPXW,0DTE,10{:}00)},
IS_t = 10^5 \cdot \left(\overline{VIX^{up}}_{t}^{(SPXW,0DTE,10{:}00)} - \overline{VIX^{dn}}_{t}^{(SPXW,0DTE,10{:}00)}\right),
$$

plus slope proxies $(slope\_up_t,slope\_dn_t)$ and one-day-lagged realized SPX moments, namely
lagged realized returns, variance, and skewness, where the lag enforces strict no-look-ahead
construction. These variables form the remaining two families. The *baseline* family consists of the
strategy moneyness center $M$, yesterday's strategy PNL $P^{(1)}$, the five-day average strategy PNL
$\bar P^{(5)}$, the five-day standard deviation of strategy PNL $\sigma^{(5)}_P$, and the option-
level descriptors used elsewhere in the models (time value, mid price, $\Delta$, $\Gamma$, and
$\nu$); together they capture strategy geometry, local carry, and short-horizon persistence. The
*market-state* family consists of $(IV,IS)$, slope proxies, and lagged realized SPX measures; they
capture the entry-time variance/skew environment and recent underlying conditions observed by 10:00
ET. We derive these families to separate strategy-specific signals from market-wide states. The
continuous target in the model zoo is the strategy return-per-underlying $y_{i,t}=reth\_und_{i,t}$,
and the directional target is $\mathbbm{1}(y_{i,t}>0)$. A full predictor dictionary for the compact
notation used here is reported in Appendix Table [tab:cond_feature_dictionary].

<!-- @section-type: results
  @key-claim: An unconditional strategy opens option positions each day without conditioning on state variables that would change timing or size. Positions are then held to expiration, and performance is measured...
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

## Unconditional 0DTE Results

An unconditional strategy opens option positions each day without conditioning on state variables
that would change timing or size. Positions are then held to expiration, and performance is measured
by holding-period return and PNL.  The simplest positions use a single option; richer structures
combine multiple legs to target specific payoff and Greeks profiles.

We compute return and PNL to expiration for all individual options and strategy combinations and
summarize their distributions.

<!-- @section-type: results
  @key-claim: Over the full sample (09/2016 to 01/2026), 0DTE options deliver a positive and statistically significant variance risk premium: implied variance exceeds realized variance to expiration (Figure...
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: fig:vrp, fig:optprices
-->

### Variance Risk Premium Is Not Enough

Over the full sample (09/2016 to 01/2026), 0DTE options deliver a positive and statistically
significant variance risk premium: implied variance exceeds realized variance to expiration (Figure
[fig:vrp], Panel B).

> **Figure** (fig:vrp): **Variance Risk Premiums at 0DTE Horizons.** The figure shows average variance and its risk premiums (VRP) to expiration for 0DTE SPXW options by intraday 30-minute bars. VRP is computed as implied minus realized variances from a given bar to expiration at 16:00 ET. Panels C and D are based on average volatility and differences in volatilities, respectively. 
We average variables measured at the end of each bar to expiration at 16:00 that day (with 95% confidence bounds based on Newey-West standard errors with three lags; (newey/west/1987)). Bars show mean values; each bar is accompanied by median, 25th, and 75th percentiles. X-axis labels show the endpoints of intraday bars. The sample period is from 09/2016 to 01/2026.

> Panel A: Variance Swap Rate to Expiration [figures/ vrp_bytime_bars.pdf]
> Panel B: Variance Risk Premiums to Expiration [figures/ vrp_bytime_bars.pdf]
> Panel C: Volatility Swap Rate to Expiration [figures/ vrp_bytime_bars.pdf]
> Panel D: Volatility Risk Premiums to Expiration [figures/ vrp_bytime_bars.pdf]

Trading variance requires a strip of options across strikes, with weights determined by the chosen
model-free variance formula. Under the Cboe VIX construction, risk-neutral variance is a weighted
sum of OTM option prices (equation [eq:iv]). In practice, the 0DTE variance swap rate to expiration
is a weighted sum of OTM call and put time values (Figure [fig:vrp], Panel A). The realized variance
risk premium can be interpreted as time value retained after paying exercised options at expiration.
As realized variance increases, terminal index moves are larger on average and payouts rise; with
zero realized variance, the index is unchanged and the initial time value is retained.

Panel B of Figure [fig:vrp] suggests that selling 0DTE variance is profitable on average. The
economic magnitude, however, is small because OTM time values only hours before expiration are small
(Figure [fig:optprices]). Even in an extreme zero-realized-variance scenario, collecting roughly
twice the average OTM time value implies only about 0.20% of spot. Translating the variance object
into a spot-relative monetization benchmark, the median realized VRP from 10:00 ET to expiration is
approximately 0.0011% of underlying. These magnitudes are difficult to monetize after realistic
trading frictions.

> **Figure** (fig:optprices): **0DTE Option Prices and Time Value at 10:00 ET.** The figure provides statistics on prices of 0DTE call and put options at 10:00 ET. Panels on the left show option mid-price relative to the underlying price in %. Panels on the right show time value relative to underlying, also in %. Bars show mean values, and each bar is accompanied by median, 25th, and 75th percentiles. X-axis labels show the moneyness of the analyzed options. The sample period is from 09/2016 to 01/2026.

> Panel A: Call Price, 10:00 [figures/ calls_mid_bymnes_bars.pdf]
> Panel B: Call Time Value, 10:00 [figures/ calls_tv_bymnes_bars.pdf]
> Panel C: Put Price, 10:00 [figures/ puts_mid_bymnes_bars.pdf]
> Panel D: Put Time Value, 10:00 [figures/ puts_tv_bymnes_bars.pdf]

<!-- @section-type: results
  @key-claim: For individual calls and puts, we report both return relative to mid price and realized PNL relative to spot at entry, $(payoff-mid price)/underlying price \times 100%$, using 10:00 ET entries in the...
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: fig:optret
-->

### Individual Options

For individual calls and puts, we report both return relative to mid price and realized PNL relative
to spot at entry, $(payoff-mid price)/underlying price \times 100%$, using 10:00 ET entries in the
main text (Figure [fig:optret]).          Individual option returns are highly volatile, especially
for OTM contracts, and mean returns are often not statistically distinct from zero. Because return
distributions are strongly skewed, inference on means alone is not very informative. PNL scaled by
spot is less asymmetric and shows that ATM-to-OTM calls and most puts lose on average, while OTM
contracts have relatively tighter distributions. Selling slightly OTM calls and puts is positive in
up to 75% of observations; deeper ITM contracts are much more volatile, with interquartile ranges
around 0.7--0.8% of spot.

> **Figure** (fig:optret): **Individual 0DTE Option Returns and PNL.** The figure provides statistics on the profitability of naked 0DTE call and put option buying at 10:00 ET and holding to expiry at 16:00 ET. Panels on the left show realized return in % relative to option mid-price. Panels on the right show the realized profit per one unit of underlying relative to underlying price $(payoff - mid price)/underlying price \times 100%$. Bars show mean values, and each bar is accompanied by median, 25th, and 75th percentiles. X-axis labels show the moneyness of the analyzed options. 
The sample period is from 09/2016 to 01/2026.

> Panel A: Call Returns [figures/ calls_reth_bymnes_bars.pdf]
> Panel B: Call PNL/Underlying [figures/ calls_reth_und_bymnes_bars.pdf]
> Panel C: Put Returns [figures/ puts_reth_bymnes_bars.pdf]
> Panel D: Put PNL/Underlying [figures/ puts_reth_und_bymnes_bars.pdf]

Option performance for entries at 16:00 ET on the previous day, 13:00 ET, and 15:00 ET is shown in
Figures [fig:optret-1600prev], [fig:optret-1300], and [fig:optret-1500]. The main qualitative
findings are unchanged across entry times.

<!-- @section-type: results
  @key-claim: Single-option positions load on both volatility and direction, while multi-leg structures allow tighter exposure design. We analyze: (i) ATM straddles and strangles, which are close to delta-neutral...
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:0dte_strat_ret
  @figures: fig:strat-ret
-->

### Option Strategies

Single-option positions load on both volatility and direction, while multi-leg structures allow
tighter exposure design. We analyze: (i) ATM straddles and strangles, which are close to delta-
neutral at initiation and primarily volatility-focused; (ii) iron butterflies and iron condors,
which are short-volatility with bounded downside; (iii) risk reversals, which load on skewness
premia; (iv) bull call and bear put spreads, which are directional structures; and (v) call and put
ratio spreads ($1\times2$), which target moderate directional moves with asymmetric tails.

These structures have distinct payoff profiles (Figure [fig:strat-example]). Straddles/strangles
have limited downside and open upside; iron butterfly/condor have bounded upside and downside;
bull/bear spreads are directional with bounded payoffs; ratio spreads have capped upside and open
downside; risk reversals are open-tailed on both sides. Entry cost reflects these shapes: narrow-
strike risk reversals and ratio spreads are often credit structures, while straddles and strangles
are typically the most expensive debit structures.

> **Figure** (fig:strat-ret): **Unconditional 0DTE Strategy PNL at 10:00 ET.** The figure shows statistics on the profitability of 0DTE option strategies from 10:00 ET to expiry at 16:00 ET. All panels show strategies' realized PNL relative to underlying price $(payoff - mid price)/underlying price \times 100%$. Bars show mean values, accompanied by the median and the 25th and 75th percentiles. X-axis labels show the combination of moneyness of options used for a strategy. The sample period is from 09/2016 to 01/2026.

> Panel A: Straddle/Strangle [figures/ strangle_reth_und_bymnes_bars.pdf]
> Panel B: Iron Butterfly/Condor [figures/ iron_condor_reth_und_bymnes_bars.pdf]
> Panel C: Risk Reversal [figures/ risk_reversal_reth_und_bymnes_bars.pdf]
> Panel D: Bull Call Spread [figures/ bull_call_spread_reth_und_bymnes_bars.pdf]
> Panel E: Bear Put Spread [figures/ bear_put_spread_reth_und_bymnes_bars.pdf]
> Panel F: Call Ratio Spread [figures/ call_ratio_spread_reth_und_bymnes_bars.pdf]
> Panel G: Put Ratio Spread [figures/ put_ratio_spread_reth_und_bymnes_bars.pdf]

For some combinations, the initial net premium is close to zero or negative, so the return scaled by
the initial premium is unstable or not economically meaningful. We therefore focus on realized PNL
scaled by spot.  Figure [fig:strat-ret] and Table [tab:0dte_strat_ret] show broad dispersion for
most structures. To keep the table readable, we report only near-ATM combinations there, while
Figure [fig:strat-ret] continues to show the full grid. Spread and ratio-spread outcomes are highly
variable, with interquartile ranges often roughly symmetric around zero. For ATM-long directional
spreads, median PNL is typically negative. Using ITM long legs yields more symmetric distributions
with means close to zero and small sign differences between call and put structures, consistent with
the upward index drift in much of 2017--2023. OTM strangles are relatively compact and mostly
negative in the upper and lower quartiles, so short strangles outperform in at least 75% of
observations. Risk reversals are the only structure with consistently positive mean, median, and
25th percentile PNL, but the average effect is small (about 0.01% of spot), which limits
implementability.

> **Table** (tab:0dte_strat_ret): **Unconditional 0DTE Strategy PNL at 10:00 ET.** The table shows the summary statistics for the holding period PNL of 0DTE option strategies (from 10:00 ET to 16:00 ET) relative to underlying price $(payoff - mid price)/underlying price \times 100%$. To keep the table concise, we report only near-ATM combinations with all legs within 1% of ATM; Figure [fig:strat-ret] shows the full grid. The SR, p.a. is the Sharpe Ratio annualized by scaling it up by $\sqrt{252}$. The sample period is from 09/2016 to 01/2026.
> Source: `tables/ 0dte_stratret2000_2100_main.tex`

Performance for the same strategy set at alternative entry times (16:00 ET previous day, 13:00 ET,
and 15:00 ET) is shown in Figures [fig:strat-ret-1600prev], [fig:strat-ret-1300], and [fig:strat-
ret-1500]. Full-sample unconditional patterns are broadly similar across entry times.  Subperiod
tables for 2022--2023 and 01/2024--01/2026 (Tables [tab:0dte_stratret2022_2023] and
[tab:0dte_stratret2024_2026]) show substantial time variation. This motivates conditional rules that
explicitly tie strategy exposure to observable market states.

<!-- @section-type: robustness
  @key-claim: To formalize the regime-shift discussion, we test whether average strategy PNL at 10:00 ET changed after the expansion to daily SPXW expirations in 2022. We define the pre-period as dates up to April...
  @importance: supporting
  @data-source: none
  @equations: none
  @tables: tab:structbreak_2022
  @figures: none
-->

### Regime Stability Around the 2022 Expansion of Daily Expirations

To formalize the regime-shift discussion, we test whether average strategy PNL at 10:00 ET changed
after the expansion to daily SPXW expirations in 2022. We define the pre-period as dates up to April
14, 2022, the post-period as dates from May 11, 2022 onward, and exclude the transition window
between April 18 and May 10, 2022. For each strategy, we estimate a pooled model with combo fixed
effects and a post-2022 indicator, clustering standard errors by date. The post-2022 coefficient is
the structural-break estimate.

> **Table** (tab:structbreak_2022): **Regime Stability Around the 2022 Expansion of Daily Expirations.** The table reports pre/post means of strategy PNL relative to underlying (in percentage points), volatility ratios, and the post-period coefficient from regressions with combo fixed effects and date-clustered standard errors. The pre-period ends on April 14, 2022; the post-period starts on May 11, 2022; observations in between are excluded as transition.
> Source: `tables/ 0dte_structbreak_post2022.tex`

Table [tab:structbreak_2022] shows economically non-trivial shifts in several strategies' average
PNL signs between the two regimes (for example, bull call and call ratio spreads improve, while bear
put and put ratio spreads deteriorate), but these shifts are statistically weak once we account for
strong within-day cross-combo dependence and heavy-tailed realizations. This evidence supports
treating unconditional 0DTE strategy performance as regime-sensitive and motivates conditional rules
rather than relying on full-sample means.

<!-- @section-type: results
  @key-claim: The baseline results above are based on mid-quote marking. To evaluate implementability, we add execution friction at entry in three layers: (i) mid benchmark, (ii) bid/ask execution, where each...
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:implementable_pnl
  @figures: none
-->

### Implementability: Execution Costs, Turnover, and Capital Usage

The baseline results above are based on mid-quote marking. To evaluate implementability, we add
execution friction at entry in three layers: (i) mid benchmark, (ii) bid/ask execution, where each
strategy leg pays half of the observed option bid-ask spread, and (iii) bid/ask plus an additional
0.5 bp slippage-and-fee charge (in underlying-relative PNL units). We report strategy-level daily
series at 10:00 ET, aggregating equally across moneyness combinations within each strategy.

> **Table** (tab:implementable_pnl): **Implementable Net PNL and Capital Usage at 10:00 ET.** Means are in % of underlying. ``B/A'' subtracts half-spread execution cost computed from option-level bid-ask spreads and strategy legs. ``B/A+0.5bp'' further subtracts 0.5 basis points. Turnover is a gross entry premium proxy (in basis points of underlying), computed as absolute strategy premium plus half-spread cost. ES$_{1%}$ is a 1% expected-shortfall proxy on daily strategy PNL under B/A+0.5bp.
> Source: `tables/ 0dte_implementable_pnl.tex`

Table [tab:implementable_pnl] shows that moving from mid to implementable execution mechanically
lowers mean PNLs and annualized Sharpe ratios for most strategies. The turnover proxy is small for
all structures in underlying-relative terms, but capital usage measured by left-tail ES$_{1%}$
remains large relative to average daily PNL, implying low mean-over-tail-capital ratios for most
strategies. This reinforces that gross unconditional premiums can look materially better than
implementable net performance, especially once friction and downside capital usage are accounted
for.

<!-- @section-type: results
  @key-claim: Because strategy distributions are highly non-Gaussian, we explicitly report tail-focused risk diagnostics for implementable net PNL at 10:00 ET (half-spread plus 0.5bp slippage/fees).
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:tail_risk_diag
  @figures: fig:strat-ret-ts
-->

### Tail Risk and Capital at Risk

Because strategy distributions are highly non-Gaussian, we explicitly report tail-focused risk
diagnostics for implementable net PNL at 10:00 ET (half-spread plus 0.5bp slippage/fees).

> **Table** (tab:tail_risk_diag): **Tail Risk of Implementable 0DTE Strategy PNL.** Daily PNL is in % of underlying and is net of half-spread and 0.5bp costs. Max drawdown is computed strategy-by-strategy as peak-to-trough decline of the cumulative daily net-PNL series (not a one-day loss and not a cross-strategy sum). ES$_{1%}$, max drawdown, worst day, and worst 5-day outcomes emphasize downside exposure and distribution asymmetry.
> Source: `tables/ 0dte_tail_risk_diagnostics.tex`

Table [tab:tail_risk_diag] confirms that left-tail risk dominates mean effects for most strategies:
ES$_{1%}$ values are economically large (roughly 0.58--1.58% of underlying), worst-day outcomes are
severe, and cumulative drawdowns are substantial. Combined with mixed skewness signs across
strategies, this reinforces that mean PNL alone is an insufficient summary for 0DTE strategy
evaluation.

Taken together, these unconditional results are weak from an implementation perspective. Realized
PNL is highly volatile, and the average variance premium is economically small at 0DTE horizons.
Some favorable strategy outcomes may partly reflect the upward drift of the S&P 500 over our sample.
We intentionally do not winsorize extremes, because rare tail realizations are central to the risk
of these strategies. From a practical standpoint, these numbers argue against treating 0DTE as a
simple carry trade or income strategy. For allocators and risk managers, the relevant comparison is
not mean PNL in isolation but mean PNL relative to execution cost, expected shortfall, and drawdown
capacity. Any live implementation should therefore be justified as a risk-budgeted tactical exposure
rather than as a broad unconditional premium harvest.

> **Figure** (fig:strat-ret-ts): **Time Variation in 0DTE Strategy PNL.** The figure shows 63-trading day moving average PNL of option strategies (from 10:00 ET to expiry at 16:00 ET) relative to underlying price $(payoff - mid price)/underlying price \times 100%$. Secondary y-axis shows scaled to (0,1) series of SPX and VIX. The sample period is from 09/2016 to 01/2026.

> Panel A: Straddle/Strangle [figures/ ts_strangle_reth_und.pdf]
> Panel B: Iron Butterfly/Condor [figures/ ts_iron_condor_reth_und.pdf]
> Panel C: Risk Reversal [figures/ ts_risk_reversal_reth_und.pdf]
> Panel D: Bull Call Spread [figures/ ts_bull_call_spread_reth_und.pdf]
> Panel E: Bear Put Spread [figures/ ts_bear_put_spread_reth_und.pdf]
> Panel F: Call Ratio Spread [figures/ ts_call_ratio_spread_reth_und.pdf]
> Panel G: Put Ratio Spread [figures/ ts_put_ratio_spread_reth_und.pdf]

To illustrate time variation, Figure [fig:strat-ret-ts] plots 63-trading-day moving averages of
strategy PNL together with scaled SPX and VIX series. No strategy looks close to risk-free; profit
and loss regimes can persist, but it is unclear whether they are identifiable *ex ante*.  This
motivates a more systematic look at what these strategies are actually loading on before we ask
whether that variation can be forecast in real time.

<!-- @section-type: drivers
  @key-claim: Before turning to implementable conditional rules, we examine which ex-post shocks drive daily strategy PNL. The candidate channels are not symmetric across structures. Variance-focused positions...
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:strat_ret_expl
  @figures: none
-->

## What Drives Strategy PNL in Practice?

Before turning to implementable conditional rules, we examine which ex-post shocks drive daily
strategy PNL. The candidate channels are not symmetric across structures. Variance-focused positions
such as straddles and strangles should load more directly on realized variance, while directional
spreads and skew trades should react more strongly to directional asymmetry and skewness
realizations. We therefore relate strategy PNL at 10:00 ET to implied and realized moment measures,
combining moneyness configurations within each strategy and absorbing mechanical differences across
strike designs with combo fixed effects.

> **Table** (tab:strat_ret_expl): **What Drives Strategy PNL? Implied and Realized Moments.** The table shows the results of regressing realized PNL of option strategies (10:00 ET to expiry at 16:00 ET) on implied and realized distribution moments. PNL is specified per one unit of underlying relative to underlying price $(payoff - mid price)/underlying price \times 100%$. The result in each column is based on a pooled regression of strategy PNL for several moneyness combinations, including combo fixed effects (Combo FE) and date-clustered standard errors. The sample period is from 09/2016 to 01/2026.
> Source: `tables/ strat_PNL_rv_rvrs_ivrv_ivisrvrs.tex`

To quantify strategy sensitivities, we run strategy-specific pooled regressions of realized PNL on
implied and realized moments (and related premium terms), combining multiple moneyness
configurations within each strategy. Because changing strike configuration changes moment exposure,
we include combo fixed effects.[^1] Main results are reported in Table [tab:strat_ret_expl]; last-
hour analogs (15:00 ET to close) are in Appendix Table [tab:strat_ret_expl_1500].

Panel A of Table [tab:strat_ret_expl] shows that realized variance alone explains little of cross-
day strategy PNL. Even when $RV$ coefficients are statistically significant, $R^2$ values are low
(maximum about 4.5% for put ratio spreads). Directional spreads show opposite variance signs,
consistent with negative index-variance comovement: higher variance tends to coincide with stronger
down moves, helping bear structures and hurting bull structures. Adding realized skewness in Panel B
materially improves fit for asymmetric strategies such as risk reversals and directional spreads.
Ratio-spread fit also improves, though sensitivity becomes unstable when large directional moves
dominate.

Including implied variance and implied skewness proxies in Panels C and D adds further explanatory
power, typically by 2--7 percentage points of $R^2$. The increase is strongest for strangles,
directional spreads, and call ratio spreads. Overall, strategy PNL is driven more by realized
skewness (directional movement) than by realized variance alone, except for variance-focused
structures such as strangles. Relative to evidence for delta-hedged options (e.g.,
(BuechnerKelly2022)), the link to variance premia is weaker and less predictable in our unhedged
0DTE setting. Last-hour regressions in Appendix Table [tab:strat_ret_expl_1500] amplify this
pattern: fits are often higher, and realized skewness remains the dominant channel.

[^1]: Allowing interactions between combo dummies and other regressors leaves the main conclusions
intact. Adding Greeks directly as controls produced unstable and difficult-to-interpret estimates.

<!-- @section-type: robustness
  @key-claim: As an inference-robustness step, we re-estimate the key pooled regressions at 10:00 ET with combo fixed effects and date-clustered standard errors, and then adjust p-values for multiple testing using...
  @importance: supporting
  @data-source: none
  @equations: none
  @tables: tab:inference_cluster_mht
  @figures: none
-->

### Inference Robustness: Date Clustering and Multiple Testing

As an inference-robustness step, we re-estimate the key pooled regressions at 10:00 ET with combo
fixed effects and date-clustered standard errors, and then adjust p-values for multiple testing
using Benjamini-Hochberg FDR control (BenjaminiHochberg1995), following concerns about repeated
testing in predictability settings (HarveyLiuZhu2015).

> **Table** (tab:inference_cluster_mht): **Inference Robustness for Strategy PNL Drivers.** Reported coefficients are from pooled regressions with combo fixed effects and date-clustered standard errors; q-values are Benjamini-Hochberg adjustments across all reported tests.
> Source: `tables/ 0dte_inference_cluster_mht.tex`

Table [tab:inference_cluster_mht] shows that the strongest directional-skew links remain after
clustering and BH adjustment, while many weaker coefficients lose support. This sharpens the main
interpretation of the unconditional evidence: what matters for many 0DTE strategies is not generic
exposure to same-day variance, but directional asymmetry and the skewness channel. The conditional
question is therefore whether states observed by 10:00 ET can capture enough of that variation to
improve trading decisions in real time.

<!-- @section-type: conditional
  @key-claim: The conditional exercise asks whether these state links can be turned into implementable real-time rules. The answer is not uniform across strategies, but neither is it uniformly negative. Once the...
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

## Conditional Signals and Portfolio Use

The conditional exercise asks whether these state links can be turned into implementable real-time
rules. The answer is not uniform across strategies, but neither is it uniformly negative. Once the
problem is cast as a directional classification task and estimated under strict no-look-ahead
windows, several strategy-specific rules and diversified baskets deliver meaningful net out-of-
sample performance.

All strategies in this paper are unhedged and held to expiration. Their PNL is therefore driven by
entry cost and by the realized underlying move from entry to close. Entry costs are observable and
related to expected variance, but forecasting the *signed* daily index move is notoriously
difficult.[^1] A more realistic target is directionally asymmetric intraday payoffs and higher-
moment variation rather than fine-grained return magnitudes. This is why the conditional analysis
below treats direction prediction, probability calibration, and net economic value as separate
objects. The practical claim is therefore modest but useful. We do not show that 0DTE can be
forecast broadly with complex models; rather, we show that a narrow set of transparent, auditable
signals can improve timing for selected strategy families under disciplined real-time rules.

[^1]: Our baseline view is that short-horizon directional predictability in the index is
economically weak.

<!-- @section-type: conditional
  @key-claim: As a minimal implementable conditioning step, we sort trading days into low/mid/high regimes using full-sample terciles of the 10:00 ET 0DTE implied-variance measure (from our VIX-style estimator)....
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:vix_regime_1000
  @figures: none
-->

### A Simple State Filter at 10:00 ET: VIX Regimes (Ex-Post Cutoffs)

As a minimal implementable conditioning step, we sort trading days into low/mid/high regimes using
full-sample terciles of the 10:00 ET 0DTE implied-variance measure (from our VIX-style estimator).
Concretely, this variable is the integrated implied variance from 10:00 ET to end-of-day (16:00 ET),
constructed from SPXW 0DTE options using the VIX methodology. Regime thresholds are therefore
identified *ex post* (full-sample terciles), while the state variable itself is observed *ex ante*
at entry on each day. We then compute strategy PNL means across these regimes.

> **Table** (tab:vix_regime_1000): **A Simple State Filter: Strategy PNL by 10:00 ET VIX Regime.** Means are daily strategy PNL in % of underlying (10:00 ET to 16:00 ET), with equal weighting across moneyness combinations within each strategy and day. Regimes are based on full-sample terciles of the 10:00 ET integrated implied variance (10:00--16:00) from SPXW 0DTE options, constructed with the VIX methodology.
> Source: `tables/ 0dte_vix_regime_1000.tex`

Table [tab:vix_regime_1000] shows clear regime heterogeneity: downside-oriented structures (bear put
and put ratio spreads) earn substantially higher average PNL in high-VIX states than in low-VIX
states, while upside-oriented call-spread structures perform better in low-VIX states. Economically,
this suggests that even simple state partitions can alter unconditional performance profiles
materially, though statistical strength remains limited for most strategies in this static design.

<!-- @section-type: methodology
  @key-claim: To keep the conditioning exercise implementable, we use a strict out-of-sample protocol with no look-ahead construction of features. On each date, predictors include only variables observed by 10:00...
  @importance: supporting
  @data-source: none
  @equations: none
  @tables: tab:cond_rep_moneyness, tab:cond_model_zoo, tab:cond_oos
  @figures: none
-->

### Strict Out-of-Sample Trading Protocol

To keep the conditioning exercise implementable, we use a strict out-of-sample protocol with no
look-ahead construction of features. On each date, predictors include only variables observed by
10:00 ET: contemporaneous implied/state quantities (including the 10:00 ET SPXW 0DTE integrated
implied variance from 10:00 to 16:00, plus skew/slope proxies), lagged realized variables, and
lagged strategy-PNL terms. We estimate both expanding-window and rolling-window schemes (252
trading-day minimum) for sign prediction of net strategy PNL and evaluate them using hit rate,
probability calibration, and net economic value after execution costs. With this burn-in, OOS
forecasts begin in April 2019 and run through February 2, 2026. This design follows the practical
OOS discipline emphasized by (GoyalWelch2008) and aligns with the event-state perspective in
(KnoxLondonoSamadiVissingJorgensen2025).

To avoid pooling many overlapping strike designs in one conditional exercise, we fix one
representative near-ATM moneyness configuration for each strategy type. Candidate configurations
satisfy $\max_l |M_l-1| \le 0.01$; within each strategy we select the most frequent configuration by
trading-day coverage. The selected configurations are listed in Table [tab:cond_rep_moneyness] and
all lie within $\pm 0.5%$ of ATM.

> **Table** (tab:cond_rep_moneyness): **Representative Strategy Moneyness Configurations for Conditional Tests.** For each strategy type, we select one near-ATM configuration under $\max_l|M_l-1|\le 0.01$, choosing the configuration with the highest day coverage in the sample. The last column reports the maximum across legs of the absolute distance from ATM, $\max_l|M_l-1|$, not the width of the full structure. Thus, a configuration such as 0.995/0.997/1.003/1.005 has Max $|M-1|=0.005$ because its most extreme legs are 0.5% away from ATM.
> Source: `tables/ 0dte_conditional_representative_moneyness.tex`

After this representative-moneyness restriction, we keep the main-text conditional evidence fully
strategy-specific to avoid additional averaging across strategy panels. The benchmark specifications
in Tables [tab:cond_oos] and [tab:cond_oos_investment] use a deliberately parsimonious predictor
set: market-state variables plus lagged strategy-PNL terms. Richer baseline/GEX/flow/liquidity
stacks are examined separately in the model-zoo exercises, where GEX variables are flow/exposure
proxies based on traded volume, open interest, and leg Greeks rather than a dealer-inventory
reconstruction. Define net strategy return as

$$
y^{net}_{s,t}=PNL^{net}_{s,t}=reth\_und_{s,t}-c_{s,t},
$$

where $c_{s,t}$ is implementation cost (half-spread plus fixed 0.5bp).

We compare three prediction-to-position designs under one common OOS setup:

$$
**(A) Return target:**\quad \hat y_{s,t}=f_s(X_{s,t}), \qquad w^{(R)}_{s,t}=sign(\hat y_{s,t}),
**(B) Binary target (Logistic):**\quad
p_{s,t}=\Pr(y^{net}_{s,t}>0\mid X_{s,t})=\Lambda(\alpha_s+\beta_s^\top X_{s,t}),
\log\!\left(\frac{p_{s,t}}{1-p_{s,t}}\right)=\alpha_s+\beta_s^\top X_{s,t},
**(B1) Hard map:**\quad w^{(H)}_{s,t}=sign(p_{s,t}-0.5)\in\{-1,+1\},
**(B2) Soft map:**\quad w^{(S)}_{s,t}=2p_{s,t}-1\in[-1,+1].
$$

For all three designs, realized trading return is $r_{s,t}=w_{s,t} y^{net}_{s,t}$. In words, hard
mapping is full-size directional timing, while soft mapping is confidence-weighted directional
timing.

The economic difference is simple: (A) forecasts payoff magnitude and then trades direction; (B1)
forecasts only direction and trades full size; (B2) forecasts direction and scales position size by
confidence. In our implementation, each strategy is estimated separately at 10:00 ET using only
information available by that time (contemporaneous implied-state predictors and lagged realized
predictors), with no-look-ahead feature construction.

We keep logistic as the benchmark in strategy-level tests for three reasons. First, the trading
decision is directional (long vs short), so modeling $\Pr(y^{net}>0)$ is directly aligned with the
traded object. Second, logistic outputs are bounded probabilities, which allows transparent
calibration and Brier-score diagnostics in addition to hit rates. Third, in short-horizon 0DTE data
with noisy payoffs and limited per-strategy sample sizes, a low-variance parametric benchmark is
less prone to unstable overfitting than higher-capacity nonlinear models. For a practitioner
audience, this simplicity is a feature rather than a limitation: the benchmark is easy to audit,
stable enough to monitor, and disciplined enough to update under rolling OOS governance.

> **Table** (tab:cond_model_zoo): **Which Forecasting Target Is Most Usable?** Each row compares the same model family under three implementations on the same representative stacked OOS panel: we first estimate strategy-specific models separately (one representative moneyness per strategy), then stack resulting strategy-day OOS returns across strategies and dates. Thus, pooling here is at the evaluation stage (stacked strategy-day observations), not a single cross-strategy model fit. Implementations are: (A) direct return prediction with $w_t=sign(\hat y_t)$, (B1) logistic binary prediction with hard mapping $w_t=sign(\hat p_t-0.5)$, and (B2) logistic binary prediction with soft mapping $w_t=2\hat p_t-1$. Columns report annualized SR net and mean net PNL (bps).
> Source: `tables/ 0dte_conditional_target_choice.tex`

**Construction details for Table [tab:cond_oos**.] Let $s$ index strategy and $t$ index date. For
each strategy $s$, we estimate a separate logistic model on that strategy's own daily series only
(no cross-strategy pooling): $y_{s,t}=\mathbf{1}\{PNL^{net}_{s,t}>0\}$ with predictors observed by
10:00 ET. In the benchmark, these predictors are the 10:00 ET implied-state variables
$(IV_t,IS_t,slope\_up_t,slope\_dn_t)$, one-day-lagged realized SPX moments, and lagged strategy-PNL
terms $(P^{(1)},\bar P^{(5)},\sigma^{(5)}_P)$. After representative-moneyness selection (Table
[tab:cond_rep_moneyness]), each strategy contributes one moneyness configuration; we do not average
across multiple strike designs in this table. At each OOS date $t$, estimation uses data through
$t-1$ only, with either expanding or 252-day rolling windows; predictors are standardized within the
training window, and the logistic model is estimated with L2 regularization. We then form $\hat
p_{s,t}$, trade with $sign(\hat p_{s,t}-0.5)$, and report OOS diagnostics.

> **Table** (tab:cond_oos): **Strategy-Level OOS Timing Performance at 10:00 ET.** Each strategy is estimated separately with a logistic benchmark on the representative moneyness configuration (Table [tab:cond_rep_moneyness]), using only 10:00 ET information: contemporaneous implied/state variables, lagged realized variables, and lagged strategy-PNL terms. No cross-strategy pooling is used in this table; each row is one strategy under one protocol. We report two window rules (expanding and rolling, minimum 252 trading days). Directional signal is $sign(\hat{p}_t-0.5)$ and net daily strategy return is this sign times strategy PNL net of half-spread and 0.5bp. OOS forecasts begin after the 252-day burn-in in April 2019 and run through 01/2026. Reported columns are hit rate (%), Brier score, calibration slope, mean net PNL (bps), annualized SR net, and OOS observations.
> Source: `tables/ 0dte_conditional_oos.tex`

Table [tab:cond_model_zoo] shows a clear ranking of target designs. Direct return prediction remains
weak in this short-horizon setting: only the Elastic Net and LightGBM regressions stay modestly
positive after costs, while the other return-target specifications are negative. By contrast, binary
designs are materially stronger. Within the binary setup, hard mapping dominates soft mapping in
four of the five model families once costs are applied, with the ridge-logit and elastic-net-logit
variants reaching SRs slightly above 1.0. We therefore keep binary-hard as the main implementation
and treat soft mapping as robustness. For strategy-level tests, we retain the logistic benchmark
because it remains stable across mappings and provides transparent probability diagnostics.

Table [tab:cond_oos] shows that the fixed logistic benchmark delivers economically meaningful
strategy-specific OOS results under both the expanding and rolling protocols. Put ratio spreads are
the strongest individual series, with 2.58 bps mean net and an SR of 1.26 in the expanding
specification. Strangle/straddle structures remain positive, although more modest once the 2025--
2026 extension is included. Iron butterfly/condor structures still retain an attractive Sharpe ratio
of 0.82. Other strategies are weaker, and call ratio spreads remain unattractive. Conditioning
therefore helps, but selectively rather than uniformly.

<!-- @section-type: conditional
  @key-claim: To translate Table [tab:cond_oos] into an investable object, we construct daily OOS strategy returns from the same strategy-specific logistic benchmark signals using a common rolling-window...
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:cond_oos_investment
  @figures: fig:cond_oos_investment_ts
-->

### OOS Portfolio Implementation

To translate Table [tab:cond_oos] into an investable object, we construct daily OOS strategy returns
from the same strategy-specific logistic benchmark signals using a common rolling-window estimation
rule (252-trading-day window) for all strategies. The strategy signal is then applied to the same
representative moneyness configuration and net-PNL definition as in Table [tab:cond_oos].

> **Table** (tab:cond_oos_investment): **Portfolio Implementation of OOS Strategy Signals.** For each strategy, we use the same logistic benchmark as in Table [tab:cond_oos], estimated with a 252-trading-day rolling window. Daily strategy return is sign($\hat{p}_t-0.5$)$\times$net strategy PNL for the representative moneyness configuration. OOS forecasts begin after the 252-day burn-in in April 2019 and run through 01/2026. The table reports all strategy series and equal-weight baskets (top-three by mean net PNL, top-three by SR net, and all-strategies), with columns for mean net (bps), annualized SR net, hit rate (%), long share (%), ES$_{1%}$, worst day, max drawdown, and days.
> Source: `tables/ 0dte_conditional_oos_investment.tex`

> **Figure** (fig:cond_oos_investment_ts): **Cumulative Net PNL of OOS Strategy Sleeves and Baskets.** Panel A plots cumulative net PNL (bps) for all individual strategy series. Panel B plots three equal-weight baskets built from the same strategy-level series: top-three selected by mean net PNL, top-three selected by SR net, and all-strategies basket. Each strategy uses the same logistic benchmark as in Table [tab:cond_oos], estimated with a 252-trading-day rolling window.

> File: figures/ oos_conditional_investment_ts.pdf

Table [tab:cond_oos_investment] and Figure [fig:cond_oos_investment_ts] show the strategy-specific
implementation directly implied by Table [tab:cond_oos]. Diversification still matters in the
extended sample. The top-three basket ranked by SR net reaches a net Sharpe ratio of 1.27, the top-
three basket ranked by mean net PNL reaches 1.17, and even the all-strategies basket reaches 1.01.
Equal weighting does not eliminate tail risk, but it materially smooths strategy-specific drawdowns
and makes the conditional evidence easier to interpret as a tradable object rather than a collection
of isolated strategy series. From an implementation standpoint, this basket evidence is at least as
important as the single-strategy ranking. A professional user is more likely to allocate a modest
risk budget across a governed sleeve than to concentrate on one ratio-spread signal. The diversified
baskets therefore provide the more realistic benchmark for portfolio use, even if they do not remove
tail risk altogether.

<!-- @section-type: conclusion
  @key-claim: This paper studies realized payoff distributions of 0DTE SPXW options and common multi-leg strategy templates from 09/2016 to 01/2026. A positive 0DTE variance risk premium exists, but at same-day...
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

## Conclusion and Practical Implications

This paper studies realized payoff distributions of 0DTE SPXW options and common multi-leg strategy
templates from 09/2016 to 01/2026. A positive 0DTE variance risk premium exists, but at same-day
horizons its economic magnitude is modest and it is not the central object from an implementation
perspective. Once attention shifts from a synthetic variance object to actual option positions and
feasible multi-leg structures, the main empirical fact is the wide and state-dependent distribution
of realized outcomes rather than a single unconditional average edge. For practitioners, the
relevant benchmark is therefore not an idealized premium, but the net PNL distribution of actual
traded structures after realistic frictions.

The dominant message of the unconditional evidence is risk. Strategy-level distributions are
dispersed, asymmetric, and fragile across regimes. After execution costs, downside risk remains
large relative to mean carry: Table [tab:tail_risk_diag] reports substantial ES$_{1%}$ values,
severe worst-day outcomes, and meaningful cumulative drawdowns even for strategies with positive
average PNL. Mean PNL on its own is therefore an inadequate summary statistic for 0DTE
implementation. For allocators and risk managers, this argues against viewing 0DTE as an income
sleeve or a generic premium source. These positions consume downside capital quickly, and they
should be assessed with expected shortfall, worst-day realizations, and drawdown limits rather than
with average PNL alone.

At the same time, the conditional evidence is stronger than a purely pessimistic reading would
suggest. The main economic link runs through directional asymmetry and skewness rather than through
realized variance alone (Tables [tab:strat_ret_expl] and [tab:inference_cluster_mht]). When the
forecasting problem is posed as directional classification and evaluated under strict out-of-sample
discipline, several strategy-specific rules remain attractive after costs. In the updated benchmark,
put ratio spreads reach a net Sharpe ratio of 1.26 and iron butterfly/condor structures 0.82 in
Table [tab:cond_oos]; at the portfolio level, the top-three-by-SR basket reaches 1.27 and the all-
strategies basket reaches 1.01 in Table [tab:cond_oos_investment]. The evidence is therefore
selective rather than diffuse: some strategy families can be timed meaningfully, many others cannot.
The practical implication is not that every 0DTE strategy can be forecast, but that a small subset
can be deployed more intelligently when signals are transparent, estimation is disciplined, and
implementation remains diversified.

The main limitation of the paper is scope. We study SPXW 0DTE positions held to the close, without
dynamic intraday hedging and without a causal market-impact design. Our conditioning variables are
restricted to information available by 10:00 ET, which is intentional but demanding. For that
reason, if 0DTE is used in live portfolios, we view the appropriate use case as a small tactical
overlay with explicit turnover control, tail-risk budgeting, and ongoing out-of-sample monitoring,
not as a standing core allocation. The natural next step is therefore not to search for one
universal 0DTE rule, but to continue updating the data under locked out-of-sample governance, extend
event conditioning, and monitor whether the better-performing strategy families remain stable as the
market evolves.

<!-- @section-type: acknowledgments
  @key-claim: The author used OpenAI Codex with the GPT-5.4 model during manuscript preparation. The tool was used to download and preprocess data through author-directed code workflows, run unit tests, regenerate...
  @importance: peripheral
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

## Acknowledgments and AI Use Disclosure

The author used OpenAI Codex with the GPT-5.4 model during manuscript preparation. The tool was used
to download and preprocess data through author-directed code workflows, run unit tests, regenerate
tables and figures from the underlying data and code, and provide recommendations on manuscript
formatting for *Financial Analysts Journal*. It was used to accelerate reproducible data processing,
testing, and document preparation. All empirical design choices, interpretation of results, and
final manuscript text were reviewed and approved by the author.

{\openup -8pt

}

} }

}

<!-- @section-type: appendix
  @key-claim: 
  @importance: supporting
  @data-source: none
  @equations: none
  @tables: none
  @figures: none
-->

## Additional Tables and Figures


<!-- @section-type: results
  @key-claim: 
  @importance: core
  @data-source: none
  @equations: none
  @tables: none
  @figures: fig:strat-example
-->

### Strategy Templates

> **Figure** (fig:strat-example): **Sample Payoffs of Option Strategies.** The figure shows illustrative terminal payoffs for the strategy templates used in the paper. Payoffs exclude entry premium. Converting payoff to PNL shifts debit structures downward by the paid premium and credit structures upward by the received premium.

> Panel A: Straddle [figures/ strat_payoffs.pdf]
> Panel B: Strangle [figures/ strat_payoffs.pdf]
> Panel C: Risk Reversal [figures/ strat_payoffs.pdf]
> Panel D: Bull Call Spread [figures/ strat_payoffs.pdf]
> Panel E: Bear Put Spread [figures/ strat_payoffs.pdf]
> Panel F: Call Ratio Spread [figures/ strat_payoffs.pdf]
> Panel G: Put Ratio Spread [figures/ strat_payoffs.pdf]
> Panel H: Iron Butterfly [figures/ strat_payoffs.pdf]
> Panel I: Iron Condor [figures/ strat_payoffs.pdf]

<!-- @section-type: results
  @key-claim: 
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:0dte_stratret2022_2023, tab:0dte_stratret2024_2026
  @figures: fig:optret-1600prev, fig:optret-1300, fig:optret-1500, fig:strat-ret-1600prev, fig:strat-ret-1300, fig:strat-ret-1500
-->

### Unconditional Trading Rules

> **Figure** (fig:optret-1600prev): **Option Returns, 16:00 ET on the Previous Day to Next-Day Expiry.** The figure provides statistics on profitability of call and put option buying at 16:00 ET on the previous day and holding to expiry at 16:00 ET on the next day. These contracts are one day from expiry at entry and become 0DTE on the following morning. Panels on the left show realized return in % relative to option mid-price. Panels on the right show option realized profit (payoff - mid-price) per one unit of underlying relative to underlying price, in %. Bars show mean values, and each bar is accompanied by median, 25th and 75th percentiles. X-axis labels show moneyness of the analyzed options. 
The sample period is from 09/2016 to 01/2026.

> Panel A: Call Returns, 16:00 ET Previous Day [figures/ calls_reth_bymnes_bars.pdf]
> Panel B: Call PNL/Underlying, 16:00 ET Previous Day [figures/ calls_reth_und_bymnes_bars.pdf]
> Panel C: Put Returns, 16:00 ET Previous Day [figures/ puts_reth_bymnes_bars.pdf]
> Panel D: Put PNL/Underlying, 16:00 ET Previous Day [figures/ puts_reth_und_bymnes_bars.pdf]

> **Figure** (fig:optret-1300): **0DTE Option Returns, 13:00 ET to Expiry.** The figure provides statistics on profitability of naked 0DTE call and put option buying at 13:00 ET and holding to expiry at 16:00 ET. Panels on the left show realized return in % relative to option mid-price. Panels on the right show option realized profit (payoff - mid-price) per one unit of underlying relative to underlying price, in %. Bars show mean values, and each bar is accompanied by median, 25th and 75th percentiles. X-axis labels show moneyness of the analyzed options. 
The sample period is from 09/2016 to 01/2026.

> Panel A: Call Returns, 13:00 ET [figures/ calls_reth_bymnes_bars.pdf]
> Panel B: Call PNL/Underlying, 13:00 ET [figures/ calls_reth_und_bymnes_bars.pdf]
> Panel C: Put Returns, 13:00 ET [figures/ puts_reth_bymnes_bars.pdf]
> Panel D: Put PNL/Underlying, 13:00 ET [figures/ puts_reth_und_bymnes_bars.pdf]

> **Figure** (fig:optret-1500): **0DTE Option Returns, 15:00 ET to Expiry.** The figure provides statistics on profitability of naked 0DTE call and put option buying at 15:00 ET and holding to expiry at 16:00 ET. Panels on the left show realized return in % relative to option mid-price. Panels on the right show option realized profit (payoff - mid-price) per one unit of underlying relative to underlying price, in %. Bars show mean values, and each bar is accompanied by median, 25th and 75th percentiles. X-axis labels show moneyness of the analyzed options. 
The sample period is from 09/2016 to 01/2026.

> Panel A: Call Returns, 15:00 ET [figures/ calls_reth_bymnes_bars.pdf]
> Panel B: Call PNL/Underlying, 15:00 ET [figures/ calls_reth_und_bymnes_bars.pdf]
> Panel C: Put Returns, 15:00 ET [figures/ puts_reth_bymnes_bars.pdf]
> Panel D: Put PNL/Underlying, 15:00 ET [figures/ puts_reth_und_bymnes_bars.pdf]

> **Figure** (fig:strat-ret-1600prev): **Static Option Strategies, 16:00 ET on the Previous Day to Next-Day Expiry.** The figure provides statistics on profitability of several well-known option strategies based on short-dated call and put options; positions are taken at 16:00 ET on the previous day and held to expiry at 16:00 ET on the next day. These positions are one day from expiry at entry. All panels show option strategy realized profit (payoff - mid-price) per one unit of underlying relative to the underlying price, in %. Bars show mean values, and each bar is accompanied by median, 25th and 75th percentiles. X-axis labels show combination of moneyness for options used for a strategy. The sample period is from 09/2016 to 01/2026.

> Panel A: Strangle PNL/Underlying [figures/ strangle_reth_und_bymnes_bars.pdf]
> Panel B: Iron Butterfly/ Condor PNL/Underlying [figures/ iron_condor_reth_und_bymnes_bars.pdf]
> Panel C: Risk Reversal PNL/Underlying [figures/ risk_reversal_reth_und_bymnes_bars.pdf]
> Panel D: Bull Call Spread PNL/Underlying [figures/ bull_call_spread_reth_und_bymnes_bars.pdf]
> Panel E: Bear Put Spread PNL/Underlying [figures/ bear_put_spread_reth_und_bymnes_bars.pdf]
> Panel F: Call Ratio Spread PNL/Underlying [figures/ call_ratio_spread_reth_und_bymnes_bars.pdf]
> Panel G: Put Ratio Spread PNL/Underlying [figures/ put_ratio_spread_reth_und_bymnes_bars.pdf]

> **Figure** (fig:strat-ret-1300): **0DTE Static Option Strategies, 13:00 ET to Expiry.** The figure provides statistics on profitability of several well-known option strategies based on 0DTE call and put options; positions are taken at 13:00 ET and held to expiry at 16:00 ET. All panels show option strategy realized profit (payoff - mid-price) per one unit of underlying relative to the underlying price, in %. Bars show mean values, and each bar is accompanied by median, 25th and 75th percentiles. X-axis labels show combination of moneyness for options used for a strategy. The sample period is from 09/2016 to 01/2026.

> Panel A: Strangle PNL/Underlying [figures/ strangle_reth_und_bymnes_bars.pdf]
> Panel B: Iron Butterfly/ Condor PNL/Underlying [figures/ iron_condor_reth_und_bymnes_bars.pdf]
> Panel C: Risk Reversal PNL/Underlying [figures/ risk_reversal_reth_und_bymnes_bars.pdf]
> Panel D: Bull Call Spread PNL/Underlying [figures/ bull_call_spread_reth_und_bymnes_bars.pdf]
> Panel E: Bear Put Spread PNL/Underlying [figures/ bear_put_spread_reth_und_bymnes_bars.pdf]
> Panel F: Call Ratio Spread PNL/Underlying [figures/ call_ratio_spread_reth_und_bymnes_bars.pdf]
> Panel G: Put Ratio Spread PNL/Underlying [figures/ put_ratio_spread_reth_und_bymnes_bars.pdf]

> **Figure** (fig:strat-ret-1500): **0DTE Static Option Strategies, 15:00 ET to Expiry.** The figure provides statistics on profitability of several well-known option strategies based on 0DTE call and put options; positions are taken at 15:00 ET and held to expiry at 16:00 ET. All panels show option strategy realized profit (payoff - mid-price) per one unit of underlying relative to the underlying price, in %. Bars show mean values, and each bar is accompanied by median, 25th and 75th percentiles. X-axis labels show combination of moneyness for options used for a strategy. The sample period is from 09/2016 to 01/2026.

> Panel A: Strangle PNL/Underlying [figures/ strangle_reth_und_bymnes_bars.pdf]
> Panel B: Iron Butterfly/ Condor PNL/Underlying [figures/ iron_condor_reth_und_bymnes_bars.pdf]
> Panel C: Risk Reversal PNL/Underlying [figures/ risk_reversal_reth_und_bymnes_bars.pdf]
> Panel D: Bull Call Spread PNL/Underlying [figures/ bull_call_spread_reth_und_bymnes_bars.pdf]
> Panel E: Bear Put Spread PNL/Underlying [figures/ bear_put_spread_reth_und_bymnes_bars.pdf]
> Panel F: Call Ratio Spread PNL/Underlying [figures/ call_ratio_spread_reth_und_bymnes_bars.pdf]
> Panel G: Put Ratio Spread PNL/Underlying [figures/ put_ratio_spread_reth_und_bymnes_bars.pdf]

> **Table** (tab:0dte_stratret2022_2023): **0DTE Static Option Strategies Performance.** The table shows the summary statistics for the holding-period PNL of several well-known option strategies based on 0DTE call and put options; positions are taken at 10:00 ET and held to expiry at 16:00 ET. The statistics are based on option strategies' realized profit per one unit of underlying relative to underlying price $(payoff - mid price)/underlying price \times 100%$. The SR, p.a. is the Sharpe Ratio annualized by scaling it up by $\sqrt{252}$. The sample period is from 01/2022 to 12/2023.
> Source: `tables/ 0dte_stratret2022_2023.tex`

> **Table** (tab:0dte_stratret2024_2026): **0DTE Static Option Strategies Performance.** The table shows the summary statistics for the holding-period PNL of several well-known option strategies based on 0DTE call and put options; positions are taken at 10:00 ET and held to expiry at 16:00 ET. The statistics are based on option strategies' realized profit per one unit of underlying relative to underlying price $(payoff - mid price)/underlying price \times 100%$. The SR, p.a. is the Sharpe Ratio annualized by scaling it up by $\sqrt{252}$. The sample period is from 01/2024 to February 2, 2026.
> Source: `tables/ 0dte_stratret2024_2026.tex`

<!-- @section-type: conditional
  @key-claim: 
  @importance: core
  @data-source: none
  @equations: none
  @tables: tab:cond_feature_dictionary, tab:strat_ret_expl_1500
  @figures: none
-->

### Additional Conditional Tables

> **Table** (tab:cond_feature_dictionary): **Conditional Predictor Dictionary for Equations in This Subsection.** The table maps the compact notation used in the feature equations to construction formulas and economic interpretation. Family labels are aligned with model-zoo feature groups (Baseline, Flow, GEX, Liquidity); *market controls* are included in all specifications. Duplicate-key preprocessing at $(datetime,option\_type,moneyness)$ uses means for price/Greek-like fields (including open interest) and sums for traded/depth flow fields before leg aggregation.

> **Table** (tab:strat_ret_expl_1500): **Option Strategies PNL vs. Implied and Realized Moments.** The table shows the results of regressing realized PNL of option strategies (15:00 ET to expiry at 16:00 ET) on implied and realized distribution moments. PNL is specified per one unit of underlying relative to underlying price $(payoff - mid price)/underlying price \times 100%$. The result in each column is based on a pooled regression of strategy PNL for several moneyness combinations, including combo fixed effects (Combo FE) and date-clustered standard errors. The sample period is from 09/2016 to 01/2026.
> Source: `tables/ strat_PNL_rv_rvrs_ivrv_ivisrvrst15:00:00.tex`
