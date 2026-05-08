# EDA Output Interpretation Guide

How to read each plot and section of the soiling signal report produced by
`scripts/5_eda/soiling_signals.py`.

Generated outputs live in `artifacts/eda/`. The report is
`eda_signal_report.md`; plots are in `artifacts/eda/plots/`.

---

## What This Project Is About

This project monitors a large solar power plant — thousands of panels spread
across a field in Sri Lanka. Over time, dust, pollen, bird droppings, and
general grime settle on the panels. This is called **soiling**. Dirty panels
produce less electricity than clean ones.

The question this project tries to answer: **can we predict how fast panels get
dirty and when they should be cleaned, using the data we already collect?**

The EDA (Exploratory Data Analysis) is the first real check: before building any
predictive models, we need evidence that soiling is even *detectable* in our
data. If we cannot see it, there is no point building a model to predict it.

---

## Key Terminology

These definitions cover every domain-specific term that appears in the plots
and report. No solar engineering background is assumed.

**Soiling**: The accumulation of dust, dirt, and particulates on solar panel
surfaces. It reduces the amount of sunlight reaching the photovoltaic cells,
which reduces electricity output. Removed naturally by rain or manually by
cleaning crews.

**Inverter**: A device that converts the DC electricity from solar panels into
AC electricity for the grid. Each inverter serves a group of panels. This plant
has 34 inverters total; we monitor 6 of them (3 per physical block).

**Block**: The plant is split into two physical sections. Block B2 is on one
side, Block B1 on the other.

**Tier-1 / T1 (B2 block)**: The three inverters from Block B2 (B2-08, B2-13,
B2-17). Their data is the most complete and reliable (availability 0.77-1.0).
We use this as *training data* — the data we trust most.

**Tier-2 / T2 (B1 block)**: The three inverters from Block B1 (B1-08, B1-01,
B1-13). Their data has more gaps (availability 0.10-1.0). We use this as a
*validation set* — if patterns we find in B2 also appear in B1, we know the
pattern is real and plant-wide, not a fluke of one block.

**Normalised output**: Daily energy produced divided by daily sunlight received.
This removes the effect of sunny vs cloudy days. If normalised output drops on
a sunny day, something is wrong with the panels (likely dirt).

**Rolling clean baseline**: The "best" normalised output the plant achieved in
the last 30 days (specifically the 95th percentile). This represents "what
output should be if panels were clean." Soiling is measured relative to this.

**Loss proxy (`t1_performance_loss_pct_proxy`)**: The main metric. It answers
"how much worse is the plant performing today compared to its recent best?" It
is a percentage from 0 to ~80. A value of 0% means "performing as well as
expected." A value of 30% means "producing 30% less energy than it should." It
goes up as panels get dirty and drops when they get cleaned.

**Performance Ratio (PR)**: Similar to normalised output but scaled by the
panel's rated capacity. Our PR values are inflated (~240 instead of the normal
~0.8) due to a known unit mismatch in the ground irradiance sensor — the
*relative trends* are still valid, just the absolute numbers are not meaningful.

**Cycle**: The period between two "reset events" (either rain or a cleaning
campaign). During a cycle, dust accumulates. At the next rain or cleaning, it
resets. Each cycle gets a numeric ID (`cycle_id`).

**Cycle deviation (`cycle_deviation_pct`)**: Within each cycle, this measures
how far performance has fallen from the best day in that cycle. Starts near 0%
after a reset and rises as panels get dirtier.

**Dry spell**: A consecutive stretch of days with no rain. During dry spells,
dust accumulates uninterrupted — these are the best windows for measuring the
soiling rate.

**HQ days / Training-ready days**: Days that passed the strictest data quality
filter — `transfer_quality_tier == "high"` AND `flag_count == 0`. These are the
246 most trustworthy days out of 361 total. All statistical tests in the EDA
use only these days.

**PM10 / PM2.5**: Particulate Matter — tiny particles floating in the air,
measured in micrograms per cubic metre (ug/m3). PM10 is coarser dust (diameter
< 10 micrometres), PM2.5 is finer (< 2.5 um). These are what settle on panels
and cause soiling.

**Solcast**: A commercial satellite weather data provider. We get PM levels,
rainfall, temperature, humidity, wind, and cloud data from them. This data is
independent of our ground sensors at the plant.

**pvlib**: An open-source Python library for solar energy modelling. It has
physics-based models that estimate how dirty panels should be based on rainfall,
dust levels, and panel tilt angle. We use it as an independent reference to
compare against our data-driven loss proxy.

**Wilcoxon signed-rank test**: A statistical test that checks whether paired
measurements differ systematically (e.g., is loss at the end of a dry spell
consistently higher than at the start?). A p-value below 0.05 means the
difference is statistically significant (unlikely to be random chance).

**Partial correlation**: The correlation between two variables after
mathematically removing the influence of one or more confounding variables. For
example, the partial correlation between PM10 and loss rate *controlling for
cloud opacity* tells us whether dust predicts soiling after stripping out the
weather effect.

**Confound / Confounder**: A third variable that distorts the apparent
relationship between two other variables. In this project, cloud opacity is the
main confounder: cloudy days simultaneously reduce PM (less dust in humid
air) and inflate the loss proxy (normalised output looks worse against the
clear-sky baseline). Naive analyses that ignore this confound will produce
misleading results.

---

## The Three Go/No-Go Signals

The entire EDA is organised around three questions. If we can answer "yes" to
at least two, the research is worth continuing to the modeling phase.

**Signal 1 — "Can we see the sawtooth?"**
When panels get dirty gradually and then get cleaned suddenly (by rain), a
time-series of performance loss should look like a sawtooth wave — slowly
rising, then sharply dropping, repeatedly. This shape is *unique* to soiling.
Equipment failures, temperature changes, and sensor noise do not produce it.

**Signal 2 — "Does dustier air mean faster soiling?"**
If PM10 (dust in the air) is high, panels should get dirty faster. If we can
show a statistical link between dust levels and the rate of performance decline,
that means environmental data can *predict* soiling — which is the foundation
for building a forecasting model.

**Signal 3 — "Does rain clean the panels?"**
After heavy rain, the loss proxy should drop (panels got washed, performance
recovered). If rain visibly resets soiling, it confirms that what we are
measuring really is dirt, not some other issue like equipment degradation.

---

## Signal 1: Sawtooth Detection

The sawtooth is the fingerprint of soiling: gradual performance decline as dust
accumulates, followed by a sudden recovery when rain washes the panels or a
cleaning crew intervenes. Temperature drift, equipment degradation, and sensor
noise do not produce this shape.

### s1_loss_proxy_timeseries.png

**Layout**: Two vertically-stacked panels sharing the same time axis.

- **Top panel**: Two metrics on dual axes.
  - Left y-axis (purple): `t1_performance_loss_pct_proxy` (Tier-1 loss proxy).
    Higher values mean worse performance relative to the rolling clean baseline.
  - Right y-axis (amber, faint): `domain_soiling_index` (cumulative DSPI).
    This is a physics-based soiling pressure estimate built entirely from
    environmental data (PM, humidity, precipitation). It rises during dry,
    dusty periods and resets after significant rain or cleaning.
  - Faint blue vertical lines mark days with significant rain (>= 5 mm).
  - Shaded orange bands mark the three known cleaning campaigns (Sep/Oct/Nov
    2025, 20th-30th).
- **Bottom panel**: Daily precipitation (mm) as a blue bar chart.

**What to look for**:

- Gradual upward slopes between rain events (soiling accumulation) in the
  loss proxy.
- Sudden downward steps at rain lines or within cleaning bands (recovery).
- Multi-week ascending runs during dry spells, especially Feb-Apr.
- Whether the amber DSPI line rises and falls at similar times as the loss
  proxy. Agreement indicates the physics-based model captures the same
  soiling dynamics that the plant performance data shows. Divergences may
  reveal periods dominated by non-soiling losses (equipment issues, clouds).

**Caveat**: This plot shows all data, not just high-quality days. Cloudy periods
depress normalised output against the clear-sky baseline, so loss proxy spikes
on overcast days are weather artefacts, not soiling. Focus on trends during
clear, dry stretches. The DSPI line is unaffected by clouds since it uses
only environmental satellite data.

### s1_per_inverter_output.png

**Layout**: Six vertically-stacked panels, one per inverter (B2-08, B2-13,
B2-17, B1-08, B1-01, B1-13). Same rain and cleaning overlays.

**What to look for**:

- Sawtooth pattern should appear per-inverter even if the aggregated proxy is
  noisier.
- B2 inverters (top three) should have smoother, more complete traces because
  their data availability is consistently high (0.77-1.0).
- B1 inverters (bottom three) may show gaps from low-availability periods.
- Look for PR jumps within cleaning campaign bands — these confirm that
  cleaning events are captured in the per-inverter data.
- If sawtooth is visible in B2 but not B1, it may be masked by B1 data gaps
  rather than absent.

### s1_cycle_deviation.png

**Layout**: Single panel. `cycle_deviation_pct` over time, with faint grey
vertical lines at each cycle boundary (where `cycle_id` changes).

**What to look for**:

- Each cycle should start near 0% deviation (best performance in the cycle)
  and rise toward the cycle peak before resetting at the next rain event.
- This feature was engineered to isolate within-cycle soiling by normalising
  each day's output against the best day in its rain-to-rain cycle.
- Short cycles (median 4 days at this site due to frequent rain) compress
  the sawtooth. Look for the handful of longer cycles (7+ days) where the
  rising pattern is clearest.
- If the plot looks like random noise with no upward ramps, the cycle-aware
  approach is not capturing a soiling pattern.

### s1_dryspell_slopes.png

**Layout**: Single panel. The HQ loss proxy is plotted in faint purple.
Overlaid in orange are linear regression lines fitted within each dry spell
(consecutive no-rain days >= 3).

**What to look for**:

- Orange lines sloping **upward** represent soiling accumulation (positive
  rate, typically 0.1-0.5 %/day for tropical sites).
- Orange lines sloping **downward** may indicate equipment recovery, baseline
  shifts, or noise.
- The report gives the median slope and what fraction of spells have positive
  slopes. A median in the 0.05-1.0 %/day range with >50% positive slopes
  supports the sawtooth signal.
- Only 10 dry spells of >= 3 days exist (rain is frequent at this site),
  so statistical power is limited.

---

## Signal 2: PM/Dust Correlation

Tests whether airborne particulate matter (PM10, PM2.5) predicts how fast
panels soil. If PM does not predict soiling rate, the ML model cannot forecast
soiling from environmental data alone.

### How Signal 2 correlations work

Every correlation in this section has two sides:

- **Side A — Environmental inputs** (independent variables): PM10, PM2.5,
  rainfall, humidity, wind, cumulative dust since rain, days since rain, etc.
  These come from Solcast satellite data and are available without any solar
  plant — they describe what the atmosphere is doing.
- **Side B — Observed plant performance** (dependent variables): metrics
  derived from the plant's actual energy generation that measure how well the
  panels are converting sunlight into electricity.

The go/no-go question is: **do environmental factors predict what actually
happens to the plant?** This requires crossing data domains — environment on
one side, real performance on the other. Correlating one environmental feature
against another environmental feature would be circular and prove nothing about
whether soiling affects the plant.

**The three performance targets (Side B):**

| Target | What it measures | Derived from |
|---|---|---|
| `t1_performance_loss_pct_proxy` | All-cause daily performance deficit vs the rolling 30-day clean baseline | Actual Tier-1 energy generation and irradiance |
| `t1_perf_loss_rate_14d_pct_per_day` | How fast loss is changing (14-day slope of the proxy) | 14-day rolling regression on the loss proxy above |
| `cycle_deviation_pct` | Within-cycle performance decline from the cycle's best day | Actual energy generation normalised by irradiance, reset at each rain/cleaning event |

**Which target matters most for go/no-go?** `cycle_deviation_pct` is the
cleanest soiling signal. Because it resets at every rain or cleaning event, it
measures only within-cycle decline and removes long-term baseline drift and
seasonal effects. The `vs cycle deviation` column in the partial correlation
table is the most informative for the go/no-go decision.

The loss proxy is noisier because it includes weather effects (cloudy days look
like worse performance even though the panels are not dirtier). The loss rate
captures trends but is smoothed over 14 days, blurring short-cycle signals.

**Why not use an environmental soiling estimate as the target?** A feature
built purely from environmental data (e.g., cumulative PM weighted by humidity)
would correlate with PM10 and rainfall by construction — you put those inputs
in, so getting correlations back out proves nothing. The test must cross from
*environment* to *observed performance* to demonstrate that soiling is real and
detectable in the plant's energy output.

**Critical context**: The raw PM10 correlation with loss proxy is **negative**
(r = -0.248) — counterintuitive. This is because dry, clear weather brings both
high PM and good system performance simultaneously. Cloud opacity (r = -0.405)
is a stronger raw correlator than any dust feature. The EDA must deconfound
weather effects to find the real PM-soiling relationship.

### s2_pm10_scatter_panels.png

**Layout**: Two side-by-side scatter panels.

- **Left**: All HQ days. `pm10_mean` (x-axis) vs
  `t1_perf_loss_rate_14d_pct_per_day` (y-axis), colour-coded by season
  (amber = dry, teal = wet). Annotated with Pearson r and Spearman rho.
- **Right**: Same scatter restricted to clear-sky HQ days only (cloud
  opacity < 25th percentile). This removes the main weather confounder.

**What to look for**:

- Left panel will likely show no clear pattern or a confounded relationship.
- Right panel (clear-sky only) should show a more positive or at least
  non-negative correlation if dust truly drives soiling. If r flips from
  negative to positive, that is evidence of successful deconfounding.
- The right panel has fewer points (~90 clear days) so expect more scatter.

### s2_top_predictors_vs_deviation.png

**Layout**: Three side-by-side scatter panels showing the strongest predictors
of cycle deviation. Left: `days_since_last_rain`, middle:
`cumulative_pm25_since_rain`, right: `cumulative_pm10_since_rain`. Each has
its own Pearson r, p-value, and regression line.

**What to look for**:

- Positive correlations (upward trends) in all three panels confirm that time
  since rain and accumulated dust predict within-cycle performance decline.
- Compare the r values across panels: the strongest predictor has the steepest
  regression line and highest r. `days_since_last_rain` and
  `cumulative_pm25_since_rain` are typically the strongest, reflecting that
  soiling accumulates over dry days and that finer PM2.5 particles adhere to
  panels more than coarser PM10 (consistent with the soiling literature).
- Points clustering near the origin are from recently-rained days (zero dust
  accumulation, near-zero deviation). The relationship emerges as you move
  right (longer dry stretches, more accumulated dust).

### s2_feature_heatmap.png

**Layout**: Square correlation matrix. Rows and columns are environmental
features, engineered features, pvlib estimates, and target variables.
Colour scale: blue = negative, red = positive; values annotated in each cell.

**What to look for**:

- The rightmost three columns (loss proxy, loss rate, cycle deviation) show
  which features are predictive targets. Look for cells with |r| > 0.2.
- Strong inter-feature correlations (e.g., PM10 and PM2.5 are highly
  correlated) inform feature selection — avoid feeding redundant features
  into ML models.
- `cloud_opacity_mean` correlating strongly with loss proxy confirms it as
  the primary confounder.
- `cycle_deviation_pct` having strong correlations with `days_since_last_rain`,
  `cumulative_pm25_since_rain`, and `cumulative_pm10_since_rain` confirms these
  engineered features capture soiling dynamics. PM2.5 accumulation typically
  shows a stronger correlation than PM10, consistent with finer particles
  adhering more to panel surfaces.

### Partial Correlation Table (in report)

The report's Signal 2 section includes a table of partial correlations —
the correlation between each dust feature and each target **after
mathematically removing** the effect of cloud opacity and temperature.

**What to look for**:

- The `vs cycle deviation` column is the most informative. Features with
  partial r > 0.15 and p < 0.05 are genuinely associated with soiling
  after deconfounding.
- `cumulative_pm25_since_rain` and `days_since_last_rain` showing strong
  partial correlations with cycle deviation (r ~ 0.3-0.35, p < 0.001)
  confirms that cumulative dust exposure predicts performance decline.
- If partial correlations flip sign or become very small after deconfounding,
  the raw correlation was a weather artefact, not a soiling signal.

---

## Signal 3: Rain Recovery

Tests whether significant rainfall visibly resets soiling. If rain does not
cause measurable recovery, the loss proxy may be dominated by non-soiling
effects.

**Challenge**: Post-rain days are cloudy, which contaminates the loss proxy.
The analysis uses multi-day windows and statistical tests rather than
relying on day+1 comparisons.

### s3_rain_event_study.png

**Layout**: Single panel. X-axis is "days relative to rain event" (-5 to +7).
Y-axis is loss proxy (%).

- Blue line: mean loss proxy trajectory across all significant rain events.
- Blue dashed line: median trajectory (more robust to outliers).
- Shaded blue band: 5th-95th percentile spread.
- Grey line: control trajectory from non-rain days (baseline comparison).
- Vertical dotted line at day 0: the rain event.

**What to look for**:

- A dip in loss proxy between day +2 and day +5 (allowing clouds to clear)
  indicates recovery. The mean line should drop below its pre-rain level.
- If the blue line rises after rain instead of falling, post-rain cloudiness
  is inflating the loss proxy. This does not necessarily mean rain does
  not clean — it means the metric is contaminated by weather.
- Compare against the grey control line. If the rain trajectory diverges
  downward from the control, recovery is real even if statistically noisy.

### s3_dryspell_start_end.png

**Layout**: Paired-dot plot. Each connected pair represents a dry spell (>= 3
days). Left dot: loss proxy on the first dry day. Right dot: loss proxy on the
last dry day.

**What to look for**:

- Lines sloping **upward** (left to right) mean soiling accumulated during
  the dry spell — the complement of rain recovery.
- The title shows the Wilcoxon signed-rank p-value testing whether end > start.
  p < 0.05 means soiling accumulation during dry spells is statistically
  significant.
- Even if the rain event study (S3-A) is ambiguous due to cloud contamination,
  this test can confirm soiling by showing accumulation during the dry gaps
  between rain events.

### s3_recovery_vs_precipitation.png

**Layout**: Scatter. X-axis: precipitation amount (mm). Y-axis: loss proxy
change from day -1 to day +3 (percentage points).

- Teal points: moderate rain (5-10 mm).
- Blue points: heavy rain (>= 10 mm).
- Horizontal dashed line at y = 0.

**What to look for**:

- Points below the zero line indicate recovery (loss decreased after rain).
- A downward trend (heavier rain = more recovery) would confirm dose-response.
- If points scatter randomly around zero, rain amount does not predict
  recovery magnitude (the signal is too noisy at this plant).

### s3_rain_event_study_seasonal.png

**Layout**: Two side-by-side panels. Same event-study as S3-A but split by
season (dry on the left, wet on the right).

**What to look for**:

- Rain recovery should be **more visible in the dry season** because dust
  accumulation is higher and there is more soiling to "wash off."
- In the wet season, frequent rain keeps panels relatively clean, so each
  individual rain event has less impact to reveal.
- If the dry-season trajectory shows a clear post-rain dip but the
  wet-season one does not, that is consistent with the soiling hypothesis.

---

## Supporting Analyses

These are not go/no-go tests but provide context for interpreting results
and making modeling decisions.

### s4_univariate_distributions.png

**Layout**: 2x3 grid of histograms on HQ days.

- **Row 1** (primary variables):
  - Left: `t1_performance_loss_pct_proxy`. Look for a large spike at 0% (days
    where output met or exceeded baseline) and a right-skewed tail. Zero-loss
    days are structurally expected (the proxy clips at 0).
  - Centre: `precipitation_total_mm`. Heavy right skew with many low-rain days
    and a few heavy events (up to ~90 mm). Most days have some rain (tropical
    site).
  - Right: `pm10_mean`. Should be roughly symmetric or slightly right-skewed,
    centred around 50-55 ug/m3.
- **Row 2** (soiling-specific indicators):
  - Left: `cycle_deviation_pct`. Distribution of within-cycle performance
    decline. Most values cluster near 0% (start of cycles); a right tail
    shows how far performance degrades before the next reset.
  - Centre: `domain_soiling_daily`. Distribution of the DSPI daily
    accumulation rate. This is a physics-based metric, so its shape reflects
    environmental conditions rather than plant performance.
  - Right: `t1_perf_loss_rate_14d_pct_per_day`. The 14-day rolling rate of
    change in loss proxy. Values near zero mean stable performance; positive
    values mean performance is worsening (active soiling).

### s4_pvlib_vs_observed.png

**Layout**: 2x2 grid comparing two physics-based soiling estimates against the
observed loss proxy.

- **Top row (pvlib Kimber)**:
  - Left: Scatter of pvlib Kimber loss (%) vs observed loss proxy (%).
  - Right: Time-series with observed proxy on the left y-axis and pvlib
    Kimber loss on the right y-axis.
- **Bottom row (Domain Soiling Index / DSPI)**:
  - Left: Scatter of DSPI cumulative value vs observed loss proxy (%).
  - Right: Time-series with observed proxy on the left y-axis and DSPI on
    the right y-axis.

**What to look for**:

- pvlib predicts small losses (~0-8%) while the observed proxy ranges 0-80%.
  The magnitude mismatch is expected because pvlib models pure soiling while
  the proxy is all-cause.
- The DSPI uses a different scale (cumulative environmental pressure units).
  Compare the **shape** of rises and falls, not absolute values.
- Look for **relative pattern agreement** in both rows: do the physics lines
  rise and fall at the same times as the observed proxy, even at different
  scales? That confirms the physics model captures the soiling component.
- Compare the `r` values in the scatter titles to see which physics estimate
  tracks observed loss better. Both are expected to be weak because the
  observed proxy includes non-soiling losses (clouds, equipment).
- pvlib uses a generic deposition model; DSPI is calibrated for this site's
  environmental profile. The comparison reveals whether site-specific tuning
  improves tracking.

### s4_sensor_dirt_check.png

**Layout**: Single time-series. Ratio of `solcast_gti_sum / irradiance_tilted_sum`
over time, with a 30-day rolling mean.

**What to look for**:

- The absolute ratio (~140) is meaningless — it reflects the ThingsBoard
  irradiance unit ambiguity (summed W/m2 readings rather than true W-s/m2).
- The **trend** matters. An upward slope means the satellite is reading
  progressively higher relative to the ground sensor, suggesting the ground
  sensor is getting dirty.
- A flat or downward trend means no detectable sensor drift.
- A negative trend (as observed: -0.32/day) may indicate seasonal variation
  in the ratio rather than sensor cleaning.

### s4_tier_validation.png

**Layout**: Single time-series with T1 loss proxy (purple) and T2 loss proxy
(pink) overlaid.

**What to look for**:

- The two traces should track each other closely. The title shows the median
  tier-loss correlation (0.976), confirming plant-wide soiling.
- Divergences (e.g., T1 rises but T2 stays flat) would indicate block-specific
  issues rather than soiling.
- T2 (B1) may show more noise and gaps due to lower data availability.

### s4_seasonal_boxplots.png

**Layout**: Monthly box plots of loss proxy on HQ days. Amber boxes are dry
months, teal boxes are wet months.

**What to look for**:

- Dry months (Jan-Mar, Jun-Sep) with higher median loss are consistent with
  faster soiling accumulation when rain is less frequent.
- Wet months (Apr-May, Oct-Dec) with lower median loss suggest rain keeps
  panels cleaner.
- Feb-Apr showing the highest medians aligns with the inter-monsoon dry
  period for this tropical site (~8.5 deg N latitude).

### s4_quality_gating.png

**Layout**: Two panels.

- Left: Histogram of `transfer_quality_score`. Should cluster near 100 with a
  tail toward lower scores.
- Right: Bar chart of quality tiers (high / medium / low). A horizontal
  dashed line shows the count of HQ + zero-flag days (the strictest filter).

**What to look for**:

- Confirm that enough days survive the strictest filter for meaningful
  analysis (246 days in the current run).
- If the "high" bar is much smaller than total days, many days have quality
  issues and the pipeline may need stricter cleaning or additional data
  sources.

---

## Domain Soiling Pressure Index (DSPI)

The DSPI is a physics-based daily soiling estimate built entirely from
environmental satellite data (PM2.5, PM10, humidity, dewpoint,
precipitation). **No plant performance data is used**, so there is no
data-leakage concern. It represents what the soiling literature says
*should* be happening to panels given the environmental conditions.

**Formula**:

    daily_rate = (w_pm25 * PM2.5 + w_pm10 * PM10)
                 * humidity_factor * dew_factor * cementation_factor

- **Base deposition**: PM2.5 is weighted higher than PM10 because finer
  particles fill interparticle gaps more completely and resist wind/rain
  removal (Appels et al.; confirmed by our data where cumulative PM2.5
  outperforms cumulative PM10 in predicting cycle deviation).
- **Humidity adhesion factor**: adhesion increases with relative humidity
  due to capillary bridges between particles and glass (Said et al.: ~80%
  adhesion increase from 40% to 80% RH). Factor ranges 1.0 to 2.0.
- **Dew proximity factor**: when the air-dewpoint temperature spread is
  small (< 10 C), dew forms on panel surfaces, promoting dust coagulation
  and cementation. Factor ranges 1.0 to 1.5.
- **Light-rain cementation**: rainfall below 1 mm/day wets dust without
  washing it away, increasing adhesion (Mejia et al.). Rain >= 1 mm triggers
  a cleaning reset.
- **Cumulative index**: the daily rate accumulates over time and resets to
  zero on cleaning rain (>= 1 mm) or known cleaning campaigns.

**Weight calibration**: the five scale parameters (PM2.5 weight, PM10 weight,
humidity scale, dew scale, cementation boost) are calibrated via constrained
optimisation. The objective maximises positive correlation with PM10/PM2.5 and
negative correlation with precipitation, while penalising correlation with
non-soiling factors (cloud opacity, temperature). Domain-knowledge bounds
enforce physically meaningful ranges. No plant performance metrics are used
in the optimisation.

**Important caveats**:

- The DSPI is **not** ground truth for soiling. It is a theoretical estimate
  that correlates with environmental soiling drivers by construction.
- **Tropical humidity paradox**: at this site (8.5 N latitude), humidity is
  always high (78-98%). Between-day humidity variation is dominated by rain
  proximity, not by the micro-physics adhesion effect described in the
  literature. This means the humidity factor adds some noise at daily
  resolution despite being physically correct at the particle level.
- The DSPI should be used as: (a) a visualization tool showing expected
  soiling accumulation patterns, (b) a modeling feature carrying domain
  physics into ML, (c) a qualitative reference for comparison against
  observed performance metrics. It should NOT be used as a correlation
  target (correlating environmental features against it would be circular).

### s5_domain_soiling_index.png

**Layout**: Single time-series with dual y-axes.

- Left y-axis (amber): `domain_soiling_index` — the cumulative DSPI.
- Right y-axis (purple): `cycle_deviation_pct` — the observed within-cycle
  performance deviation (from actual plant energy generation).
- Faint blue vertical lines mark significant rain events (>= 5 mm).
- Shaded orange bands mark cleaning campaigns.

**What to look for**:

- Both traces should show sawtooth-like patterns: gradual accumulation during
  dry periods and sharp drops at rain/cleaning events.
- When the amber line (physics estimate) and the purple line (observed
  performance decline) rise and fall at the same times, that confirms the
  DSPI captures real soiling dynamics.
- Periods where the DSPI rises but cycle deviation does not may indicate
  other factors (e.g., cloud contamination in the performance metric) masking
  the soiling signal.
- The magnitude scales will differ because the DSPI is in arbitrary
  composite units while cycle deviation is in percentage points.

### s5_dspi_correlation_profile.png

**Layout**: Horizontal bar chart. Each bar shows the Pearson r between
`domain_soiling_index` and one environmental or performance feature on HQ
days. Green bars = positive correlation, red bars = negative, grey = near
zero.

**What to look for**:

- **Expected positive correlations**: PM2.5, PM10, cumulative PM features,
  days since rain, humidity x PM10. These indicate the DSPI correctly
  represents soiling pressure from dust exposure.
- **Expected negative correlations**: precipitation (rain cleans), humidity
  (the tropical paradox — see caveat above).
- **Near-zero expected**: cloud opacity, air temperature. These should be
  close to zero if the optimisation successfully decoupled the DSPI from
  non-soiling weather variation. Values > 0.15 in absolute terms suggest
  residual weather contamination.
- **Performance feature correlations**: positive correlation with loss proxy,
  loss rate, and especially cycle deviation would mean the physics-based
  estimate aligns with what is actually observed at the plant — the strongest
  possible validation that the DSPI captures real soiling.

---

## Clear-Sky Soiling Analysis (C-series plots)

### Why this section exists

The soiling metrics in this dataset are heavily contaminated by tropical
weather. The site averages 36% cloud opacity and receives rain on over 40%
of days. Cloud reduces normalised output against the clear-sky baseline,
creating loss proxy spikes that are weather artefacts, not soiling. Equipment
shutdowns (11 zero-output days) add further 100% loss spikes.

The **Clear-Sky Analyzable (CSA)** filter retains only days where weather
contamination is minimal:

- Cloud opacity < 35%
- Precipitation < 1 mm
- Equipment operating (output > 0)
- At least 1 day since last rain (no carry-over cloud)
- High-quality data (HQ tier, no flags)

This keeps ~57 / 235 HQ days (~24%). On these days, the real soiling signal
emerges from under the weather noise.

### c1_clear_sky_loss_timeseries.png

**Layout**: Time-series with faded grey/purple line showing all HQ days, and
green dots (connected) showing only CSA-qualified days.

**What to look for**:

- The grey backdrop shows the full HQ loss proxy — noisy, with many weather-
  driven spikes.
- The green CSA dots should trace smoother, rising trends during dry spells
  and drop sharply at rain events — the soiling sawtooth pattern.
- Dry-spell clusters (e.g., consecutive CSA dots at 5-13 days since rain)
  with progressively increasing loss values confirm real soiling accumulation
  that is masked in the full HQ series.
- Large gaps between CSA dots indicate prolonged cloudy/rainy periods where
  no clean-condition days were available.

### c2_clean_vs_all_correlations.png

**Layout**: Horizontal grouped bar chart. For each feature, two bars show
its Pearson r with loss proxy on All HQ days (purple) vs CSA-only days
(green). Asterisks (*) mark statistically significant correlations (p < 0.05).

**What to look for**:

- **Cumulative PM2.5 since rain** and **Days since rain** should show
  markedly stronger positive correlations on CSA days than on all HQ days.
  This confirms that the soiling signal was being diluted by weather noise.
- **Cloud opacity** should show a weaker (less negative) correlation on CSA
  days, confirming the filter reduced weather contamination.
- **Raw PM10/PM2.5** will likely remain weak or slightly negative even on CSA
  days — this is expected because daily PM concentration measures today's air
  quality, not accumulated panel dust.
- Features with asterisks on the CSA bar but not on the HQ bar are features
  whose soiling signal only becomes visible when weather is controlled for.

### c3_clean_scatter_matrix.png

**Layout**: 2x2 scatter plot matrix showing the two strongest soiling
predictors against two loss metrics, on CSA days only:

- Top-left: `cumulative_pm25_since_rain` vs `loss_proxy`
- Top-right: `days_since_last_rain` vs `loss_proxy`
- Bottom-left: `cumulative_pm25_since_rain` vs `cycle_deviation_pct`
- Bottom-right: `days_since_last_rain` vs `cycle_deviation_pct`

Each panel includes a regression line, Pearson r, p-value, and significance
marker.

**What to look for**:

- Positive slopes with p < 0.05 confirm statistically significant soiling
  relationships: the longer since rain (or the more PM2.5 has accumulated),
  the greater the performance loss.
- Scatter around the regression line indicates how much unexplained variance
  remains — tighter clusters mean stronger predictive power.
- If `cycle_deviation_pct` shows stronger correlations than `loss_proxy`, it
  suggests the within-cycle metric is a better soiling target for modeling.

---

## Reading the Signal Report

`artifacts/eda/eda_signal_report.md` is structured as:

1. **Data Summary**: Row counts, date range, training-ready day count.
2. **Signal 1 section**: Verdict (PASS/WEAK/FAIL), the measured soiling rate
   in %/day, how many dry spells were analysed, and what fraction had positive
   (soiling) slopes.
3. **Signal 2 section**: Raw correlations (with a caveat that they are
   confounded by weather), partial correlations after deconfounding (the real
   test), and within-cycle correlations.
4. **Signal 3 section**: Wilcoxon p-values for the event study and the
   dry-spell accumulation test, recovery-vs-precipitation correlation, and
   event counts.
5. **Supporting Findings**: pvlib comparison, sensor dirt trend, tier
   agreement, seasonal patterns, DSPI, and clear-sky soiling analysis.
6. **Overall Go/No-Go Verdict**: A summary table of all three signal verdicts
   and a recommendation.

### Verdict Thresholds

| Signal 1 | Criteria for PASS |
|---|---|
| Sawtooth visible | In >= 2 of 3 views (time-series, per-inverter, cycle deviation) |
| Soiling rate | Median 0.05-1.0 %/day across dry spells |

| Signal 2 | Criteria for PASS |
|---|---|
| Partial correlation | PM or cumulative PM vs loss > 0.15 after deconfounding |
| OR within-cycle | PM10-rate correlation > 0.2 across cycles |

| Signal 3 | Criteria for PASS |
|---|---|
| Event study | Wilcoxon p < 0.05 for loss decrease at day +2..+5 |
| OR dry-spell test | Wilcoxon p < 0.05 for end > start accumulation |

### Overall Verdict Logic

| Signals passing | Verdict | Meaning |
|---|---|---|
| 3/3 | **Strong go** | All signals confirmed. Proceed to modeling. |
| 2/3 | **Conditional go** | Two signals confirmed. Proceed with caution; note the weak signal. |
| 1/3 or 2+ weak | **Weak go** | Consider additional data sources or features before heavy modeling. |
| 0/3 | **No-go** | Loss proxy may be dominated by equipment/data issues. Re-evaluate. |

### Interpreting the Current Result

The current EDA produced a **CONDITIONAL GO** verdict: Signals 1 and 2 passed,
Signal 3 failed.

**Why Signal 3 failed**: It is not because rain does not clean the panels — it
almost certainly does. The failure is because the metric we are measuring (the
loss proxy) is contaminated by post-rain cloudiness. Rain days and the days
immediately following are typically cloudy. Cloudy days depress normalised
output against the clear-sky baseline, making it look like performance got
*worse* after rain, even though the panels are actually cleaner. The recovery
signal is real but buried under weather noise.

**What this means for next steps**: The modeling phase should be able to
separate weather effects from soiling effects. ML models that take cloud
opacity, temperature, and irradiance as inputs can learn to "see through" the
weather contamination that the simple statistical tests in the EDA could not
resolve. The fact that Signals 1 and 2 are strong provides sufficient
confidence that soiling signal exists in the data.

---

## Data Quality Diagnostics (DQ)

### Why this section exists

The Signal plots (S1-S3) answer "is there a soiling signal?" The Data
Quality (DQ) plots answer a different question: "can we trust the underlying
data?" They cross-validate data sources, verify unit consistency, compare
old and new telemetry pipelines, and confirm that derived features behave as
expected before any modelling begins.

DQ plots are numbered DQ1 through DQ6. Some were split into separate
time-series and non-time-series files to avoid compressing year-long x-axes
into unreadable panels.

### dq1_irradiance_vs_generation_timeseries.png

**Layout**: Single full-width panel with dual y-axes.

- Left y-axis (teal): on-site irradiance sensor sum (10 AM-2 PM window).
- Right y-axis (amber): T1 inverter generation in kWh (same window).
- Faint blue vertical lines mark rain events; orange bands mark cleaning
  campaigns.

**What to look for**:

- Both traces should broadly co-vary: sunny days produce both high
  irradiance and high generation. Systematic divergence (e.g., generation
  dropping while irradiance stays high) may indicate soiling, equipment
  faults, or sensor problems.
- Sharp generation drops on otherwise clear days are red flags for inverter
  trips or data dropouts.
- Look for the traces to reconverge after rain/cleaning events, which
  supports the soiling hypothesis.
- Because this uses the on-site sensor sum (which has known unit ambiguity),
  the absolute irradiance values are less important than the shape and
  co-movement with generation.

### dq1_irradiance_vs_generation.png

**Layout**: Three panels side by side.

- Panel 1 (left): Scatter of on-site irradiance vs T1 generation, coloured
  by month. A Pearson r value is shown in the title.
- Panel 2 (centre): Scatter of Solcast peak-hour GTI (kWh/m^2, 10-14h) vs
  T1 generation, coloured by month. A small annotation shows the correlation
  against full-plant generation if available.
- Panel 3 (right): Monthly boxplot of normalised output
  (energy/irradiance). The CV of monthly medians is shown as a text
  annotation.

**What to look for**:

- **Panel 1**: Low r is expected for the T1 subset (3 inverters) because
  individual inverter variability (clipping, faults) breaks the simple
  linear relationship. A cloud of points with no clear trend is typical.
- **Panel 2**: A higher r here (especially for full-plant generation)
  confirms that Solcast satellite irradiance is a reliable reference. If
  the full-plant annotation shows r > 0.4, the data pipeline is trustworthy.
- **Panel 3**: Monthly medians should be roughly stable across the year if
  the normalisation is working. A high CV (> 20%) suggests seasonal
  confounders or sensor issues that the normalisation does not remove. Dry
  months (Jan-Mar, Jun-Sep) may show slightly lower medians from soiling.

### dq2_daily_gen_validation_timeseries.png

**Layout**: Two stacked time-series panels (full width).

- Top panel: old-source generation (active power integral, kWh, purple)
  overlaid with new-source generation (daily_generated_electricity, kWh,
  teal). A vertical dashed line marks the start of new-source data.
- Bottom panel: plant average irradiance (avg_solar_radiation, teal)
  overlaid with Solcast peak GTI mean (purple). Same new-source start
  annotation.

**What to look for**:

- **Top panel**: Where both traces overlap (post new-source start), they
  should track each other in shape even if absolute magnitudes differ. The
  old source covers 10 AM-2 PM only; the new source is full-day. Large
  divergences on specific days suggest data quality issues in one source.
- **Bottom panel**: The plant irradiance and Solcast lines should correlate
  well. If the plant sensor consistently reads higher or lower than Solcast,
  that is a scaling offset — acceptable as long as the trends match. If
  they diverge for a stretch, the on-site sensor may have been obstructed
  or miscalibrated.
- The grey region before the vertical dashed line shows where only old data
  exists. This is expected — the new telemetry was not available before
  Apr 2025.

### dq2_daily_gen_validation.png

**Layout**: Two scatter panels side by side.

- Left panel: Old generation (active power integral, kWh) vs new generation
  (daily_generated_electricity, kWh). A 1:1 reference line is shown.
- Right panel: Solcast peak GTI mean (W/m^2) vs plant average irradiance
  (W/m^2).

**What to look for**:

- **Left panel**: Points should cluster around the 1:1 line if both sources
  measure the same underlying quantity. Systematic offset (points
  consistently above or below the line) indicates a scaling difference
  between peak-hour and full-day aggregation. Extreme outliers suggest
  data quality issues on specific days.
- **Right panel**: A tight cluster along a diagonal confirms irradiance
  consistency. r > 0.6 is a positive sign. Scatter that widens at high
  irradiance values may indicate cloud variability on those days.
- Low old-vs-new generation r is expected when the two sources use different
  time windows (peak-hour vs full-day).

### dq3_gen_irr_ratio_timeseries.png

**Layout**: Two stacked time-series panels (full width).

- Top panel: Generation/irradiance ratio (daily as faint dots, 7-day median
  as a bold dark blue line), overlaid with scaled generation (orange) and
  scaled irradiance (yellow) for context. A black dashed horizontal line
  marks the overall median ratio. Rain and cleaning overlays are shown.
  A vertical dashed line marks new-source data start.
- Bottom panel: Smoothed gen/irr ratio (dark blue) overlaid with inverted
  loss proxy (purple, right y-axis). The Pearson r between them is shown
  in the title.

**What to look for**:

- **Top panel**: The 7-day median line should show a sawtooth pattern —
  gradual decline during dry spells (soiling accumulation) with sharp
  recoveries at rain events or cleaning campaigns. If the line is flat or
  random, the ratio may be too noisy to isolate soiling.
- **Top panel**: The scaled generation and irradiance traces provide
  context. If generation drops but irradiance stays high, that is genuine
  performance loss. If both drop together, that is a cloudy period.
- **Bottom panel**: The smoothed ratio and inverted loss proxy should
  broadly co-move. Positive r confirms agreement between the two
  independent measures of performance. Weak r is expected given the
  different time windows and the Jan-Mar data gap.

### dq3_gen_irr_ratio.png

**Layout**: Single boxplot panel.

- Monthly boxplots of the generation/irradiance ratio. Each box shows the
  distribution for one calendar month.

**What to look for**:

- Months in the dry season (Jan-Mar, Jun-Sep) should show slightly lower
  medians and tighter distributions if soiling accumulates during those
  periods.
- Wet-season months (Apr-May, Oct-Dec) should show higher medians if rain
  cleaning is effective.
- Large boxes with many outliers suggest high day-to-day variability, which
  is partly weather-driven.
- If all months have similar medians, soiling may not produce enough
  seasonal variation to distinguish from noise — but intra-month variation
  (the sawtooth) can still be present.

### dq4_power_at_ref_irradiance.png

**Layout**: Two panels side by side.

- Left panel: Time-series of active power at the dataset's median
  irradiance level. Rain and cleaning overlays are shown. This feature
  controls for irradiance variation — on days with the same irradiance
  level, power differences come from soiling, temperature, or equipment.
- Right panel: Horizontal bar chart of Pearson r between
  `power_at_ref_irradiance_w` and various soiling features on HQ days.

**What to look for**:

- **Left panel**: A sawtooth pattern (gradual decline between rain events,
  recovery after rain/cleaning) confirms soiling drives the power decrease.
  If the line is flat or shows random fluctuation, the feature may not
  isolate soiling well.
- **Right panel**: Strong negative correlation with `t1_performance_loss_pct_proxy`
  is expected and confirms the feature captures degradation. Negative
  correlations with `cumulative_pm25_since_rain` and `days_since_last_rain`
  indicate that soiling (not equipment) drives the decline. Positive
  correlations with humidity-related features may reflect the tropical
  paradox (humidity both promotes dust adhesion and cleans via dew).
- Caveat: This feature uses active power from the T1 inverter subset, which
  is subject to inverter-level variability (trips, clipping). Per-inverter
  faults can create artificial drops unrelated to soiling.

### dq5_old_vs_new_timeseries.png

**Layout**: Two stacked time-series panels (full width).

- Top panel: Performance loss proxy (%) from the old source (teal, T1
  active-power-based) overlaid with the new source (amber,
  daily_gen/avg_irr-based). A vertical dashed line marks new-source data
  start.
- Bottom panel: Cycle deviation (%) from the old source (teal) overlaid
  with the new source (amber). Same new-source start annotation.

**What to look for**:

- **Top panel**: In the overlap region (post new-source start), the two
  traces should show similar trends — both rising during dry spells and
  dropping after rain. The magnitudes may differ because the two pipelines
  use different time windows and baselines. Persistent divergence suggests
  one source has a calibration or data quality issue.
- **Bottom panel**: Cycle deviation should also broadly agree in the
  overlap region. If the new source shows larger deviations, the full-day
  aggregation may be more sensitive to soiling. If it shows smaller
  deviations, the peak-hour window may better isolate soiling.
- The old source covers the entire date range; the new source starts in
  Apr 2025. The Jan-Mar gap misses the peak dry season, so the new source
  cannot be evaluated under the harshest soiling conditions.

### dq5_old_vs_new_comparison.png

**Layout**: Two panels side by side.

- Left panel: Scatter of old loss proxy (%) vs new loss proxy (%). A 1:1
  reference line is shown. The Pearson r and sample size are in the title.
- Right panel: Grouped bar chart comparing soiling feature correlations
  against the old vs new loss proxy. Features: DSPI, cumulative PM2.5,
  cumulative PM10, days dry, humidity x PM10.

**What to look for**:

- **Left panel**: Points near the 1:1 line mean both pipelines agree on
  the severity of each day's loss. Scatter away from the line means the
  two pipelines disagree — expected given different time windows. A low
  or negative r is not alarming; it reflects the structural differences
  between the two pipelines rather than data quality problems.
- **Right panel**: Compare the bar heights for each feature. If the new
  source shows stronger correlations (taller bars) with environmental
  soiling drivers, it may be a better target for modelling. If the old
  source is consistently stronger, the peak-hour window may isolate
  soiling better. Mixed results (some features stronger in old, some in
  new) are common and suggest complementary value from both pipelines.
- Caveat: The new source covers fewer months (missing Jan-Mar), so its
  correlations may be biased toward wetter months with less soiling.

### dq6_performance_index.png

**Layout**: Three panels using a grid layout.

- Top panel (spans full width): Time-series of the new-source performance
  index (gen_irr_ratio / rolling_clean_baseline, clipped at 1.5). Rain
  and cleaning overlays are shown. A horizontal dashed line at 1.0 marks
  clean-panel performance. A vertical dashed line marks new-source data
  start.
- Bottom-left panel: Histogram of the performance index distribution, with
  vertical lines for the median and the 0.9 threshold.
- Bottom-right panel: Scatter plots of the performance index vs each of
  five soiling features (DSPI, cumulative PM2.5/PM10, days dry,
  humidity x PM10), with Pearson r values annotated.

**What to look for**:

- **Top panel**: The index should hover near 1.0 after rain/cleaning events
  and decline during dry spells. Values consistently below 0.8 indicate
  heavy persistent losses. If the line is flat near a low value, the
  rolling baseline may not be resetting properly, or equipment issues
  dominate.
- **Bottom-left panel**: The distribution should peak near 0.6-0.9 for a
  soiled plant with occasional cleaning. A bimodal distribution (one peak
  near 1.0, another lower) would suggest distinct clean/dirty states — a
  strong soiling signal.
- **Bottom-right panel**: Negative correlations with cumulative PM and
  days-since-rain features confirm that soiling pressure reduces the
  performance index. Weak correlations (|r| < 0.1) may indicate the
  index is dominated by non-soiling factors or that the data gap
  removes the strongest soiling period from the analysis.
- If the median is well below 1.0 (e.g., 0.6-0.7), this indicates
  substantial and persistent performance loss. If the mean is much lower
  than the median, extreme outlier days (equipment faults, severe soiling)
  are pulling the distribution down.
