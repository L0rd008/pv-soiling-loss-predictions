# EDA Output Interpretation Guide

How to read each plot and section of the soiling signal report produced by
`scripts/eda_soiling_signals.py`.

Generated outputs live in `artifacts/eda/`. The report is
`eda_signal_report.md`; plots are in `artifacts/eda/plots/`.

---

## Signal 1: Sawtooth Detection

The sawtooth is the fingerprint of soiling: gradual performance decline as dust
accumulates, followed by a sudden recovery when rain washes the panels or a
cleaning crew intervenes. Temperature drift, equipment degradation, and sensor
noise do not produce this shape.

### s1_loss_proxy_timeseries.png

**Layout**: Two vertically-stacked panels sharing the same time axis.

- **Top panel**: `t1_performance_loss_pct_proxy` (Tier-1 loss proxy) plotted
  over the full date range. Higher values mean worse performance relative to
  the rolling clean baseline.
  - Faint blue vertical lines mark days with significant rain (>= 5 mm).
  - Shaded orange bands mark the three known cleaning campaigns (Sep/Oct/Nov
    2025, 20th-30th).
- **Bottom panel**: Daily precipitation (mm) as a blue bar chart.

**What to look for**:

- Gradual upward slopes between rain events (soiling accumulation).
- Sudden downward steps at rain lines or within cleaning bands (recovery).
- Multi-week ascending runs during dry spells, especially Feb-Apr.

**Caveat**: This plot shows all data, not just high-quality days. Cloudy periods
depress normalised output against the clear-sky baseline, so loss proxy spikes
on overcast days are weather artefacts, not soiling. Focus on trends during
clear, dry stretches.

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

### s2_cumulative_pm10_vs_deviation.png

**Layout**: Single scatter. `cumulative_pm10_since_rain` (x-axis) vs
`cycle_deviation_pct` (y-axis) on HQ days, with a fitted regression line.

**What to look for**:

- A positive correlation (upward trend) means accumulated dust since the last
  rain predicts within-cycle performance decline — the most direct
  dust-to-soiling signal.
- The annotation gives Pearson r and p-value. r > 0.15 with p < 0.05 is a
  meaningful signal.
- Points clustering near origin are from recently-rained days (reset to zero
  dust, near-zero deviation). The relationship shows as you move right (more
  accumulated dust).

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
- `cycle_deviation_pct` having strong correlations with `days_since_last_rain`
  and `cumulative_pm10_since_rain` confirms these engineered features capture
  soiling dynamics.

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

**Layout**: Three side-by-side histograms on HQ days.

- Left: `t1_performance_loss_pct_proxy`. Look for a large spike at 0% (days
  where output met or exceeded baseline) and a right-skewed tail. The 117
  zero-loss days are structurally expected (the proxy clips at 0).
- Centre: `precipitation_total_mm`. Heavy right skew with many low-rain days
  and a few heavy events (up to ~90 mm). Most days have some rain (this is a
  tropical site).
- Right: `pm10_mean`. Should be roughly symmetric or slightly right-skewed,
  centred around 50-55 ug/m3.

### s4_pvlib_vs_observed.png

**Layout**: Two panels.

- Left: Scatter of pvlib Kimber loss (%) vs observed loss proxy (%).
- Right: Time-series with observed proxy on the left y-axis and pvlib
  Kimber loss on the right y-axis.

**What to look for**:

- pvlib predicts small losses (~0-8%) while the observed proxy ranges 0-80%.
  The magnitude mismatch is expected because pvlib models pure soiling while
  the proxy is all-cause.
- Look for **relative pattern agreement**: do both lines rise and fall at the
  same times, even if at different scales? That confirms pvlib captures the
  soiling component.
- Weak correlation (r ~ -0.14) is expected. pvlib will be more useful as a
  feature inside an ML model than as a standalone predictor.

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

## Reading the Signal Report

`artifacts/eda/eda_signal_report.md` is structured as:

1. **Data Summary**: Row counts, date range, training-ready day count.
2. **Signal 1-3**: Each has a verdict (PASS/WEAK/FAIL), a prose summary, and
   key quantitative metrics. Refer to the corresponding plots for visual
   confirmation.
3. **Supporting Findings**: pvlib comparison, sensor dirt trend, tier
   agreement, and seasonal patterns.
4. **Overall Go/No-Go Verdict**: Aggregates the three signal verdicts.

The verdict thresholds:

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
