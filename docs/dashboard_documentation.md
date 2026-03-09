# Organ Donor Analytics Documentation

End to end pipeline build by **Shck Tchamna**, from syntetic data generation to dashboard.

## 1. Overview

This project builds an end-to-end analytics workflow for organ donor operations using a medallion architecture:

- `Bronze (l1_bronze)`: synthetic raw data generation
- `Silver (l2_silver)`: validation, cleansing, quarantine handling
- `Gold (l3_gold)`: business-ready aggregate metrics for reporting and dashboards

## 2. Synthetic Data Generation (Bronze)

Synthetic data is generated from the Bronze layer scripts:

- `pipelines/l1_bronze/generate_synthetic_data.py` (Poisson)
- `pipelines/l1_bronze/generate_synthetic_data_negative_binomial.py` (Negative Binomial)

### Key generation controls

- `--generator`: `poisson` or `negbin`
- `--days`: number of simulated days
- `--avg-referrals-per-day`: expected daily referrals
- `--seed`: deterministic random seed for reproducibility
- `--overdisp-k`: dispersion parameter for negative binomial variability

### Bronze outputs

- `sample_data/l1_bronze/referrals.csv`
- `sample_data/l1_bronze/placement_outcomes.csv`

## 3. Data Validation and Curation (Silver)

Silver layer processes raw records and enforces quality rules:

- Type normalization and null checks
- Business-rule validation
- Referential integrity checks between referrals and outcomes
- Invalid records routed to quarantine for auditability

### Silver outputs

- `sample_data/l2_silver/silver_referrals.csv`
- `sample_data/l2_silver/silver_placement_outcomes.csv`
- `sample_data/l2_silver/silver_quality_report.csv`

### Quarantine outputs

- `sample_data/quarantine/quarantine_referrals.csv`
- `sample_data/quarantine/quarantine_placement_outcomes.csv`

## 4. Business Metrics (Gold)

Gold creates analytics-ready tables for reporting:

- `gold_daily_metrics.csv`
- `gold_hospital_metrics.csv`
- `gold_blood_type_metrics.csv`
- `gold_organ_type_metrics.csv`
- `gold_quality_metrics.csv`
- `gold_rejection_reason_metrics.csv` (when rejection reasons are present)

## 5. Dashboard Layer (Dash)

The dashboard app is in `scripts/dash_dashboard.py` and includes:

- Daily Operations
- Hospital Benchmarking
- Organ Type Mix
- Blood Type
- Rejection Root Cause
- Data Quality
- Case Drilldown
- Documentation (this page)

## 6. Pipeline Flow

Execution order:

1. Bronze synthetic generation
2. Silver validation and quarantine
3. Gold metric aggregation
4. Dash visualization

Main orchestrator:

- `pipelines/main_pipeline.py`

Fabric launcher:

- `scripts/fabric_run_pipeline.py`

## 7. How To Run

### Run pipeline only

```bash
python pipelines/main_pipeline.py --generator negbin --days 90 --avg-referrals-per-day 12 --seed 7 --no-plot
```

### Run dashboard only

```bash
python scripts/dash_dashboard.py --project-root . --port 8050
```

### Run pipeline then dashboard

```bash
python scripts/fabric_run_pipeline.py --install-deps --launch-dashboard --dashboard-port 8050
```

## 8. Design Notes

- Dashboards should use `Gold` for KPIs and executive reporting.
- Use `Silver` for case-level drilldown and root-cause investigation.
- Quarantine tables provide transparency for data-quality exclusions.

## 9. Predictive Algorithms Used

This implementation uses practical operational analytics models (not deep-learning or external training pipelines yet).

### 9.1 Referral Propensity Scoring (Heuristic + Regression)

Output table: `gold_referral_propensity_metrics.csv`

Each referral now has three probability fields:

- `heuristic_donor_probability`
- `regression_donor_probability`
- `predicted_donor_probability` (blended operational score)

Heuristic baseline:

`heuristic_donor_probability = 0.50 * triage_score + 0.40 * hospital_historical_conversion_rate + 0.10 * hour_factor`

Where:

- `hospital_historical_conversion_rate = donor_referrals / referrals_count` (hospital-level history)
- `hour_factor = 1.0` if referral hour is between 07:00 and 22:00, else `0.9`
- score is clipped to `[0.0, 0.99]`

Regression model:

- Model: `LogisticRegression` (scikit-learn)
- Numeric features: `triage_score`, `referral_hour`, `hospital_historical_conversion_rate`
- Categorical features (one-hot): `hospital_id`, `blood_type`
- Missing handling: median/mode imputation in pipeline

Blended operational score:

`predicted_donor_probability = 0.35 * heuristic_donor_probability + 0.65 * regression_donor_probability`

Fallback behavior:

- If class balance/data volume is insufficient for logistic training, `regression_donor_probability` falls back to heuristic.

Priority bands:

- `low`: probability <= 0.40
- `medium`: 0.40 < probability <= 0.70
- `high`: probability > 0.70

### 9.2 Donor Volume Forecast

Output table: `gold_donor_volume_forecast.csv`

Forecast horizon: next 14 days.

Model:

- baseline = rolling mean of last 14 days
- trend = linear slope from `numpy.polyfit`
- forecast(day_i) = baseline + slope * i
- negatives are clipped to 0

Applied to:

- `predicted_referrals_count`
- `predicted_donor_referrals`

### 9.3 Missed Opportunity Analysis

Output table: `gold_missed_opportunity_metrics.csv`

Hospital expected donors are estimated from global baseline conversion:

- `expected_donor_referrals = hospital_referrals_count * global_conversion_rate`
- `donor_gap_vs_baseline = expected_donor_referrals - donor_referrals`
- `underperforming_flag = donor_gap_vs_baseline > 0`

### 9.4 Discard Delay Risk Analysis

Output table: `gold_discard_delay_metrics.csv`

Cold ischemia delay buckets:

- `0-4h`, `4-6h`, `6-8h`, `8h+`

For each bucket:

- organs total
- organs discarded
- average cold ischemia
- discard rate (`organs_discarded / organs_total`)

### 9.5 Current Scope

These predictive components are intentionally lightweight and transparent for operational use and demos. They can be upgraded to trained ML models (e.g., calibrated logistic regression, gradient boosting, and probabilistic time-series) once richer features are added.
