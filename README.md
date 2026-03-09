# Organ Donors Analytics (Medallion Pipeline)

Python-based medallion pipeline for organ donor referral analytics:

- `l1_bronze`: synthetic raw data generation
- `l2_silver`: validation, cleaning, quarantine
- `l3_gold`: business-ready analytics tables

## Repository Structure

- `pipelines/main_pipeline.py` orchestrates Bronze -> Silver -> Gold
- `pipelines/l1_bronze/` synthetic data generators
- `pipelines/l2_silver/` silver validation/cleansing logic
- `pipelines/l3_gold/` gold metric tables
- `docs/fabric_demo_playbook.md` stakeholder demo guide
- `docs/fabric_transfer_checklist.md` move/run in Microsoft Fabric

## Prerequisites

- Python 3.10+

## Install

```bash
pip install -r requirements.txt
```

## Run End-to-End Pipeline

```bash
python pipelines/main_pipeline.py --generator negbin --days 90 --avg-referrals-per-day 12 --seed 7 --no-plot
```

## Common Run Options

```bash
# Poisson generator
python pipelines/main_pipeline.py --generator poisson --days 30 --avg-referrals-per-day 10 --seed 7 --no-plot

# Bronze only
python pipelines/main_pipeline.py --skip-silver --skip-gold --no-plot

# Run pipeline, then launch Dash dashboard
python scripts/fabric_run_pipeline.py --install-deps --launch-dashboard --dashboard-port 8050
```

## Output Folders

- `sample_data/l1_bronze/`
- `sample_data/l2_silver/`
- `sample_data/quarantine/`
- `sample_data/l3_gold/`

## Gold Tables

- `gold_daily_metrics.csv`
- `gold_hospital_metrics.csv`
- `gold_blood_type_metrics.csv`
- `gold_organ_type_metrics.csv`
- `gold_quality_metrics.csv`
- `gold_rejection_reason_metrics.csv`
- `gold_referral_funnel_metrics.csv` (referral -> donor funnel)
- `gold_hospital_conversion_metrics.csv` (hospital referral conversion)
- `gold_missed_opportunity_metrics.csv` (expected vs observed donor gaps)
- `gold_referral_propensity_metrics.csv` (referral donor probability scoring)
- `gold_discard_delay_metrics.csv` (discard rate by cold ischemia bucket)
- `gold_donor_volume_forecast.csv` (14-day operational referral/donor forecast)

## Microsoft Fabric

Use:

- `docs/fabric_transfer_checklist.md` to move and run this pipeline in Fabric
- `docs/fabric_demo_playbook.md` to build stakeholder report pages
- `docs/ec2_deploy.md` to deploy the Dash dashboard on EC2 for stakeholder access

## Dash Dashboard

Run the dashboards from Gold/Silver outputs:

```bash
python scripts/dash_dashboard.py
```

Optional args:

```bash
python scripts/dash_dashboard.py --project-root . --host 0.0.0.0 --port 8050 --debug
```

Dashboards included:

- Daily Operations
- Hospital Benchmarking
- Organ Type Mix
- Blood Type
- Rejection Root Cause
- Data Quality
- Case Drilldown
