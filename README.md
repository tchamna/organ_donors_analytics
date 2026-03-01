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

## Microsoft Fabric

Use:

- `docs/fabric_transfer_checklist.md` to move and run this pipeline in Fabric
- `docs/fabric_demo_playbook.md` to build stakeholder report pages

