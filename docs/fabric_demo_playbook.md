# Microsoft Fabric Demo Playbook

This playbook shows how to demo your organ donor analytics pipeline in Microsoft Fabric using the CSV outputs from this repo.

## 1. Files To Use

Use these generated files:

- `sample_data/l3_gold/gold_daily_metrics.csv`
- `sample_data/l3_gold/gold_hospital_metrics.csv`
- `sample_data/l3_gold/gold_blood_type_metrics.csv`
- `sample_data/l3_gold/gold_organ_type_metrics.csv`
- `sample_data/l3_gold/gold_quality_metrics.csv`
- `sample_data/l2_silver/silver_referrals.csv` (detail / drillthrough)
- `sample_data/l2_silver/silver_placement_outcomes.csv` (detail / drillthrough)
- `sample_data/quarantine/quarantine_referrals.csv` (data quality evidence)
- `sample_data/quarantine/quarantine_placement_outcomes.csv` (data quality evidence)

## 2. Create Fabric Assets

1. In Fabric, create a Workspace (for example: `OrganDonor-Demo`).
2. Create a Lakehouse in that workspace.
3. Upload the CSV files into Lakehouse `Files` (or create Delta tables if preferred).
4. Create a Semantic Model from the uploaded tables.
5. Create a Power BI report in the same workspace.

## 3. Model Setup (Important)

Use these primary keys / dimensions:

- `Date`: from `gold_daily_metrics.referral_date`
- `Hospital`: `hospital_id`
- `Blood Type`: `blood_type`
- `Organ Type`: `organ_type`

Suggested relationships:

- `silver_referrals.referral_id` 1 -> many `silver_placement_outcomes.referral_id`
- Keep gold tables mostly independent for KPI pages (no complex joins required).

## 4. Recommended Demo Pages

## Page A: Executive KPI Overview

Cards:
- Total referrals (`sum referrals_count` from daily table)
- Total organs (`sum organs_total`)
- Placement rate (`sum organs_placed / sum organs_total`)
- Outcomes rejection rate (`gold_quality_metrics`)

Visuals:
- Line chart: `referral_date` vs `referrals_count`
- Line chart: `referral_date` vs `placement_rate`
- Clustered column: `organs_placed` vs `organs_discarded` by date

## Page B: Hospital Performance

Visuals:
- Bar chart: `hospital_id` vs `placement_rate`
- Bar chart: `hospital_id` vs `avg_cold_ischemia_minutes`
- Table: `hospital_id`, `organs_total`, `organs_placed`, `organs_discarded`, `placement_rate`

## Page C: Clinical Mix (Blood + Organ)

Visuals:
- Bar chart: `blood_type` vs `placement_rate`
- Bar chart: `organ_type` vs `p90_cold_ischemia_minutes`
- Table: blood/organ metrics with conditional formatting on `placement_rate`

## Page D: Data Quality and Trust

Visuals:
- Cards from `gold_quality_metrics`:
  - `referrals_rejection_rate`
  - `outcomes_rejection_rate`
  - `outcomes_missing_fk_rate`
- Stacked bar: `rejection_reason` counts from quarantine outcomes
- Table sample: quarantined rows including `referral_triage_score` and `rejection_reason`

This page explains why invalid records are excluded and why stakeholder KPIs are trustworthy.

## 5. Demo Narrative (5-7 Minutes)

1. Start with KPI trend: "How many referrals and placements over time?"
2. Show hospital variation: "Where are performance gaps?"
3. Show blood/organ mix: "What cohorts are harder to place?"
4. Show data quality page: "What got excluded and why?"
5. Close with action items:
   - Reduce cold ischemia tail (p90)
   - Focus low-performing hospitals
   - Monitor data-quality rejection trends

## 6. Optional DAX Measures

If you want model-level measures instead of precomputed columns:

- `Placement Rate = DIVIDE(SUM([organs_placed]), SUM([organs_total]))`
- `Discard Rate = DIVIDE(SUM([organs_discarded]), SUM([organs_total]))`

## 7. Refresh Workflow

For each demo refresh:

1. Run local pipeline to regenerate CSVs.
2. Replace/refresh files in Fabric Lakehouse.
3. Refresh semantic model.
4. Open report and validate KPI cards.

