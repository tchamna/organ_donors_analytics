# Transfer This Pipeline To Microsoft Fabric

Use this to move the current local pipeline (`pipelines/main_pipeline.py`) into Fabric and run it there.

## 1. Put This Repo In Git

Push this project to GitHub or Azure DevOps repo.

## 2. Connect Fabric Workspace To Git

In Fabric workspace:

1. `Workspace settings` -> `Git integration`
2. Connect to your repo + branch
3. Sync workspace from Git

Reference:
- https://learn.microsoft.com/en-us/fabric/cicd/git-integration/git-get-started
- https://learn.microsoft.com/en-us/fabric/cicd/git-integration/git-integration-process

## 3. Create Runtime Items In Fabric

Create these items in the same workspace:

1. Lakehouse: `OrganDonorLakehouse`
2. Notebook: `run_organ_pipeline`
3. Data Pipeline: `orchestrate_organ_pipeline` (optional, for schedule)

## 4. Notebook Code (Run Local Python Pipeline In Fabric)

In `run_organ_pipeline` notebook, use:

```python
import os, sys, subprocess

# Update this path to where your synced repo/files live in Fabric runtime.
# If using Lakehouse Files upload, use /lakehouse/default/Files/<folder>.
PROJECT_ROOT = "/lakehouse/default/Files/organ_donors_analytics"

os.chdir(PROJECT_ROOT)
print("Working dir:", os.getcwd())

# Install deps if your environment doesn't already have them
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy", "matplotlib", "loguru"])

# Run end-to-end medallion pipeline
subprocess.check_call([
    sys.executable, "pipelines/main_pipeline.py",
    "--generator", "negbin",
    "--days", "90",
    "--avg-referrals-per-day", "12",
    "--seed", "7",
    "--no-plot",
])

print("Pipeline completed")
```

## 5. Verify Output Paths

After run, validate these folders:

- `sample_data/l1_bronze`
- `sample_data/l2_silver`
- `sample_data/quarantine`
- `sample_data/l3_gold`

## 6. Load Gold CSVs Into Lakehouse Tables

Load these to tables for reporting:

- `gold_daily_metrics.csv`
- `gold_hospital_metrics.csv`
- `gold_blood_type_metrics.csv`
- `gold_organ_type_metrics.csv`
- `gold_quality_metrics.csv`
- `gold_rejection_reason_metrics.csv`

## 7. Orchestrate With Fabric Data Pipeline (Recommended)

In Data Pipeline:

1. Add `Notebook activity`
2. Select `run_organ_pipeline`
3. Save + Run
4. Add Schedule trigger (daily/hourly as needed)

Reference:
- https://learn.microsoft.com/en-ca/fabric/data-factory/notebook-activity

## 8. Stakeholder Demo Flow

1. Run notebook/pipeline
2. Refresh semantic model/report
3. Present pages: Executive, Hospital, Blood/Organ Mix, Data Quality

