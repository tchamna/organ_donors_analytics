import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR = PROJECT_ROOT / "sample_data" / "l2_silver"
QUARANTINE_DIR = PROJECT_ROOT / "sample_data" / "quarantine"
GOLD_DIR = PROJECT_ROOT / "sample_data" / "l3_gold"


def build_gold():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    referrals = pd.read_csv(SILVER_DIR / "silver_referrals.csv")
    outcomes = pd.read_csv(SILVER_DIR / "silver_placement_outcomes.csv")
    quarantine_referrals = pd.read_csv(QUARANTINE_DIR / "quarantine_referrals.csv")
    quarantine_outcomes = pd.read_csv(QUARANTINE_DIR / "quarantine_placement_outcomes.csv")

    referrals["referral_ts"] = pd.to_datetime(referrals["referral_ts"], errors="coerce")
    referrals["referral_date"] = referrals["referral_ts"].dt.date
    if "referral_date" in outcomes.columns:
        outcomes["referral_date"] = pd.to_datetime(outcomes["referral_date"], errors="coerce").dt.date
    elif "referral_ts" in outcomes.columns:
        outcomes["referral_date"] = pd.to_datetime(outcomes["referral_ts"], errors="coerce").dt.date
    else:
        outcomes = outcomes.merge(
            referrals[["referral_id", "referral_date"]], on="referral_id", how="left"
        )

    # Daily referral-level metrics
    daily_referrals = (
        referrals.groupby("referral_date", as_index=False)
        .agg(referrals_count=("referral_id", "nunique"), avg_triage_score=("triage_score", "mean"))
    )

    # Daily organ outcome metrics
    merged = outcomes.merge(
        referrals[["referral_id", "hospital_id", "blood_type"]], on="referral_id", how="left"
    )

    daily_outcomes = (
        merged.groupby("referral_date", as_index=False)
        .agg(
            organs_total=("organ_type", "count"),
            organs_placed=("placed_flag", "sum"),
            organs_discarded=("discard_flag", "sum"),
            avg_cold_ischemia_minutes=("cold_ischemia_minutes", "mean"),
        )
    )
    daily_outcomes["placement_rate"] = (
        daily_outcomes["organs_placed"] / daily_outcomes["organs_total"]
    ).fillna(0.0)

    gold_daily = daily_referrals.merge(daily_outcomes, on="referral_date", how="outer").sort_values(
        "referral_date"
    )

    hospital_gold = (
        merged.groupby("hospital_id", as_index=False)
        .agg(
            organs_total=("organ_type", "count"),
            organs_placed=("placed_flag", "sum"),
            organs_discarded=("discard_flag", "sum"),
            avg_cold_ischemia_minutes=("cold_ischemia_minutes", "mean"),
        )
        .sort_values("hospital_id")
    )
    hospital_gold["placement_rate"] = (
        hospital_gold["organs_placed"] / hospital_gold["organs_total"]
    ).fillna(0.0)

    blood_referrals = (
        referrals.groupby("blood_type", as_index=False)
        .agg(referrals_count=("referral_id", "nunique"), avg_triage_score=("triage_score", "mean"))
    )
    blood_outcomes = (
        merged.groupby("blood_type", as_index=False)
        .agg(
            organs_total=("organ_type", "count"),
            organs_placed=("placed_flag", "sum"),
            organs_discarded=("discard_flag", "sum"),
            avg_cold_ischemia_minutes=("cold_ischemia_minutes", "mean"),
        )
    )
    blood_type_gold = blood_referrals.merge(blood_outcomes, on="blood_type", how="outer").sort_values(
        "blood_type"
    )
    blood_type_gold["placement_rate"] = (
        blood_type_gold["organs_placed"] / blood_type_gold["organs_total"]
    ).fillna(0.0)

    organ_type_gold = (
        outcomes.groupby("organ_type", as_index=False)
        .agg(
            organs_total=("organ_type", "count"),
            organs_placed=("placed_flag", "sum"),
            organs_discarded=("discard_flag", "sum"),
            avg_cold_ischemia_minutes=("cold_ischemia_minutes", "mean"),
            p90_cold_ischemia_minutes=("cold_ischemia_minutes", lambda s: s.quantile(0.90)),
        )
        .sort_values("organ_type")
    )
    organ_type_gold["placement_rate"] = (
        organ_type_gold["organs_placed"] / organ_type_gold["organs_total"]
    ).fillna(0.0)

    referrals_in = len(referrals) + len(quarantine_referrals)
    outcomes_in = len(outcomes) + len(quarantine_outcomes)
    missing_fk_count = 0
    if "rejection_reason" in quarantine_outcomes.columns:
        missing_fk_count = int(
            quarantine_outcomes["rejection_reason"].astype(str).str.contains("missing_referral_fk").sum()
        )
    quality_gold = pd.DataFrame(
        [
            {"metric": "referrals_in", "value": referrals_in},
            {"metric": "referrals_valid", "value": len(referrals)},
            {"metric": "referrals_rejected", "value": len(quarantine_referrals)},
            {
                "metric": "referrals_rejection_rate",
                "value": (len(quarantine_referrals) / referrals_in) if referrals_in else 0.0,
            },
            {"metric": "outcomes_in", "value": outcomes_in},
            {"metric": "outcomes_valid", "value": len(outcomes)},
            {"metric": "outcomes_rejected", "value": len(quarantine_outcomes)},
            {
                "metric": "outcomes_rejection_rate",
                "value": (len(quarantine_outcomes) / outcomes_in) if outcomes_in else 0.0,
            },
            {"metric": "outcomes_missing_fk_rejected", "value": missing_fk_count},
            {
                "metric": "outcomes_missing_fk_rate",
                "value": (missing_fk_count / outcomes_in) if outcomes_in else 0.0,
            },
        ]
    )

    referral_reasons = (
        quarantine_referrals.get("rejection_reason", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
    )
    outcome_reasons = (
        quarantine_outcomes.get("rejection_reason", pd.Series(dtype="object"))
        .dropna()
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
    )
    referral_reason_counts = (
        referral_reasons[referral_reasons != ""]
        .value_counts()
        .rename_axis("rejection_reason")
        .reset_index(name="count")
    )
    referral_reason_counts["domain"] = "referral"
    outcome_reason_counts = (
        outcome_reasons[outcome_reasons != ""]
        .value_counts()
        .rename_axis("rejection_reason")
        .reset_index(name="count")
    )
    outcome_reason_counts["domain"] = "outcome"
    rejection_reason_gold = pd.concat(
        [referral_reason_counts, outcome_reason_counts], ignore_index=True
    )
    if not rejection_reason_gold.empty:
        rejection_reason_gold = rejection_reason_gold[
            ["domain", "rejection_reason", "count"]
        ].sort_values(["domain", "count", "rejection_reason"], ascending=[True, False, True])

    gold_daily.to_csv(GOLD_DIR / "gold_daily_metrics.csv", index=False)
    hospital_gold.to_csv(GOLD_DIR / "gold_hospital_metrics.csv", index=False)
    blood_type_gold.to_csv(GOLD_DIR / "gold_blood_type_metrics.csv", index=False)
    organ_type_gold.to_csv(GOLD_DIR / "gold_organ_type_metrics.csv", index=False)
    quality_gold.to_csv(GOLD_DIR / "gold_quality_metrics.csv", index=False)
    rejection_reason_gold.to_csv(GOLD_DIR / "gold_rejection_reason_metrics.csv", index=False)

    print("Gold layer created:")
    print(f"- {GOLD_DIR / 'gold_daily_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_hospital_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_blood_type_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_organ_type_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_quality_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_rejection_reason_metrics.csv'}")


if __name__ == "__main__":
    build_gold()

