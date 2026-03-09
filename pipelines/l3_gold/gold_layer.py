import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

    # Observable referral-to-donor funnel.
    referral_conversion = (
        merged.groupby(["referral_id", "hospital_id", "referral_date"], as_index=False)
        .agg(
            organs_total=("organ_type", "count"),
            organs_placed=("placed_flag", "sum"),
            organs_discarded=("discard_flag", "sum"),
            avg_cold_ischemia_minutes=("cold_ischemia_minutes", "mean"),
        )
    )
    referral_conversion["donor_flag"] = (referral_conversion["organs_placed"] > 0).astype(int)

    daily_funnel = (
        referral_conversion.groupby("referral_date", as_index=False)
        .agg(
            referrals_count=("referral_id", "nunique"),
            donor_referrals=("donor_flag", "sum"),
            total_organs_placed=("organs_placed", "sum"),
            total_organs_discarded=("organs_discarded", "sum"),
        )
        .sort_values("referral_date")
    )
    daily_funnel["non_donor_referrals"] = (
        daily_funnel["referrals_count"] - daily_funnel["donor_referrals"]
    )
    daily_funnel["referral_to_donor_conversion_rate"] = (
        daily_funnel["donor_referrals"] / daily_funnel["referrals_count"]
    ).fillna(0.0)

    hospital_conversion_gold = (
        referral_conversion.groupby("hospital_id", as_index=False)
        .agg(
            referrals_count=("referral_id", "nunique"),
            donor_referrals=("donor_flag", "sum"),
            organs_placed=("organs_placed", "sum"),
            organs_discarded=("organs_discarded", "sum"),
            avg_cold_ischemia_minutes=("avg_cold_ischemia_minutes", "mean"),
        )
        .sort_values("hospital_id")
    )
    hospital_conversion_gold["non_donor_referrals"] = (
        hospital_conversion_gold["referrals_count"] - hospital_conversion_gold["donor_referrals"]
    )
    hospital_conversion_gold["referral_to_donor_conversion_rate"] = (
        hospital_conversion_gold["donor_referrals"] / hospital_conversion_gold["referrals_count"]
    ).fillna(0.0)

    global_conv_rate = (
        float(referral_conversion["donor_flag"].mean()) if len(referral_conversion) else 0.0
    )
    missed_opportunity_gold = hospital_conversion_gold[
        ["hospital_id", "referrals_count", "donor_referrals", "referral_to_donor_conversion_rate"]
    ].copy()
    missed_opportunity_gold["expected_donor_referrals"] = (
        missed_opportunity_gold["referrals_count"] * global_conv_rate
    )
    missed_opportunity_gold["donor_gap_vs_baseline"] = (
        missed_opportunity_gold["expected_donor_referrals"] - missed_opportunity_gold["donor_referrals"]
    )
    missed_opportunity_gold["underperforming_flag"] = (
        missed_opportunity_gold["donor_gap_vs_baseline"] > 0
    ).astype(int)

    # Lightweight referral propensity score (operational prioritization).
    referrals_for_score = referrals.copy()
    referrals_for_score["referral_date"] = pd.to_datetime(
        referrals_for_score["referral_ts"], errors="coerce"
    ).dt.date
    referrals_for_score["referral_hour"] = pd.to_datetime(
        referrals_for_score["referral_ts"], errors="coerce"
    ).dt.hour

    donor_lookup = referral_conversion[["referral_id", "donor_flag"]].copy()
    hosp_lookup = hospital_conversion_gold[
        ["hospital_id", "referral_to_donor_conversion_rate"]
    ].rename(columns={"referral_to_donor_conversion_rate": "hospital_historical_conversion_rate"})

    referral_propensity_gold = (
        referrals_for_score.merge(donor_lookup, on="referral_id", how="left")
        .merge(hosp_lookup, on="hospital_id", how="left")
    )
    referral_propensity_gold["donor_flag"] = referral_propensity_gold["donor_flag"].fillna(0).astype(int)
    referral_propensity_gold["hospital_historical_conversion_rate"] = referral_propensity_gold[
        "hospital_historical_conversion_rate"
    ].fillna(global_conv_rate)
    referral_propensity_gold["triage_score"] = pd.to_numeric(
        referral_propensity_gold["triage_score"], errors="coerce"
    ).fillna(referral_propensity_gold["triage_score"].median())
    referral_propensity_gold["referral_hour"] = pd.to_numeric(
        referral_propensity_gold["referral_hour"], errors="coerce"
    ).fillna(12)

    # Heuristic score (interpretable operational baseline).
    hour_factor = np.where(
        referral_propensity_gold["referral_hour"].between(7, 22),
        1.0,
        0.9,
    )
    referral_propensity_gold["heuristic_donor_probability"] = (
        0.50 * referral_propensity_gold["triage_score"]
        + 0.40 * referral_propensity_gold["hospital_historical_conversion_rate"]
        + 0.10 * hour_factor
    ).clip(0.0, 0.99)

    # Logistic regression score (data-driven model).
    feature_cols_num = ["triage_score", "referral_hour", "hospital_historical_conversion_rate"]
    feature_cols_cat = ["hospital_id", "blood_type"]
    X = referral_propensity_gold[feature_cols_num + feature_cols_cat].copy()
    y = referral_propensity_gold["donor_flag"].astype(int)

    if y.nunique() >= 2 and len(referral_propensity_gold) >= 50:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                    feature_cols_num,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    feature_cols_cat,
                ),
            ]
        )
        clf = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=7)),
            ]
        )
        clf.fit(X, y)
        reg_probs = clf.predict_proba(X)[:, 1]
        referral_propensity_gold["regression_donor_probability"] = np.clip(reg_probs, 0.0, 0.99)
        model_name = "logistic_regression"
    else:
        # Fallback when data lacks class balance for model training.
        referral_propensity_gold["regression_donor_probability"] = referral_propensity_gold[
            "heuristic_donor_probability"
        ]
        model_name = "fallback_heuristic"

    # Operational blended score to rank cases.
    referral_propensity_gold["predicted_donor_probability"] = (
        0.35 * referral_propensity_gold["heuristic_donor_probability"]
        + 0.65 * referral_propensity_gold["regression_donor_probability"]
    ).clip(0.0, 0.99)

    referral_propensity_gold["priority_band"] = pd.cut(
        referral_propensity_gold["predicted_donor_probability"],
        bins=[-0.01, 0.4, 0.7, 1.0],
        labels=["low", "medium", "high"],
    )
    referral_propensity_gold["propensity_model"] = model_name
    referral_propensity_gold = referral_propensity_gold[
        [
            "referral_id",
            "hospital_id",
            "blood_type",
            "referral_ts",
            "referral_date",
            "triage_score",
            "referral_hour",
            "hospital_historical_conversion_rate",
            "heuristic_donor_probability",
            "regression_donor_probability",
            "predicted_donor_probability",
            "priority_band",
            "propensity_model",
            "donor_flag",
        ]
    ].sort_values(["predicted_donor_probability", "referral_id"], ascending=[False, True])

    # Discard by ischemia delay bucket.
    discard_delay = outcomes.copy()
    discard_delay["cold_ischemia_minutes"] = pd.to_numeric(
        discard_delay["cold_ischemia_minutes"], errors="coerce"
    )
    discard_delay["discard_flag"] = pd.to_numeric(discard_delay["discard_flag"], errors="coerce").fillna(0)
    discard_delay["delay_bucket"] = pd.cut(
        discard_delay["cold_ischemia_minutes"],
        bins=[-1, 240, 360, 480, 3000],
        labels=["0-4h", "4-6h", "6-8h", "8h+"],
    )
    discard_delay_gold = (
        discard_delay.groupby("delay_bucket", as_index=False)
        .agg(
            organs_total=("organ_type", "count"),
            organs_discarded=("discard_flag", "sum"),
            avg_cold_ischemia_minutes=("cold_ischemia_minutes", "mean"),
        )
        .sort_values("delay_bucket")
    )
    discard_delay_gold["discard_rate"] = (
        discard_delay_gold["organs_discarded"] / discard_delay_gold["organs_total"]
    ).fillna(0.0)

    # 14-day referral/donor volume forecast.
    forecast_days = 14
    forecast_base = daily_funnel[["referral_date", "referrals_count", "donor_referrals"]].copy()
    forecast_base["referral_date"] = pd.to_datetime(forecast_base["referral_date"], errors="coerce")
    forecast_base = forecast_base.sort_values("referral_date").dropna(subset=["referral_date"])
    if not forecast_base.empty:
        x = np.arange(len(forecast_base), dtype=float)
        y_ref = forecast_base["referrals_count"].to_numpy(dtype=float)
        y_don = forecast_base["donor_referrals"].to_numpy(dtype=float)
        ref_baseline = float(pd.Series(y_ref).tail(14).mean())
        don_baseline = float(pd.Series(y_don).tail(14).mean())
        ref_slope = float(np.polyfit(x, y_ref, 1)[0]) if len(y_ref) >= 7 else 0.0
        don_slope = float(np.polyfit(x, y_don, 1)[0]) if len(y_don) >= 7 else 0.0
        start_date = forecast_base["referral_date"].max()
        rows = []
        for i in range(1, forecast_days + 1):
            rows.append(
                {
                    "forecast_date": (start_date + pd.Timedelta(days=i)).date(),
                    "predicted_referrals_count": round(max(0.0, ref_baseline + ref_slope * i), 2),
                    "predicted_donor_referrals": round(max(0.0, don_baseline + don_slope * i), 2),
                    "model": "rolling_mean_plus_trend",
                }
            )
        donor_volume_forecast_gold = pd.DataFrame(rows)
    else:
        donor_volume_forecast_gold = pd.DataFrame(
            columns=["forecast_date", "predicted_referrals_count", "predicted_donor_referrals", "model"]
        )

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
    daily_funnel.to_csv(GOLD_DIR / "gold_referral_funnel_metrics.csv", index=False)
    hospital_conversion_gold.to_csv(GOLD_DIR / "gold_hospital_conversion_metrics.csv", index=False)
    missed_opportunity_gold.to_csv(GOLD_DIR / "gold_missed_opportunity_metrics.csv", index=False)
    referral_propensity_gold.to_csv(GOLD_DIR / "gold_referral_propensity_metrics.csv", index=False)
    discard_delay_gold.to_csv(GOLD_DIR / "gold_discard_delay_metrics.csv", index=False)
    donor_volume_forecast_gold.to_csv(GOLD_DIR / "gold_donor_volume_forecast.csv", index=False)

    print("Gold layer created:")
    print(f"- {GOLD_DIR / 'gold_daily_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_hospital_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_blood_type_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_organ_type_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_quality_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_rejection_reason_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_referral_funnel_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_hospital_conversion_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_missed_opportunity_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_referral_propensity_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_discard_delay_metrics.csv'}")
    print(f"- {GOLD_DIR / 'gold_donor_volume_forecast.csv'}")


if __name__ == "__main__":
    build_gold()
