from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("silver_layer")


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[2]
    bronze_dir: Path = project_root / "sample_data" / "l1_bronze"
    silver_dir: Path = project_root / "sample_data" / "l2_silver"
    quarantine_dir: Path = project_root / "sample_data" / "quarantine"


@dataclass(frozen=True)
class Outputs:
    silver_referrals: str = "silver_referrals.csv"
    silver_outcomes: str = "silver_placement_outcomes.csv"
    report: str = "silver_quality_report.csv"
    q_referrals: str = "quarantine_referrals.csv"
    q_outcomes: str = "quarantine_placement_outcomes.csv"


@dataclass(frozen=True)
class Rules:
    allowed_blood_types: Tuple[str, ...] = ("O", "A", "B", "AB")
    allowed_organs: Tuple[str, ...] = ("kidney", "liver", "heart", "lung")
    triage_min: float = 0.0
    triage_max: float = 1.0
    max_cold_ischemia_minutes: int = 3000


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {path.name}: rows={len(df):,}, cols={len(df.columns)}")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Wrote {path.name}: rows={len(df):,}")


def add_rejection_reason(
    df: pd.DataFrame, mask: pd.Series, reason: str, rr: Optional[pd.Series] = None
) -> pd.Series:
    if rr is None:
        rr = df.get("rejection_reason", pd.Series([""] * len(df), index=df.index))
    rr = rr.astype(str)
    rr.loc[mask] = rr.loc[mask].apply(lambda x: reason if x == "" else f"{x}|{reason}")
    return rr


def deterministic_dedupe_keep_first(
    df: pd.DataFrame, keys: List[str], sort_by: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    if sort_by:
        out = out.sort_values(by=sort_by, ascending=True, kind="mergesort")
    dup = int(out.duplicated(subset=keys).sum())
    out = out.drop_duplicates(subset=keys, keep="first")
    return out, dup


def _to_nullable_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )
    return mapped.astype("boolean")


def clean_referrals(referrals: pd.DataFrame, rules: Rules) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = referrals.copy()
    df["referral_ts"] = pd.to_datetime(df["referral_ts"], errors="coerce")
    df["triage_score"] = pd.to_numeric(df["triage_score"], errors="coerce")

    rr = pd.Series([""] * len(df), index=df.index)
    rr = add_rejection_reason(
        df, df["referral_id"].isna() | (df["referral_id"].astype(str).str.strip() == ""), "null_referral_id"
        , rr=rr
    )
    rr = add_rejection_reason(df, df["referral_ts"].isna(), "bad_timestamp", rr=rr)
    rr = add_rejection_reason(df, ~df["blood_type"].isin(rules.allowed_blood_types), "invalid_blood_type", rr=rr)
    rr = add_rejection_reason(df, df["triage_score"].isna(), "triage_non_numeric", rr=rr)
    rr = add_rejection_reason(
        df, ~df["triage_score"].between(rules.triage_min, rules.triage_max), "triage_out_of_range", rr=rr
    )
    df["rejection_reason"] = rr

    rejected = df.loc[df["rejection_reason"] != ""].copy()
    valid = df.loc[df["rejection_reason"] == ""].drop(columns=["rejection_reason"]).copy()

    valid, dup = deterministic_dedupe_keep_first(valid, keys=["referral_id"], sort_by=["referral_ts"])

    metrics = {
        "referrals_in": len(df),
        "referrals_rejected": len(rejected),
        "referrals_valid": len(valid),
        "referrals_duplicates_dropped": dup,
    }
    return valid, rejected, metrics


def clean_outcomes(outcomes: pd.DataFrame, rules: Rules) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = outcomes.copy()
    df["cold_ischemia_minutes"] = pd.to_numeric(df["cold_ischemia_minutes"], errors="coerce")

    placed = _to_nullable_bool(df["placed_flag"])
    discard = _to_nullable_bool(df["discard_flag"])

    rr = pd.Series([""] * len(df), index=df.index)
    rr = add_rejection_reason(
        df, df["referral_id"].isna() | (df["referral_id"].astype(str).str.strip() == ""), "null_referral_id"
        , rr=rr
    )
    rr = add_rejection_reason(df, ~df["organ_type"].isin(rules.allowed_organs), "invalid_organ_type", rr=rr)
    rr = add_rejection_reason(df, df["cold_ischemia_minutes"].isna(), "null_or_non_numeric_cold_ischemia", rr=rr)
    rr = add_rejection_reason(df, df["cold_ischemia_minutes"] < 0, "negative_cold_ischemia", rr=rr)
    rr = add_rejection_reason(
        df, df["cold_ischemia_minutes"] > rules.max_cold_ischemia_minutes, "cold_ischemia_too_large", rr=rr
    )
    rr = add_rejection_reason(df, placed.isna() | discard.isna(), "invalid_boolean_flags", rr=rr)
    rr = add_rejection_reason(
        df, (placed.notna() & discard.notna()) & (discard != ~placed), "discard_vs_placed_inconsistent", rr=rr
    )
    df["rejection_reason"] = rr

    rejected = df.loc[df["rejection_reason"] != ""].copy()
    valid = df.loc[df["rejection_reason"] == ""].drop(columns=["rejection_reason"]).copy()

    valid, dup = deterministic_dedupe_keep_first(valid, keys=["referral_id", "organ_type"])

    metrics = {
        "outcomes_in": len(df),
        "outcomes_rejected": len(rejected),
        "outcomes_valid": len(valid),
        "outcomes_duplicates_dropped": dup,
    }
    return valid, rejected, metrics


def enforce_fk(
    outcomes_valid: pd.DataFrame, referrals_valid: pd.DataFrame, referrals_rejected: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    ref_ids = set(referrals_valid["referral_id"].unique())
    missing_mask = ~outcomes_valid["referral_id"].isin(ref_ids)

    rejected_fk = outcomes_valid.loc[missing_mask].copy()
    if not rejected_fk.empty:
        rejected_reason_map = (
            referrals_rejected[["referral_id", "rejection_reason"]]
            .dropna(subset=["referral_id"])
            .drop_duplicates(subset=["referral_id"], keep="first")
            .set_index("referral_id")["rejection_reason"]
            .to_dict()
        )
        parent_reasons = rejected_fk["referral_id"].map(rejected_reason_map)
        rejected_fk["rejection_reason"] = parent_reasons.fillna("missing_referral_fk")
    kept = outcomes_valid.loc[~missing_mask].copy()

    metrics = {
        "outcomes_missing_fk_rejected": int(missing_mask.sum()),
        "outcomes_after_fk": len(kept),
    }
    return kept, rejected_fk, metrics


def attach_referral_triage(outcomes_df: pd.DataFrame, referrals_df: pd.DataFrame) -> pd.DataFrame:
    if outcomes_df.empty:
        return outcomes_df.copy()
    triage_lookup = (
        referrals_df[["referral_id", "triage_score"]]
        .dropna(subset=["referral_id"])
        .drop_duplicates(subset=["referral_id"], keep="first")
        .rename(columns={"triage_score": "referral_triage_score"})
    )
    return outcomes_df.merge(triage_lookup, on="referral_id", how="left")


def attach_referral_dates(outcomes_df: pd.DataFrame, referrals_df: pd.DataFrame) -> pd.DataFrame:
    if outcomes_df.empty:
        return outcomes_df.copy()
    referral_dates = (
        referrals_df[["referral_id", "referral_ts"]]
        .dropna(subset=["referral_id"])
        .drop_duplicates(subset=["referral_id"], keep="first")
        .copy()
    )
    out = outcomes_df.merge(referral_dates, on="referral_id", how="left")
    out["referral_date"] = pd.to_datetime(out["referral_ts"], errors="coerce").dt.date
    return out


def move_rejection_reason_last(df: pd.DataFrame) -> pd.DataFrame:
    if "rejection_reason" not in df.columns:
        return df
    cols = [c for c in df.columns if c != "rejection_reason"] + ["rejection_reason"]
    return df.loc[:, cols]


def build_silver(paths: Paths = Paths(), outputs: Outputs = Outputs(), rules: Rules = Rules()) -> None:
    referrals = load_csv(paths.bronze_dir / "referrals.csv")
    outcomes = load_csv(paths.bronze_dir / "placement_outcomes.csv")

    referrals_valid, referrals_rejected, m_ref = clean_referrals(referrals, rules)
    outcomes_valid, outcomes_rejected, m_out = clean_outcomes(outcomes, rules)
    outcomes_kept, outcomes_fk_rejected, m_fk = enforce_fk(
        outcomes_valid, referrals_valid, referrals_rejected
    )

    outcomes_kept = attach_referral_dates(outcomes_kept, referrals_valid)
    outcomes_rejected_all = pd.concat([outcomes_rejected, outcomes_fk_rejected], ignore_index=True)
    outcomes_rejected_all = attach_referral_dates(outcomes_rejected_all, referrals)
    outcomes_rejected_all = attach_referral_triage(outcomes_rejected_all, referrals)
    referrals_rejected = move_rejection_reason_last(referrals_rejected)
    outcomes_rejected_all = move_rejection_reason_last(outcomes_rejected_all)

    write_csv(referrals_valid, paths.silver_dir / outputs.silver_referrals)
    write_csv(outcomes_kept, paths.silver_dir / outputs.silver_outcomes)
    write_csv(referrals_rejected, paths.quarantine_dir / outputs.q_referrals)
    write_csv(outcomes_rejected_all, paths.quarantine_dir / outputs.q_outcomes)

    report = {
        **m_ref,
        **m_out,
        **m_fk,
        "silver_referrals_rows": len(referrals_valid),
        "silver_outcomes_rows": len(outcomes_kept),
        "quarantine_referrals_rows": len(referrals_rejected),
        "quarantine_outcomes_rows": len(outcomes_rejected_all),
    }
    report_df = pd.DataFrame(report.items(), columns=["metric", "value"])
    write_csv(report_df, paths.silver_dir / outputs.report)

    logger.info("Silver layer built. Invalid rows removed and quarantined.")


if __name__ == "__main__":
    build_silver()

