import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random
import math
import matplotlib.pyplot as plt


def negbin_count(mu: float, k: float) -> int:
    """
    Negative Binomial count with mean ~ mu and overdispersion controlled by k.
    Smaller k => more variance/spikes.
    Variance = mu + mu^2 / k
    """
    p = k / (k + mu)
    return int(np.random.negative_binomial(n=k, p=p))


def negbin_pmf(x_vals: np.ndarray, mu: float, k: float) -> np.ndarray:
    """
    Negative Binomial PMF parameterized by mean (mu) and dispersion (k).
    P(X=x) = C(x+k-1, x) * p^k * (1-p)^x, where p = k/(k+mu)
    Works for non-integer k via gamma functions.
    """
    p = k / (k + mu)
    x_arr = np.asarray(x_vals, dtype=float)

    log_coeff = (
        np.array([math.lgamma(x + k) for x in x_arr])
        - math.lgamma(k)
        - np.array([math.lgamma(x + 1.0) for x in x_arr])
    )
    log_pmf = log_coeff + k * math.log(p) + x_arr * math.log(1.0 - p)
    return np.exp(log_pmf)


def poisson_pmf(x_vals: np.ndarray, mu: float) -> np.ndarray:
    """Poisson PMF with mean mu."""
    x_arr = np.asarray(x_vals, dtype=float)
    log_pmf = -mu + x_arr * math.log(mu) - np.array([math.lgamma(x + 1.0) for x in x_arr])
    return np.exp(log_pmf)


def plot_negbin_vs_poisson(mu: float = 12.0, k: float = 6.0, max_x: int | None = None):
    """
    Plot Negative Binomial and Poisson PMFs on the same graph for comparison.
    """
    if max_x is None:
        var_nb = mu + (mu ** 2) / k
        sd_nb = math.sqrt(var_nb)
        sd_pois = math.sqrt(mu)
        max_x = int(max(mu + 6 * sd_nb, mu + 6 * sd_pois))

    x = np.arange(0, max_x + 1)
    nb = negbin_pmf(x, mu=mu, k=k)
    pois = poisson_pmf(x, mu=mu)

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "sample_data" / "l1_bronze"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "negbin_vs_poisson.png"

    plt.figure(figsize=(10, 5))
    plt.plot(x, nb, marker="o", linewidth=1.8, label=f"Negative Binomial (mu={mu}, k={k})")
    plt.plot(x, pois, marker="s", linewidth=1.8, label=f"Poisson (mu = Var = {mu})")
    plt.title("Negative Binomial vs Poisson")
    plt.xlabel("Number of Referrals (Count)")
    plt.ylabel("Probability (PMF)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot: {out_path}")


def inject_inconsistencies(referrals_df: pd.DataFrame, outcomes_df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed + 101)
    referrals = referrals_df.copy()
    outcomes = outcomes_df.copy()
    stats = {}

    if len(referrals) > 0:
        dup_n = min(len(referrals), max(2, len(referrals) // 100))
        referrals = pd.concat(
            [referrals, referrals.sample(n=dup_n, random_state=seed + 1)], ignore_index=True
        )
        stats["referral_duplicate_rows_added"] = int(dup_n)

        referrals["referral_ts"] = referrals["referral_ts"].astype(object)
        bad_ts_n = min(len(referrals), max(2, len(referrals) // 200))
        bad_ts_idx = rng.choice(referrals.index.to_numpy(), size=bad_ts_n, replace=False)
        referrals.loc[bad_ts_idx, "referral_ts"] = "bad_timestamp"
        stats["referral_bad_timestamp_rows"] = int(bad_ts_n)

        invalid_blood_n = min(len(referrals), max(3, len(referrals) // 150))
        invalid_blood_idx = rng.choice(referrals.index.to_numpy(), size=invalid_blood_n, replace=False)
        referrals.loc[invalid_blood_idx, "blood_type"] = "X"
        stats["referral_invalid_blood_type_rows"] = int(invalid_blood_n)

        triage_n = min(len(referrals), max(4, len(referrals) // 120))
        triage_idx = rng.choice(referrals.index.to_numpy(), size=triage_n, replace=False)
        half = triage_n // 2
        referrals.loc[triage_idx[:half], "triage_score"] = -0.2
        referrals.loc[triage_idx[half:], "triage_score"] = 1.25
        stats["referral_out_of_range_triage_rows"] = int(triage_n)

        null_ref_n = min(len(referrals), max(2, len(referrals) // 250))
        null_ref_idx = rng.choice(referrals.index.to_numpy(), size=null_ref_n, replace=False)
        referrals.loc[null_ref_idx, "referral_id"] = None
        stats["referral_null_id_rows"] = int(null_ref_n)

    if len(outcomes) > 0:
        dup_n = min(len(outcomes), max(4, len(outcomes) // 120))
        outcomes = pd.concat(
            [outcomes, outcomes.sample(n=dup_n, random_state=seed + 2)], ignore_index=True
        )
        stats["outcome_duplicate_rows_added"] = int(dup_n)

        invalid_organ_n = min(len(outcomes), max(3, len(outcomes) // 180))
        invalid_organ_idx = rng.choice(
            outcomes.index.to_numpy(), size=invalid_organ_n, replace=False
        )
        outcomes.loc[invalid_organ_idx, "organ_type"] = "pancreas"
        stats["outcome_invalid_organ_rows"] = int(invalid_organ_n)

        neg_cold_n = min(len(outcomes), max(4, len(outcomes) // 160))
        neg_cold_idx = rng.choice(outcomes.index.to_numpy(), size=neg_cold_n, replace=False)
        outcomes.loc[neg_cold_idx, "cold_ischemia_minutes"] = -rng.integers(
            low=1, high=180, size=neg_cold_n
        )
        stats["outcome_negative_cold_ischemia_rows"] = int(neg_cold_n)

        outcomes["cold_ischemia_minutes"] = outcomes["cold_ischemia_minutes"].astype(object)
        bad_cold_n = min(len(outcomes), max(3, len(outcomes) // 200))
        bad_cold_idx = rng.choice(outcomes.index.to_numpy(), size=bad_cold_n, replace=False)
        outcomes.loc[bad_cold_idx, "cold_ischemia_minutes"] = "unknown"
        stats["outcome_non_numeric_cold_ischemia_rows"] = int(bad_cold_n)

        inconsistent_flags_n = min(len(outcomes), max(4, len(outcomes) // 200))
        inconsistent_idx = rng.choice(
            outcomes.index.to_numpy(), size=inconsistent_flags_n, replace=False
        )
        outcomes.loc[inconsistent_idx, "placed_flag"] = True
        outcomes.loc[inconsistent_idx, "discard_flag"] = True
        stats["outcome_inconsistent_flag_rows"] = int(inconsistent_flags_n)

        orphan_n = min(len(outcomes), max(3, len(outcomes) // 220))
        orphan_idx = rng.choice(outcomes.index.to_numpy(), size=orphan_n, replace=False)
        outcomes.loc[orphan_idx, "referral_id"] = [
            f"ORPHAN_{i:04d}" for i in range(orphan_n)
        ]
        stats["outcome_orphan_referral_rows"] = int(orphan_n)

        null_ref_n = min(len(outcomes), max(2, len(outcomes) // 250))
        null_ref_idx = rng.choice(outcomes.index.to_numpy(), size=null_ref_n, replace=False)
        outcomes.loc[null_ref_idx, "referral_id"] = None
        stats["outcome_null_referral_id_rows"] = int(null_ref_n)

    return referrals, outcomes, stats


def gen_synthetic(days=90, avg_referrals_per_day=12, seed=7, overdisp_k=6.0):
    np.random.seed(seed)
    random.seed(seed)

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "sample_data" / "l1_bronze"
    out_dir.mkdir(parents=True, exist_ok=True)

    hospitals = [f"H{str(i).zfill(3)}" for i in range(1, 11)]
    organs = ["kidney", "liver", "heart", "lung"]

    referral_rows = []
    outcome_rows = []

    start = datetime(2025, 1, 1)
    rid = 0

    for d in range(days):
        day = start + timedelta(days=d)
        n = negbin_count(mu=avg_referrals_per_day, k=overdisp_k)

        for _ in range(n):
            rid += 1
            referral_id = f"R{rid:05d}"
            hospital_id = random.choice(hospitals)
            triage_score = round(np.random.uniform(0.2, 0.95), 2)
            referral_ts = day + timedelta(seconds=random.randint(0, 24 * 60 * 60 - 1))

            referral_rows.append(
                {
                    "referral_id": referral_id,
                    "hospital_id": hospital_id,
                    "referral_ts": referral_ts,
                    "triage_score": triage_score,
                    "blood_type": random.choice(["O", "A", "B", "AB"]),
                }
            )

            for organ in organs:
                cold_time = int(np.random.normal(400, 120))
                discard_prob = max(0, min(1, (cold_time - 300) / 600))
                discard = np.random.rand() < discard_prob

                outcome_rows.append(
                    {
                        "referral_id": referral_id,
                        "organ_type": organ,
                        "cold_ischemia_minutes": cold_time,
                        "placed_flag": not discard,
                        "discard_flag": discard,
                    }
                )

    referrals_df = pd.DataFrame(referral_rows)
    outcomes_df = pd.DataFrame(outcome_rows)
    referrals_df, outcomes_df, issue_stats = inject_inconsistencies(
        referrals_df, outcomes_df, seed=seed
    )

    referrals_df.to_csv(out_dir / "referrals.csv", index=False)
    outcomes_df.to_csv(out_dir / "placement_outcomes.csv", index=False)

    print(f"Fake data generated in {out_dir}")
    print(f"Negative Binomial overdispersion k={overdisp_k} (smaller => spikier)")
    print("Injected inconsistencies for silver-layer validation:")
    for key, value in issue_stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    gen_synthetic()
    plot_negbin_vs_poisson()

