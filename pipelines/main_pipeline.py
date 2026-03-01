from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_pipeline(
    generator: str,
    days: int,
    avg_referrals_per_day: int,
    seed: int,
    overdisp_k: float,
    skip_silver: bool,
    skip_gold: bool,
    no_plot: bool,
) -> None:
    bronze_poisson = PROJECT_ROOT / "pipelines" / "l1_bronze" / "generate_synthetic_data.py"
    bronze_negbin = (
        PROJECT_ROOT / "pipelines" / "l1_bronze" / "generate_synthetic_data_negative_binomial.py"
    )
    silver_script = PROJECT_ROOT / "pipelines" / "l2_silver" / "silver_layer_v2.py"
    gold_script = PROJECT_ROOT / "pipelines" / "l3_gold" / "gold_layer.py"

    if generator == "poisson":
        bronze = load_module(bronze_poisson, "bronze_poisson")
        print("[1/3] Running bronze (Poisson generation)")
        bronze.gen_synthetic(
            days=days, avg_referrals_per_day=avg_referrals_per_day, seed=seed
        )
    else:
        bronze = load_module(bronze_negbin, "bronze_negbin")
        print("[1/3] Running bronze (Negative Binomial generation)")
        bronze.gen_synthetic(
            days=days,
            avg_referrals_per_day=avg_referrals_per_day,
            seed=seed,
            overdisp_k=overdisp_k,
        )
        if not no_plot:
            bronze.plot_negbin_vs_poisson(mu=float(avg_referrals_per_day), k=overdisp_k)

    if skip_silver:
        print("[2/3] Skipping silver")
    else:
        silver = load_module(silver_script, "silver_layer")
        print("[2/3] Running silver")
        silver.build_silver()

    if skip_gold:
        print("[3/3] Skipping gold")
    else:
        gold = load_module(gold_script, "gold_layer")
        print("[3/3] Running gold")
        gold.build_gold()

    print("Pipeline finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bronze -> Silver -> Gold pipeline.")
    parser.add_argument(
        "--generator",
        choices=["poisson", "negbin"],
        default="negbin",
        help="Bronze generation mode.",
    )
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--avg-referrals-per-day", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--overdisp-k",
        type=float,
        default=6.0,
        help="Negative Binomial dispersion parameter (used with --generator negbin).",
    )
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable Poisson vs Negative Binomial plot generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        generator=args.generator,
        days=args.days,
        avg_referrals_per_day=args.avg_referrals_per_day,
        seed=args.seed,
        overdisp_k=args.overdisp_k,
        skip_silver=args.skip_silver,
        skip_gold=args.skip_gold,
        no_plot=args.no_plot,
    )
