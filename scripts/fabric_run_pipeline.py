from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def default_project_root() -> Path:
    # Common Fabric layouts.
    candidates = [
        Path("/lakehouse/default/Files/organ_donors_analytics"),
        Path("/lakehouse/default/Files"),
    ]

    # Repo-local execution.
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parents[1])

    # Notebook execution fallback.
    candidates.append(Path.cwd())

    for root in candidates:
        if (root / "pipelines" / "main_pipeline.py").exists():
            return root
    return Path.cwd()


def run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local medallion pipeline from Microsoft Fabric runtime."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
        help="Path containing pipelines/main_pipeline.py",
    )
    parser.add_argument("--generator", choices=["poisson", "negbin"], default="negbin")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--avg-referrals-per-day", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overdisp-k", type=float, default=6.0)
    parser.add_argument("--skip-silver", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--no-plot", action="store_true", default=True)
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install requirements.txt before running pipeline.",
    )
    parser.add_argument(
        "--launch-dashboard",
        action="store_true",
        help="Start Dash dashboard after pipeline completes.",
    )
    parser.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dash host (used with --launch-dashboard).",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8050,
        help="Dash port (used with --launch-dashboard).",
    )
    parser.add_argument(
        "--dashboard-debug",
        action="store_true",
        help="Enable Dash debug mode (used with --launch-dashboard).",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    pipeline_script = project_root / "pipelines" / "main_pipeline.py"
    dashboard_script = project_root / "scripts" / "dash_dashboard.py"
    requirements = project_root / "requirements.txt"

    if not pipeline_script.exists():
        raise FileNotFoundError(f"Missing pipeline script: {pipeline_script}")

    if args.install_deps and requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], cwd=project_root)

    cmd = [
        sys.executable,
        str(pipeline_script),
        "--generator",
        args.generator,
        "--days",
        str(args.days),
        "--avg-referrals-per-day",
        str(args.avg_referrals_per_day),
        "--seed",
        str(args.seed),
        "--overdisp-k",
        str(args.overdisp_k),
    ]
    if args.skip_silver:
        cmd.append("--skip-silver")
    if args.skip_gold:
        cmd.append("--skip-gold")
    if args.no_plot:
        cmd.append("--no-plot")

    run(cmd, cwd=project_root)

    base = project_root / "sample_data"
    print("Outputs:")
    print("-", base / "l1_bronze")
    print("-", base / "l2_silver")
    print("-", base / "quarantine")
    print("-", base / "l3_gold")

    if args.launch_dashboard:
        if not dashboard_script.exists():
            raise FileNotFoundError(f"Missing dashboard script: {dashboard_script}")

        dashboard_cmd = [
            sys.executable,
            str(dashboard_script),
            "--project-root",
            str(project_root),
            "--host",
            args.dashboard_host,
            "--port",
            str(args.dashboard_port),
        ]
        if args.dashboard_debug:
            dashboard_cmd.append("--debug")

        print(f"Starting dashboard at http://{args.dashboard_host}:{args.dashboard_port}")
        run(dashboard_cmd, cwd=project_root)


if __name__ == "__main__":
    main()
