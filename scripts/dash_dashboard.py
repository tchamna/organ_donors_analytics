from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from dash import dash_table
from dash.dash_table.Format import Format, Scheme


def default_project_root() -> Path:
    candidates = [
        Path("/lakehouse/default/Files/organ_donors_analytics"),
        Path("/lakehouse/default/Files"),
    ]
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parents[1])
    candidates.append(Path.cwd())

    for root in candidates:
        if (root / "sample_data" / "l3_gold").exists():
            return root
    return Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dash dashboards from Gold/Silver medallion outputs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(),
        help="Path containing sample_data/l3_gold and sample_data/l2_silver",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_data(project_root: Path) -> dict[str, pd.DataFrame]:
    gold_dir = project_root / "sample_data" / "l3_gold"
    silver_dir = project_root / "sample_data" / "l2_silver"
    return {
        "daily": _read_csv_if_exists(gold_dir / "gold_daily_metrics.csv"),
        "hospital": _read_csv_if_exists(gold_dir / "gold_hospital_metrics.csv"),
        "blood": _read_csv_if_exists(gold_dir / "gold_blood_type_metrics.csv"),
        "organ": _read_csv_if_exists(gold_dir / "gold_organ_type_metrics.csv"),
        "quality": _read_csv_if_exists(gold_dir / "gold_quality_metrics.csv"),
        "rejection": _read_csv_if_exists(gold_dir / "gold_rejection_reason_metrics.csv"),
        "funnel": _read_csv_if_exists(gold_dir / "gold_referral_funnel_metrics.csv"),
        "missed": _read_csv_if_exists(gold_dir / "gold_missed_opportunity_metrics.csv"),
        "propensity": _read_csv_if_exists(gold_dir / "gold_referral_propensity_metrics.csv"),
        "discard_delay": _read_csv_if_exists(gold_dir / "gold_discard_delay_metrics.csv"),
        "forecast": _read_csv_if_exists(gold_dir / "gold_donor_volume_forecast.csv"),
        "offline_forecast": _read_csv_if_exists(gold_dir / "gold_offline_model_forecast_results.csv"),
        "offline_metrics": _read_csv_if_exists(gold_dir / "gold_offline_model_forecast_metrics.csv"),
        "silver_outcomes": _read_csv_if_exists(silver_dir / "silver_placement_outcomes.csv"),
    }


def load_documentation(project_root: Path) -> str:
    doc_path = project_root / "docs" / "dashboard_documentation.md"
    if doc_path.exists():
        return doc_path.read_text(encoding="utf-8")
    return (
        "# Documentation\n\n"
        "Documentation file not found: `docs/dashboard_documentation.md`.\n"
    )


def _fmt_rate(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{100.0 * float(value):.1f}%"


def _card(title: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
        ],
        className="metric-card",
    )


def _empty_message(msg: str) -> html.Div:
    return html.Div(msg, className="empty-message")


def _style_fig(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        colorway=["#0f766e", "#0369a1", "#16a34a", "#f59e0b", "#dc2626", "#6d28d9"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Space Grotesk, Segoe UI, sans-serif", "color": "#0f172a"},
        title={"x": 0.01, "xanchor": "left", "font": {"size": 20, "family": "Fraunces, serif"}},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.01},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", linecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", linecolor="#cbd5e1")


def _styled_table(
    data: list[dict], columns: list[str], table_id: str | None = None
) -> dash_table.DataTable:
    kwargs = {}
    if table_id:
        kwargs["id"] = table_id

    df = pd.DataFrame(data, columns=columns)
    dash_columns: list[dict] = []
    for c in columns:
        col_def: dict = {"name": c, "id": c}
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            col_def["type"] = "numeric"
            col_def["format"] = Format(precision=2, scheme=Scheme.fixed)
        dash_columns.append(col_def)

    return dash_table.DataTable(
        data=data,
        columns=dash_columns,
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "borderRadius": "12px", "overflow": "hidden"},
        style_header={
            "backgroundColor": "#0f172a",
            "color": "#e2e8f0",
            "fontFamily": "Space Grotesk, Segoe UI, sans-serif",
            "fontWeight": "700",
            "fontSize": 13,
            "border": "none",
        },
        style_data={
            "backgroundColor": "#ffffff",
            "color": "#1e293b",
            "borderColor": "#e2e8f0",
            "fontFamily": "Space Grotesk, Segoe UI, sans-serif",
            "fontSize": 13,
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
        ],
        style_cell={"textAlign": "left", "padding": "10px"},
        **kwargs,
    )


def build_overview_tab(daily: pd.DataFrame) -> html.Div:
    if daily.empty:
        return _empty_message("No data in gold_daily_metrics.csv")

    daily = daily.copy()
    daily["referral_date"] = pd.to_datetime(daily["referral_date"], errors="coerce")
    daily = daily.sort_values("referral_date")

    kpi_referrals = int(daily["referrals_count"].fillna(0).sum())
    kpi_organs_total = int(daily["organs_total"].fillna(0).sum())
    kpi_organs_placed = int(daily["organs_placed"].fillna(0).sum())
    kpi_organs_discarded = int(daily["organs_discarded"].fillna(0).sum())
    placement_rate = (
        (kpi_organs_placed / kpi_organs_total) if kpi_organs_total else None
    )

    fig_activity = px.line(
        daily,
        x="referral_date",
        y=["referrals_count", "organs_placed", "organs_discarded"],
        title="Daily Referrals and Outcomes",
    )
    _style_fig(fig_activity)
    fig_activity.update_layout(legend_title_text="")

    fig_efficiency = px.line(
        daily,
        x="referral_date",
        y=["avg_cold_ischemia_minutes", "avg_triage_score"],
        title="Daily Efficiency Trends",
    )
    _style_fig(fig_efficiency)
    fig_efficiency.update_layout(legend_title_text="")

    overview_table = _styled_table(data=daily.to_dict("records"), columns=list(daily.columns))

    return html.Div(
        [
            html.Div(
                [
                    _card("Referrals", f"{kpi_referrals:,}"),
                    _card("Organs Total", f"{kpi_organs_total:,}"),
                    _card("Organs Placed", f"{kpi_organs_placed:,}"),
                    _card("Organs Discarded", f"{kpi_organs_discarded:,}"),
                    _card("Placement Rate", _fmt_rate(placement_rate)),
                ],
                className="metric-grid",
            ),
            dcc.Graph(figure=fig_activity, className="chart-panel"),
            dcc.Graph(figure=fig_efficiency, className="chart-panel"),
            html.H4("Daily Detail", className="section-title"),
            overview_table,
        ]
    )


def build_hospital_tab(hospital: pd.DataFrame) -> html.Div:
    if hospital.empty:
        return _empty_message("No data in gold_hospital_metrics.csv")

    hospital = hospital.copy()
    hospital = hospital.sort_values("placement_rate", ascending=False)
    top_10 = hospital.head(10).copy()
    bottom_10 = hospital.sort_values("placement_rate", ascending=True).head(10).copy()

    fig_rank_top = px.bar(
        top_10,
        x="hospital_id",
        y="placement_rate",
        title="Top 10 Hospital Placement Rates",
    )
    _style_fig(fig_rank_top)
    fig_rank_top.update_yaxes(tickformat=".0%")

    fig_rank_bottom = px.bar(
        bottom_10,
        x="hospital_id",
        y="placement_rate",
        title="Bottom 10 Hospital Placement Rates (Focus Areas)",
    )
    _style_fig(fig_rank_bottom)
    fig_rank_bottom.update_yaxes(tickformat=".0%")

    fig_scatter = px.scatter(
        hospital,
        x="organs_total",
        y="placement_rate",
        size="organs_placed",
        color="organs_discarded",
        hover_data=["hospital_id", "avg_cold_ischemia_minutes"],
        title="Volume vs Placement Performance",
    )
    _style_fig(fig_scatter)
    fig_scatter.update_yaxes(tickformat=".0%")

    fig_discard = px.bar(
        hospital.sort_values("organs_discarded", ascending=False),
        x="hospital_id",
        y="organs_discarded",
        title="Organs Discarded by Hospital",
    )
    _style_fig(fig_discard)

    detail = _styled_table(data=hospital.to_dict("records"), columns=list(hospital.columns))

    return html.Div(
        [
            dcc.Graph(figure=fig_rank_top, className="chart-panel"),
            dcc.Graph(figure=fig_rank_bottom, className="chart-panel"),
            dcc.Graph(figure=fig_scatter, className="chart-panel"),
            dcc.Graph(figure=fig_discard, className="chart-panel"),
            html.H4("Hospital Detail", className="section-title"),
            detail,
        ]
    )


def build_organ_tab(organ: pd.DataFrame) -> html.Div:
    if organ.empty:
        return _empty_message("No data in gold_organ_type_metrics.csv")

    organ = organ.copy()
    fig_mix = px.bar(
        organ,
        x="organ_type",
        y=["organs_total", "organs_placed", "organs_discarded"],
        barmode="group",
        title="Organ Type Mix and Outcomes",
    )
    _style_fig(fig_mix)

    fig_rate = px.bar(
        organ.sort_values("placement_rate", ascending=False),
        x="organ_type",
        y="placement_rate",
        title="Placement Rate by Organ Type",
    )
    _style_fig(fig_rate)
    fig_rate.update_yaxes(tickformat=".0%")

    fig_ischemia = px.bar(
        organ.sort_values("p90_cold_ischemia_minutes", ascending=False),
        x="organ_type",
        y=["avg_cold_ischemia_minutes", "p90_cold_ischemia_minutes"],
        barmode="group",
        title="Cold Ischemia (Average vs P90) by Organ Type",
    )
    _style_fig(fig_ischemia)

    detail = _styled_table(data=organ.to_dict("records"), columns=list(organ.columns))

    return html.Div(
        [
            dcc.Graph(figure=fig_mix, className="chart-panel"),
            dcc.Graph(figure=fig_rate, className="chart-panel"),
            dcc.Graph(figure=fig_ischemia, className="chart-panel"),
            html.H4("Organ-Type Detail", className="section-title"),
            detail,
        ]
    )


def build_blood_tab(blood: pd.DataFrame) -> html.Div:
    if blood.empty:
        return _empty_message("No data in gold_blood_type_metrics.csv")

    blood = blood.copy()

    fig_distribution = px.bar(
        blood.sort_values("referrals_count", ascending=False),
        x="blood_type",
        y="referrals_count",
        title="Referral Distribution by Blood Type",
    )
    _style_fig(fig_distribution)

    fig_outcomes = px.bar(
        blood,
        x="blood_type",
        y=["organs_total", "organs_placed", "organs_discarded"],
        barmode="group",
        title="Outcome Counts by Blood Type",
    )
    _style_fig(fig_outcomes)

    fig_rate = px.bar(
        blood.sort_values("placement_rate", ascending=False),
        x="blood_type",
        y="placement_rate",
        title="Placement Rate by Blood Type",
    )
    _style_fig(fig_rate)
    fig_rate.update_yaxes(tickformat=".0%")

    detail = _styled_table(data=blood.to_dict("records"), columns=list(blood.columns))

    return html.Div(
        [
            dcc.Graph(figure=fig_distribution, className="chart-panel"),
            dcc.Graph(figure=fig_outcomes, className="chart-panel"),
            dcc.Graph(figure=fig_rate, className="chart-panel"),
            html.H4("Blood-Type Detail", className="section-title"),
            detail,
        ]
    )


def build_rejection_tab(rejection: pd.DataFrame) -> html.Div:
    if rejection.empty:
        return _empty_message(
            "No data in gold_rejection_reason_metrics.csv (generate quarantine reasons first)."
        )

    rejection = rejection.copy()
    source_label_map = {
        "outcome": "Placement Outcomes",
        "referral": "Referrals",
    }
    rejection["source_table"] = rejection["domain"].map(source_label_map).fillna(rejection["domain"])
    fig_reasons = px.bar(
        rejection.sort_values("count", ascending=False),
        x="rejection_reason",
        y="count",
        color="source_table",
        title="Top Rejection Reasons",
    )
    _style_fig(fig_reasons)
    fig_reasons.update_xaxes(tickangle=25)
    fig_reasons.update_layout(legend_title_text="Source Table")

    fig_domain = px.bar(
        rejection.groupby("source_table", as_index=False)["count"].sum(),
        x="source_table",
        y="count",
        title="Rejections by Domain",
    )
    _style_fig(fig_domain)
    fig_domain.update_xaxes(title_text="Source Table")

    detail = _styled_table(data=rejection.to_dict("records"), columns=list(rejection.columns))

    return html.Div(
        [
            dcc.Graph(figure=fig_reasons, className="chart-panel"),
            dcc.Graph(figure=fig_domain, className="chart-panel"),
            html.H4("Rejection Detail", className="section-title"),
            detail,
        ]
    )


def build_quality_tab(quality: pd.DataFrame) -> html.Div:
    if quality.empty:
        return _empty_message("No data in gold_quality_metrics.csv")

    quality = quality.copy()
    quality["value"] = pd.to_numeric(quality["value"], errors="coerce").fillna(0.0)
    metric_map = dict(zip(quality["metric"], quality["value"]))

    cards = html.Div(
        [
            _card("Referrals In", f"{int(metric_map.get('referrals_in', 0)):,}"),
            _card("Referrals Rejected", f"{int(metric_map.get('referrals_rejected', 0)):,}"),
            _card("Outcomes In", f"{int(metric_map.get('outcomes_in', 0)):,}"),
            _card("Outcomes Rejected", f"{int(metric_map.get('outcomes_rejected', 0)):,}"),
            _card(
                "Outcomes Missing FK Rate",
                _fmt_rate(metric_map.get("outcomes_missing_fk_rate", 0.0)),
            ),
        ],
        className="metric-grid",
    )

    fig_quality = px.bar(
        quality.sort_values("metric"),
        x="metric",
        y="value",
        title="Quality Metrics",
    )
    _style_fig(fig_quality)
    fig_quality.update_xaxes(tickangle=25)

    detail = _styled_table(data=quality.to_dict("records"), columns=list(quality.columns))

    return html.Div([cards, dcc.Graph(figure=fig_quality, className="chart-panel"), html.H4("Quality Detail", className="section-title"), detail])


def build_drilldown_tab(silver_outcomes: pd.DataFrame) -> html.Div:
    if silver_outcomes.empty:
        return _empty_message("No data in silver_placement_outcomes.csv")

    silver_outcomes = silver_outcomes.copy()
    if "referral_date" in silver_outcomes.columns:
        silver_outcomes["referral_date"] = pd.to_datetime(
            silver_outcomes["referral_date"], errors="coerce"
        ).dt.date

    organ_options = sorted(
        [x for x in silver_outcomes.get("organ_type", pd.Series(dtype="object")).dropna().unique()]
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Organ Type"),
                            dcc.Dropdown(
                                id="drill-organ-filter",
                                options=[{"label": x, "value": x} for x in organ_options],
                                multi=True,
                                placeholder="All organ types",
                            ),
                        ],
                        className="filter-block",
                    ),
                    html.Div(
                        [
                            html.Label("Placement Status"),
                            dcc.Dropdown(
                                id="drill-placement-filter",
                                options=[
                                    {"label": "Placed", "value": "placed"},
                                    {"label": "Discarded", "value": "discarded"},
                                ],
                                multi=True,
                                placeholder="All statuses",
                            ),
                        ],
                        className="filter-block",
                    ),
                ],
                className="filter-row",
            ),
            _styled_table(
                table_id="drill-table",
                data=silver_outcomes.to_dict("records"),
                columns=list(silver_outcomes.columns),
            ),
            dcc.Store(id="silver-outcomes-store", data=silver_outcomes.to_dict("records")),
        ]
    )


def build_predictive_tab(
    funnel: pd.DataFrame,
    missed: pd.DataFrame,
    propensity: pd.DataFrame,
    discard_delay: pd.DataFrame,
    forecast: pd.DataFrame,
    offline_forecast: pd.DataFrame,
    offline_metrics: pd.DataFrame,
) -> html.Div:
    if (
        propensity.empty
        and forecast.empty
        and funnel.empty
        and missed.empty
        and discard_delay.empty
        and offline_forecast.empty
        and offline_metrics.empty
    ):
        return _empty_message(
            "No predictive tables found. Re-run pipeline to generate new gold_* predictive metrics."
        )

    lead_children: list = []
    children: list = []

    if not propensity.empty:
        propensity = propensity.copy()
        propensity["predicted_donor_probability"] = pd.to_numeric(
            propensity["predicted_donor_probability"], errors="coerce"
        )
        if "heuristic_donor_probability" in propensity.columns:
            propensity["heuristic_donor_probability"] = pd.to_numeric(
                propensity["heuristic_donor_probability"], errors="coerce"
            )
        if "regression_donor_probability" in propensity.columns:
            propensity["regression_donor_probability"] = pd.to_numeric(
                propensity["regression_donor_probability"], errors="coerce"
            )
        if {"heuristic_donor_probability", "regression_donor_probability"}.issubset(
            propensity.columns
        ):
            compare_df = propensity.copy()
            compare_df = compare_df.sort_values(
                "predicted_donor_probability", ascending=False
            ).reset_index(drop=True)
            compare_df["rank"] = compare_df.index + 1
            compare_top = compare_df.head(10).copy()
            compare_top["referral_label"] = compare_top["referral_id"].astype(str)
            compare_long = compare_top.melt(
                id_vars=["rank", "referral_id", "referral_label", "hospital_id", "donor_flag"],
                value_vars=["heuristic_donor_probability", "regression_donor_probability"],
                var_name="model",
                value_name="probability",
            )
            model_label_map = {
                "heuristic_donor_probability": "Heuristic",
                "regression_donor_probability": "Regression",
            }
            compare_long["model"] = compare_long["model"].map(model_label_map).fillna(compare_long["model"])

            fig_compare = px.bar(
                compare_long,
                x="referral_label",
                y="probability",
                color="model",
                hover_data=["referral_id", "hospital_id", "donor_flag"],
                barmode="group",
                title="Top 10 Referrals: Heuristic vs Regression Probability",
            )
            _style_fig(fig_compare)
            fig_compare.update_xaxes(title_text="Referral ID (Top 10 by Blended Score)", tickangle=45)
            fig_compare.update_yaxes(tickformat=".0%", range=[0, 1])
            children.append(dcc.Graph(figure=fig_compare, className="chart-panel"))
            children.append(
                html.Div(
                    "Interpretation: each referral shows two bars (Heuristic and Regression). "
                    "Referrals are ordered left-to-right by highest blended predicted probability.",
                    className="app-subtitle",
                )
            )
            children.extend(
                [
                    html.H4("Referral Propensity Detail (Top 10)", className="section-title"),
                    _styled_table(
                        data=compare_top.to_dict("records"),
                        columns=list(compare_top.columns),
                    ),
                ]
            )
        else:
            fallback_top = propensity.sort_values(
                "predicted_donor_probability", ascending=False
            ).head(10)
            children.extend(
                [
                    html.H4("Referral Propensity Detail (Top 10)", className="section-title"),
                    _styled_table(
                        data=fallback_top.to_dict("records"),
                        columns=list(fallback_top.columns),
                    ),
                ]
            )

    if not offline_forecast.empty:
        offline_forecast = offline_forecast.copy()
        offline_forecast["ds"] = pd.to_datetime(offline_forecast["ds"], errors="coerce")
        if "Historical (Last 30d)" not in offline_forecast.columns and {"ds", "y"}.issubset(
            offline_forecast.columns
        ):
            offline_forecast["Historical (Last 30d)"] = pd.NA

        if "ds" in offline_forecast.columns:
            fig_offline = go.Figure()
            plot_rows = offline_forecast.sort_values("ds")
            trace_specs = [
                ("Historical (Last 30d)", "#2563eb", "solid", 2),
                ("Actual", "#000000", "solid", 3),
                ("LightGBM", "#ea580c", "dash", 2),
                ("NHITS", "#10b981", "dash", 2),
                ("TFT", "#8b5cf6", "dash", 2),
            ]
            column_map = {
                "Historical (Last 30d)": "Historical (Last 30d)",
                "Actual": "y",
                "LightGBM": "LightGBM",
                "NHITS": "NHITS",
                "TFT": "TFT",
            }
            for trace_name, color, dash_style, width in trace_specs:
                col = column_map[trace_name]
                if col not in plot_rows.columns:
                    continue
                series_rows = plot_rows[["ds", col]].dropna()
                if series_rows.empty:
                    continue
                fig_offline.add_trace(
                    go.Scatter(
                        x=series_rows["ds"],
                        y=series_rows[col],
                        mode="lines+markers",
                        name=trace_name,
                        line={"color": color, "dash": dash_style, "width": width},
                    )
                )

            fig_offline.update_layout(
                title="SOTA Model Comparison: Daily Referral Forecasting"
            )
            _style_fig(fig_offline)
            fig_offline.update_layout(legend_title_text="")
            lead_children.append(dcc.Graph(figure=fig_offline, className="chart-panel"))

    children.append(
        html.Div(
            [
                html.H3("Predictive Analytics Command Center", className="section-title"),
                html.Div(
                    "Uses referral propensity scoring, baseline gap analysis, discard-delay patterns, and 14-day operational forecasting.",
                    className="app-subtitle",
                ),
            ],
            className="doc-panel",
        )
    )

    if not offline_metrics.empty:
        offline_metrics = offline_metrics.copy()
        if "mae" in offline_metrics.columns:
            offline_metrics["mae"] = pd.to_numeric(offline_metrics["mae"], errors="coerce")
            fig_offline_metrics = px.bar(
                offline_metrics.sort_values("mae", ascending=True),
                x="model",
                y="mae",
                title="Offline Model MAE Comparison",
                text_auto=".2f",
            )
            _style_fig(fig_offline_metrics)
            children.append(dcc.Graph(figure=fig_offline_metrics, className="chart-panel"))
        children.extend(
            [
                html.H4("Offline Forecast Detail", className="section-title"),
                _styled_table(
                    data=offline_metrics.to_dict("records"),
                    columns=list(offline_metrics.columns),
                ),
            ]
        )

    if not forecast.empty:
        forecast = forecast.copy()
        forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"], errors="coerce")
        fig_forecast = px.line(
            forecast.sort_values("forecast_date"),
            x="forecast_date",
            y=["predicted_referrals_count", "predicted_donor_referrals"],
            title="14-Day Operational Forecast",
            markers=True,
        )
        _style_fig(fig_forecast)
        children.append(dcc.Graph(figure=fig_forecast, className="chart-panel"))

    if not missed.empty:
        missed = missed.copy()
        fig_missed = px.bar(
            missed.sort_values("donor_gap_vs_baseline", ascending=False),
            x="hospital_id",
            y="donor_gap_vs_baseline",
            color="underperforming_flag",
            title="Missed Opportunity Gap by Hospital (vs Baseline)",
        )
        _style_fig(fig_missed)
        children.append(dcc.Graph(figure=fig_missed, className="chart-panel"))

    if not funnel.empty:
        funnel = funnel.copy()
        funnel["referral_date"] = pd.to_datetime(funnel["referral_date"], errors="coerce")
        fig_funnel = px.line(
            funnel.sort_values("referral_date"),
            x="referral_date",
            y=["referrals_count", "donor_referrals"],
            title="Referral to Donor Funnel Trend",
            markers=True,
        )
        _style_fig(fig_funnel)
        children.append(dcc.Graph(figure=fig_funnel, className="chart-panel"))

    if not discard_delay.empty:
        discard_delay = discard_delay.copy()
        fig_discard_delay = px.bar(
            discard_delay,
            x="delay_bucket",
            y="discard_rate",
            title="Discard Rate by Cold Ischemia Delay Bucket",
        )
        _style_fig(fig_discard_delay)
        fig_discard_delay.update_yaxes(tickformat=".0%")
        children.append(dcc.Graph(figure=fig_discard_delay, className="chart-panel"))

    return html.Div(lead_children + children)


def build_documentation_tab(documentation_text: str) -> html.Div:
    return html.Div(
        dcc.Markdown(documentation_text, className="doc-markdown"),
        className="doc-panel",
    )


def make_app(data: dict[str, pd.DataFrame], project_root: Path) -> dash.Dash:
    url_base_pathname = os.getenv("DASH_URL_BASE_PATHNAME", "/")
    if not url_base_pathname.startswith("/"):
        url_base_pathname = "/" + url_base_pathname
    if not url_base_pathname.endswith("/"):
        url_base_pathname = url_base_pathname + "/"

    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str((project_root / "assets").resolve()),
        assets_url_path="/assets",
        url_base_pathname=url_base_pathname,
    )
    app.title = "Organ Donor Analytics Dashboards"
    css_path = project_root / "assets" / "dashboard.css"
    css_text = ""
    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
    documentation_text = load_documentation(project_root)

    app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{css_text}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""

    ui_version = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(className="bg-orb bg-orb-a"),
                    html.Div(className="bg-orb bg-orb-b"),
                    html.Div(
                        [
                            html.H2("Organ Donor Analytics Dashboards", className="app-title"),
                            html.Div(
                                "Powered by Gold and Silver medallion tables",
                                className="app-subtitle",
                            ),
                            html.Div(
                                [
                                    "End to end pipeline build by ",
                                    html.Strong("Shck Tchamna"),
                                    ", from syntetic data generation to dashboard",
                                ],
                                className="app-credit",
                            ),
                            html.Div(
                                f"UI Build: {ui_version}",
                                className="app-build-stamp",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Theme", className="theme-label"),
                            dcc.RadioItems(
                                id="theme-toggle",
                                options=[
                                    {"label": "Light", "value": "theme-light"},
                                    {"label": "Dark", "value": "theme-dark"},
                                ],
                                value="theme-dark",
                                inline=True,
                                className="theme-toggle",
                                inputClassName="theme-input",
                                labelClassName="theme-option",
                            ),
                        ],
                        className="theme-row",
                    ),
                    dcc.Tabs(
                        id="main-tabs",
                        value="tab-overview",
                        className="main-tabs",
                        children=[
                            dcc.Tab(label="Daily Operations", value="tab-overview", className="tab", selected_className="tab-selected"),
                            dcc.Tab(label="Hospital Benchmarking", value="tab-hospital", className="tab", selected_className="tab-selected"),
                            dcc.Tab(label="Organ Type Mix", value="tab-organ", className="tab", selected_className="tab-selected"),
                            dcc.Tab(label="Blood Type", value="tab-blood", className="tab", selected_className="tab-selected"),
                            dcc.Tab(label="Rejection Root Cause", value="tab-rejection", className="tab", selected_className="tab-selected"),
                            dcc.Tab(label="Data Quality", value="tab-quality", className="tab", selected_className="tab-selected"),
                            dcc.Tab(label="Case Drilldown", value="tab-drill", className="tab", selected_className="tab-selected"),
                            dcc.Tab(label="Predictive Analytics", value="tab-predictive", className="tab predictive-tab", selected_className="tab-selected predictive-tab-selected"),
                            dcc.Tab(label="Documentation", value="tab-docs", className="tab", selected_className="tab-selected"),
                        ],
                    ),
                    html.Div(id="tab-content", className="tab-content"),
                    dcc.Store(id="daily-store", data=data["daily"].to_dict("records")),
                    dcc.Store(id="hospital-store", data=data["hospital"].to_dict("records")),
                    dcc.Store(id="organ-store", data=data["organ"].to_dict("records")),
                    dcc.Store(id="blood-store", data=data["blood"].to_dict("records")),
                    dcc.Store(id="rejection-store", data=data["rejection"].to_dict("records")),
                    dcc.Store(id="quality-store", data=data["quality"].to_dict("records")),
                    dcc.Store(id="funnel-store", data=data["funnel"].to_dict("records")),
                    dcc.Store(id="missed-store", data=data["missed"].to_dict("records")),
                    dcc.Store(id="propensity-store", data=data["propensity"].to_dict("records")),
                    dcc.Store(id="discard-delay-store", data=data["discard_delay"].to_dict("records")),
                    dcc.Store(id="forecast-store", data=data["forecast"].to_dict("records")),
                    dcc.Store(id="offline-forecast-store", data=data["offline_forecast"].to_dict("records")),
                    dcc.Store(id="offline-metrics-store", data=data["offline_metrics"].to_dict("records")),
                    dcc.Store(id="silver-store", data=data["silver_outcomes"].to_dict("records")),
                ],
                id="theme-root",
                className="app-shell theme-dark",
            ),
        ],
        id="theme-page",
        className="app-page theme-dark",
    )

    @app.callback(
        Output("theme-root", "className"),
        Output("theme-page", "className"),
        Input("theme-toggle", "value"),
    )
    def switch_theme(theme_class: str) -> tuple[str, str]:
        if theme_class not in {"theme-light", "theme-dark"}:
            theme_class = "theme-light"
        return f"app-shell {theme_class}", f"app-page {theme_class}"

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
        Input("daily-store", "data"),
        Input("hospital-store", "data"),
        Input("organ-store", "data"),
        Input("blood-store", "data"),
        Input("rejection-store", "data"),
        Input("quality-store", "data"),
        Input("funnel-store", "data"),
        Input("missed-store", "data"),
        Input("propensity-store", "data"),
        Input("discard-delay-store", "data"),
        Input("forecast-store", "data"),
        Input("offline-forecast-store", "data"),
        Input("offline-metrics-store", "data"),
        Input("silver-store", "data"),
    )
    def render_tab(
        tab_value: str,
        daily_data: list[dict],
        hospital_data: list[dict],
        organ_data: list[dict],
        blood_data: list[dict],
        rejection_data: list[dict],
        quality_data: list[dict],
        funnel_data: list[dict],
        missed_data: list[dict],
        propensity_data: list[dict],
        discard_delay_data: list[dict],
        forecast_data: list[dict],
        offline_forecast_data: list[dict],
        offline_metrics_data: list[dict],
        silver_data: list[dict],
    ) -> html.Div:
        daily = pd.DataFrame(daily_data)
        hospital = pd.DataFrame(hospital_data)
        organ = pd.DataFrame(organ_data)
        blood = pd.DataFrame(blood_data)
        rejection = pd.DataFrame(rejection_data)
        quality = pd.DataFrame(quality_data)
        funnel = pd.DataFrame(funnel_data)
        missed = pd.DataFrame(missed_data)
        propensity = pd.DataFrame(propensity_data)
        discard_delay = pd.DataFrame(discard_delay_data)
        forecast = pd.DataFrame(forecast_data)
        offline_forecast = pd.DataFrame(offline_forecast_data)
        offline_metrics = pd.DataFrame(offline_metrics_data)
        silver = pd.DataFrame(silver_data)

        if tab_value == "tab-overview":
            return build_overview_tab(daily)
        if tab_value == "tab-hospital":
            return build_hospital_tab(hospital)
        if tab_value == "tab-organ":
            return build_organ_tab(organ)
        if tab_value == "tab-blood":
            return build_blood_tab(blood)
        if tab_value == "tab-rejection":
            return build_rejection_tab(rejection)
        if tab_value == "tab-quality":
            return build_quality_tab(quality)
        if tab_value == "tab-predictive":
            return build_predictive_tab(
                funnel,
                missed,
                propensity,
                discard_delay,
                forecast,
                offline_forecast,
                offline_metrics,
            )
        if tab_value == "tab-docs":
            return build_documentation_tab(documentation_text)
        return build_drilldown_tab(silver)

    @app.callback(
        Output("drill-table", "data"),
        Input("drill-organ-filter", "value"),
        Input("drill-placement-filter", "value"),
        Input("silver-outcomes-store", "data"),
        prevent_initial_call=True,
    )
    def filter_drill_table(
        organ_values: list[str] | None,
        placement_values: list[str] | None,
        base_rows: list[dict],
    ) -> list[dict]:
        df = pd.DataFrame(base_rows)
        if df.empty:
            return []

        if organ_values:
            df = df[df["organ_type"].isin(organ_values)]

        if placement_values:
            status_mask = pd.Series([False] * len(df), index=df.index)
            if "placed" in placement_values and "placed_flag" in df.columns:
                status_mask = status_mask | (pd.to_numeric(df["placed_flag"], errors="coerce") == 1)
            if "discarded" in placement_values and "discard_flag" in df.columns:
                status_mask = status_mask | (pd.to_numeric(df["discard_flag"], errors="coerce") == 1)
            df = df[status_mask]

        return df.to_dict("records")

    return app


def create_runtime_app(project_root: Path | None = None) -> dash.Dash:
    root = project_root or Path(os.getenv("PROJECT_ROOT", str(default_project_root()))).resolve()
    data = load_data(root)
    return make_app(data, root)


# Gunicorn entrypoint: `gunicorn -w 2 -b 127.0.0.1:8050 scripts.dash_dashboard:server`
app = create_runtime_app()
server = app.server


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    runtime_app = create_runtime_app(project_root)
    runtime_app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
