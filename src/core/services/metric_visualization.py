"""
Metric Visualization Utilities
Generate plots and visualizations for monitoring dashboards
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class MetricVisualizer:
    """
    Comprehensive metric visualization utilities

    Generates interactive Plotly visualizations for:
    - Time series metrics
    - Confusion matrices
    - ROC/PR curves
    - Performance trends
    - Degradation alerts
    """

    @staticmethod
    def plot_metric_trend(
        timestamps: List[datetime],
        values: List[float],
        metric_name: str,
        sensor_id: str,
        baseline: Optional[float] = None,
        degradation_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
    ) -> go.Figure:
        """
        Plot metric trend over time

        Args:
            timestamps: List of timestamps
            values: Metric values
            metric_name: Name of metric
            sensor_id: Sensor identifier
            baseline: Baseline value (optional)
            degradation_threshold: Warning threshold (optional)
            critical_threshold: Critical threshold (optional)

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Main trend line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                name=metric_name,
                line=dict(color="#2E86AB", width=2),
                marker=dict(size=6),
            )
        )

        # Baseline line
        if baseline is not None:
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color="green",
                annotation_text="Baseline",
                annotation_position="right",
            )

            # Degradation threshold
            if degradation_threshold is not None:
                threshold_value = baseline * (1 - degradation_threshold)
                fig.add_hline(
                    y=threshold_value,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Warning",
                    annotation_position="right",
                )

            # Critical threshold
            if critical_threshold is not None:
                critical_value = baseline * (1 - critical_threshold)
                fig.add_hline(
                    y=critical_value,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Critical",
                    annotation_position="right",
                )

        fig.update_layout(
            title=f"{metric_name} Trend - {sensor_id}",
            xaxis_title="Time",
            yaxis_title=metric_name,
            hovermode="x unified",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray, class_names: Optional[List[str]] = None, normalize: bool = True
    ) -> go.Figure:
        """
        Plot confusion matrix heatmap

        Args:
            cm: Confusion matrix
            class_names: Class names (optional)
            normalize: Whether to normalize

        Returns:
            Plotly figure
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]

        # Normalize if requested
        if normalize:
            cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            text_template = "%{z:.2%}"
        else:
            cm_display = cm
            text_template = "%{z:d}"

        fig = go.Figure(
            data=go.Heatmap(
                z=cm_display,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                text=cm,
                texttemplate=text_template,
                textfont={"size": 16},
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="True",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, sensor_id: str = "") -> go.Figure:
        """
        Plot ROC curve

        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under curve
            sensor_id: Sensor identifier

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC (AUC = {roc_auc:.3f})",
                line=dict(color="#2E86AB", width=2),
            )
        )

        # Random classifier diagonal
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="gray", width=1, dash="dash"),
            )
        )

        fig.update_layout(
            title=f"ROC Curve - {sensor_id}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )

        return fig

    @staticmethod
    def plot_precision_recall_curve(
        precision: np.ndarray, recall: np.ndarray, pr_auc: float, sensor_id: str = ""
    ) -> go.Figure:
        """
        Plot Precision-Recall curve

        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under curve
            sensor_id: Sensor identifier

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"PR Curve (AUC = {pr_auc:.3f})",
                line=dict(color="#A23B72", width=2),
            )
        )

        fig.update_layout(
            title=f"Precision-Recall Curve - {sensor_id}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )

        return fig

    @staticmethod
    def plot_multi_metric_comparison(
        metrics_data: Dict[str, Tuple[List[datetime], List[float]]],
        sensor_id: str,
        title: str = "Multi-Metric Comparison",
    ) -> go.Figure:
        """
        Plot multiple metrics on the same chart

        Args:
            metrics_data: Dictionary mapping metric_name to (timestamps, values)
            sensor_id: Sensor identifier
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

        for idx, (metric_name, (timestamps, values)) in enumerate(metrics_data.items()):
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode="lines+markers",
                    name=metric_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                )
            )

        fig.update_layout(
            title=f"{title} - {sensor_id}",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    @staticmethod
    def plot_forecast_accuracy_by_horizon(
        mae_by_step: List[float],
        rmse_by_step: List[float],
        mape_by_step: List[float],
        sensor_id: str,
    ) -> go.Figure:
        """
        Plot forecast accuracy by time step

        Args:
            mae_by_step: MAE for each forecast step
            rmse_by_step: RMSE for each forecast step
            mape_by_step: MAPE for each forecast step
            sensor_id: Sensor identifier

        Returns:
            Plotly figure
        """
        steps = list(range(1, len(mae_by_step) + 1))

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("MAE by Step", "RMSE by Step", "MAPE by Step"),
        )

        # MAE
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mae_by_step,
                mode="lines+markers",
                name="MAE",
                line=dict(color="#2E86AB", width=2),
            ),
            row=1,
            col=1,
        )

        # RMSE
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=rmse_by_step,
                mode="lines+markers",
                name="RMSE",
                line=dict(color="#A23B72", width=2),
            ),
            row=1,
            col=2,
        )

        # MAPE
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mape_by_step,
                mode="lines+markers",
                name="MAPE (%)",
                line=dict(color="#F18F01", width=2),
            ),
            row=1,
            col=3,
        )

        fig.update_xaxes(title_text="Forecast Step", row=1, col=1)
        fig.update_xaxes(title_text="Forecast Step", row=1, col=2)
        fig.update_xaxes(title_text="Forecast Step", row=1, col=3)

        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=3)

        fig.update_layout(
            title=f"Forecast Accuracy by Horizon - {sensor_id}",
            template="plotly_white",
            showlegend=False,
            height=400,
        )

        return fig

    @staticmethod
    def plot_alert_timeline(alerts: List[Dict[str, Any]], days_back: int = 30) -> go.Figure:
        """
        Plot alert timeline

        Args:
            alerts: List of alert dictionaries
            days_back: Days to display

        Returns:
            Plotly figure
        """
        if not alerts:
            fig = go.Figure()
            fig.add_annotation(
                text="No alerts in selected period",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            return fig

        # Group by severity
        critical = [a for a in alerts if a["severity"] == "critical"]
        warning = [a for a in alerts if a["severity"] == "warning"]
        info = [a for a in alerts if a["severity"] == "info"]

        fig = go.Figure()

        # Plot critical alerts
        if critical:
            timestamps = [datetime.fromisoformat(a["timestamp"]) for a in critical]
            degradations = [a["degradation_pct"] for a in critical]
            sensors = [a["sensor_id"] for a in critical]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=degradations,
                    mode="markers",
                    name="Critical",
                    marker=dict(size=12, color="red", symbol="x"),
                    text=sensors,
                    hovertemplate="<b>%{text}</b><br>Degradation: %{y:.1f}%<br>%{x}<extra></extra>",
                )
            )

        # Plot warnings
        if warning:
            timestamps = [datetime.fromisoformat(a["timestamp"]) for a in warning]
            degradations = [a["degradation_pct"] for a in warning]
            sensors = [a["sensor_id"] for a in warning]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=degradations,
                    mode="markers",
                    name="Warning",
                    marker=dict(size=10, color="orange", symbol="circle"),
                    text=sensors,
                    hovertemplate="<b>%{text}</b><br>Degradation: %{y:.1f}%<br>%{x}<extra></extra>",
                )
            )

        # Plot info
        if info:
            timestamps = [datetime.fromisoformat(a["timestamp"]) for a in info]
            degradations = [a["degradation_pct"] for a in info]
            sensors = [a["sensor_id"] for a in info]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=degradations,
                    mode="markers",
                    name="Info",
                    marker=dict(size=8, color="blue", symbol="circle"),
                    text=sensors,
                    hovertemplate="<b>%{text}</b><br>Degradation: %{y:.1f}%<br>%{x}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Performance Alerts - Last {days_back} Days",
            xaxis_title="Time",
            yaxis_title="Degradation (%)",
            hovermode="closest",
            template="plotly_white",
        )

        return fig

    @staticmethod
    def plot_model_health_summary(
        summary_data: Dict[str, Dict[str, float]], metric_name: str = "accuracy"
    ) -> go.Figure:
        """
        Plot health summary for multiple models

        Args:
            summary_data: Dictionary mapping model_key to metrics
            metric_name: Metric to display

        Returns:
            Plotly figure
        """
        models = list(summary_data.keys())
        current_values = [data.get(f"current_{metric_name}", 0) for data in summary_data.values()]
        baseline_values = [data.get(f"baseline_{metric_name}", 0) for data in summary_data.values()]

        fig = go.Figure(
            data=[
                go.Bar(name="Current", x=models, y=current_values, marker_color="#2E86AB"),
                go.Bar(
                    name="Baseline",
                    x=models,
                    y=baseline_values,
                    marker_color="lightgray",
                ),
            ]
        )

        fig.update_layout(
            title=f"Model Health Summary - {metric_name}",
            xaxis_title="Model",
            yaxis_title=metric_name.capitalize(),
            barmode="group",
            template="plotly_white",
        )

        return fig
