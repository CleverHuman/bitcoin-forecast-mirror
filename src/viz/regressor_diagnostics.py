"""Diagnostic visualizations for Bitcoin forecasting regressors.

Provides plots for debugging regressor behavior:
- All regressor values over time
- Correlation heatmap between regressors
- Regressor values vs actual price returns
- Residuals colored by cycle phase
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from src.models.cycle_features import add_cycle_features, CyclePhase


def plot_regressor_timeseries(
    df: pd.DataFrame,
    date_col: str = "ds",
    price_col: str = "y",
    save_path: str | None = None,
    figsize: tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Plot all regressor values over time on the same chart.

    Args:
        df: DataFrame with date, price, and regressor columns.
        date_col: Name of date column.
        price_col: Name of price column.
        save_path: Path to save figure (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    regressor_cols = [
        ("reg_cycle_sin", "Cycle Sin", "tab:blue"),
        ("reg_cycle_cos", "Cycle Cos", "tab:cyan"),
        ("reg_pre_halving", "Pre-Halving Weight", "tab:green"),
        ("reg_post_halving", "Post-Halving Weight", "tab:orange"),
        ("double_top_regressor", "Double Top", "tab:red"),
        ("cycle_phase_regressor", "Cycle Phase", "tab:purple"),
        ("decay_regressor", "Decay", "tab:brown"),
    ]

    available = [(col, name, color) for col, name, color in regressor_cols if col in df.columns]

    if not available:
        print("No regressor columns found in DataFrame")
        return None

    n_regressors = len(available)
    fig, axes = plt.subplots(n_regressors + 1, 1, figsize=figsize, sharex=True)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Price plot (top)
    ax_price = axes[0]
    if price_col in df.columns:
        ax_price.semilogy(df[date_col], df[price_col], color="black", linewidth=1)
        ax_price.set_ylabel("Price (log)")
        ax_price.set_title("BTC Price with Regressor Values Over Time")
        ax_price.grid(True, alpha=0.3)

    # Add halving date vertical lines
    from src.metrics import HALVING_DATES
    for h in HALVING_DATES:
        for ax in axes:
            ax.axvline(h, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Regressor plots
    for i, (col, name, color) in enumerate(available):
        ax = axes[i + 1]
        ax.plot(df[date_col], df[col], color=color, linewidth=1)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

        # Add zero line for reference
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

        # Highlight problematic regions (extreme values)
        if col in ["cycle_phase_regressor", "decay_regressor"]:
            extreme_mask = (df[col].abs() > 1) | (df[col].isna())
            if extreme_mask.any():
                ax.scatter(
                    df.loc[extreme_mask, date_col],
                    df.loc[extreme_mask, col].fillna(0),
                    color="red", s=10, alpha=0.5, label="Extreme/NaN"
                )
                ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_regressor_correlation(
    df: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot correlation heatmap between regressors.

    High correlation (> 0.5) suggests redundancy.

    Args:
        df: DataFrame with regressor columns.
        save_path: Path to save figure (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    regressor_cols = [
        "reg_cycle_sin", "reg_cycle_cos", "reg_pre_halving", "reg_post_halving",
        "double_top_regressor", "cycle_phase_regressor", "decay_regressor",
    ]

    available = [c for c in regressor_cols if c in df.columns]

    if len(available) < 2:
        print("Need at least 2 regressor columns for correlation matrix")
        return None

    corr_matrix = df[available].corr()

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    if SEABORN_AVAILABLE:
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
    else:
        # Fallback without seaborn
        im = ax.imshow(np.ma.array(corr_matrix.values, mask=mask), cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(available)))
        ax.set_yticks(range(len(available)))
        ax.set_xticklabels(available, rotation=45, ha="right")
        ax.set_yticklabels(available)
        # Add text annotations
        for i in range(len(available)):
            for j in range(len(available)):
                if not mask[i, j]:
                    ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                            ha="center", va="center", fontsize=8)

    ax.set_title("Regressor Correlation Matrix\n(|corr| > 0.5 suggests redundancy)")

    # Highlight high correlations
    for i in range(len(available)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="red", linewidth=2
                ))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_regressor_vs_returns(
    df: pd.DataFrame,
    date_col: str = "ds",
    price_col: str = "y",
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot regressor values vs actual price returns (scatter).

    Helps identify if regressors correlate with future returns.

    Args:
        df: DataFrame with date, price, and regressor columns.
        date_col: Name of date column.
        price_col: Name of price column.
        save_path: Path to save figure (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    regressor_cols = [
        ("reg_pre_halving", "Pre-Halving Weight"),
        ("reg_post_halving", "Post-Halving Weight"),
        ("double_top_regressor", "Double Top"),
        ("cycle_phase_regressor", "Cycle Phase"),
        ("decay_regressor", "Decay"),
    ]

    available = [(col, name) for col, name in regressor_cols if col in df.columns]

    if not available or price_col not in df.columns:
        print("Missing required columns")
        return None

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Calculate forward returns (30-day, 90-day)
    df["return_30d"] = df[price_col].pct_change(30).shift(-30) * 100
    df["return_90d"] = df[price_col].pct_change(90).shift(-90) * 100

    n_cols = 2
    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (col, name) in enumerate(available):
        for j, (return_col, return_name) in enumerate([
            ("return_30d", "30-day Return"),
            ("return_90d", "90-day Return")
        ]):
            ax = axes[i, j]

            # Remove NaN values
            mask = df[col].notna() & df[return_col].notna()
            x = df.loc[mask, col]
            y = df.loc[mask, return_col]

            if len(x) > 0:
                ax.scatter(x, y, alpha=0.3, s=5)

                # Add trend line
                if len(x) > 10:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.7)

                    # Calculate correlation
                    corr = np.corrcoef(x, y)[0, 1]
                    ax.text(
                        0.05, 0.95, f"r = {corr:.3f}",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment="top"
                    )

            ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
            ax.axvline(0, color="gray", linestyle="-", alpha=0.3)

            if j == 0:
                ax.set_ylabel(f"{name}\n{return_name} (%)")
            else:
                ax.set_ylabel(f"{return_name} (%)")

            if i == 0:
                ax.set_title(return_name)

            if i == n_rows - 1:
                ax.set_xlabel("Regressor Value")

            ax.grid(True, alpha=0.3)

    fig.suptitle("Regressor Values vs Forward Returns", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_residuals_by_phase(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    date_col: str = "ds",
    price_col: str = "y",
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Plot forecast residuals colored by cycle phase.

    Helps identify if certain phases have systematic errors.

    Args:
        df: Actual data with date and price columns.
        forecast: Forecast DataFrame with yhat column.
        date_col: Name of date column.
        price_col: Name of price column.
        save_path: Path to save figure (optional).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    df = df.copy()
    forecast = forecast.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    forecast[date_col] = pd.to_datetime(forecast[date_col])

    # Add cycle features to actual data
    df = add_cycle_features(df, date_col=date_col)

    # Merge actual with forecast
    merged = df.merge(
        forecast[[date_col, "yhat", "yhat_ensemble"]],
        on=date_col,
        how="inner"
    )

    if merged.empty:
        print("No overlapping data between actual and forecast")
        return None

    # Calculate residuals (% error)
    yhat_col = "yhat_ensemble" if "yhat_ensemble" in merged.columns else "yhat"
    merged["residual_pct"] = (merged[yhat_col] - merged[price_col]) / merged[price_col] * 100

    # Color by cycle phase
    phase_colors = {
        CyclePhase.ACCUMULATION.value: "tab:blue",
        CyclePhase.PRE_HALVING_RUNUP.value: "tab:green",
        CyclePhase.POST_HALVING_CONSOLIDATION.value: "tab:orange",
        CyclePhase.BULL_RUN.value: "tab:red",
        CyclePhase.DISTRIBUTION.value: "tab:purple",
        CyclePhase.DRAWDOWN.value: "tab:brown",
    }

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot 1: Residuals over time, colored by phase
    ax1 = axes[0]
    for phase, color in phase_colors.items():
        mask = merged["cycle_phase"] == phase
        if mask.any():
            ax1.scatter(
                merged.loc[mask, date_col],
                merged.loc[mask, "residual_pct"],
                c=color, s=10, alpha=0.5, label=phase.replace("_", " ").title()
            )

    ax1.axhline(0, color="black", linestyle="-", linewidth=1)
    ax1.set_ylabel("Residual (%)")
    ax1.set_title("Forecast Residuals by Cycle Phase")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Add halving lines
    from src.metrics import HALVING_DATES
    for h in HALVING_DATES:
        ax1.axvline(h, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Plot 2: Box plot of residuals by phase
    ax2 = axes[1]
    phase_order = [
        CyclePhase.ACCUMULATION.value,
        CyclePhase.PRE_HALVING_RUNUP.value,
        CyclePhase.POST_HALVING_CONSOLIDATION.value,
        CyclePhase.BULL_RUN.value,
        CyclePhase.DISTRIBUTION.value,
        CyclePhase.DRAWDOWN.value,
    ]

    # Filter to phases that exist in data
    existing_phases = [p for p in phase_order if p in merged["cycle_phase"].values]

    box_data = [
        merged.loc[merged["cycle_phase"] == phase, "residual_pct"].dropna()
        for phase in existing_phases
    ]

    bp = ax2.boxplot(
        box_data,
        labels=[p.replace("_", " ").title() for p in existing_phases],
        patch_artist=True
    )

    for i, phase in enumerate(existing_phases):
        bp["boxes"][i].set_facecolor(phase_colors.get(phase, "gray"))
        bp["boxes"][i].set_alpha(0.5)

    ax2.axhline(0, color="black", linestyle="-", linewidth=1)
    ax2.set_ylabel("Residual (%)")
    ax2.set_xlabel("Cycle Phase")
    ax2.grid(True, alpha=0.3, axis="y")

    # Calculate mean residual per phase
    for i, phase in enumerate(existing_phases):
        phase_data = merged.loc[merged["cycle_phase"] == phase, "residual_pct"]
        mean_resid = phase_data.mean()
        ax2.text(
            i + 1, ax2.get_ylim()[1] * 0.9,
            f"μ={mean_resid:.1f}%",
            ha="center", fontsize=8
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def create_diagnostic_report(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    output_dir: str = "diagnostics",
) -> dict[str, Any]:
    """Generate a full diagnostic report with all visualizations.

    Args:
        df: Actual data with date, price, and regressor columns.
        forecast: Forecast DataFrame.
        output_dir: Directory to save plots.

    Returns:
        Dict with paths to generated files and summary statistics.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {"plots": [], "stats": {}}

    # 1. Regressor timeseries
    fig = plot_regressor_timeseries(
        df,
        save_path=os.path.join(output_dir, "regressor_timeseries.png")
    )
    if fig:
        results["plots"].append("regressor_timeseries.png")
        plt.close(fig)

    # 2. Correlation heatmap
    fig = plot_regressor_correlation(
        df,
        save_path=os.path.join(output_dir, "regressor_correlation.png")
    )
    if fig:
        results["plots"].append("regressor_correlation.png")
        plt.close(fig)

        # Save correlation stats
        regressor_cols = [c for c in df.columns if "reg" in c or "regressor" in c]
        if len(regressor_cols) >= 2:
            corr = df[regressor_cols].corr()
            # Find high correlations
            high_corr = []
            for i in range(len(regressor_cols)):
                for j in range(i + 1, len(regressor_cols)):
                    c = corr.iloc[i, j]
                    if abs(c) > 0.5:
                        high_corr.append((regressor_cols[i], regressor_cols[j], c))
            results["stats"]["high_correlations"] = high_corr

    # 3. Regressor vs returns
    fig = plot_regressor_vs_returns(
        df,
        save_path=os.path.join(output_dir, "regressor_vs_returns.png")
    )
    if fig:
        results["plots"].append("regressor_vs_returns.png")
        plt.close(fig)

    # 4. Residuals by phase
    fig = plot_residuals_by_phase(
        df, forecast,
        save_path=os.path.join(output_dir, "residuals_by_phase.png")
    )
    if fig:
        results["plots"].append("residuals_by_phase.png")
        plt.close(fig)

    # Summary stats
    regressor_cols = [c for c in df.columns if "reg" in c or "regressor" in c]
    stats = {}
    for col in regressor_cols:
        stats[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "nan_pct": df[col].isna().mean() * 100,
        }
    results["stats"]["regressor_summary"] = stats

    print(f"\nDiagnostic report saved to {output_dir}/")
    print(f"Generated {len(results['plots'])} plots")

    return results


if __name__ == "__main__":
    # Example usage
    print("Run from main script with data loaded")
