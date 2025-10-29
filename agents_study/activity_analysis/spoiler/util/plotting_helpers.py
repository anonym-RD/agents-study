"""Reusable plotting utilities for standard plot types in our style."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from typing import Optional, Tuple
from pathlib import Path


cur_path = Path(__file__).parent
root_path = cur_path.parent.parent
plots_root = root_path / "plots"


def sort_agents_human_first(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a dataframe so Human comes first, then others alphabetically."""
    human_row = df.filter(pl.col("agent") == "Human")
    other_rows = df.filter(pl.col("agent") != "Human").sort("agent")
    return pl.concat([human_row, other_rows])


def wilson_score_interval(successes, trials, confidence=0.95):
    """Calculate Wilson score confidence interval for a proportion.
    
    References:
        Wilson, E. B. (1927). Probable inference, the law of succession, 
        and statistical inference. Journal of the American Statistical 
        Association, 22(158), 209-212.
        
        Agresti, A., & Coull, B. A. (1998). Approximate is better than 
        "exact" for interval estimation of binomial proportions. 
        The American Statistician, 52(2), 119-126.
    """
    if trials == 0:
        return 0.0, 0.0
    
    p = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    
    return lower, upper


def plot_simple_bar(
    df: pl.DataFrame,
    value_col: str,
    ylabel: str,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    with_confidence_intervals: bool = False,
    count_col: Optional[str] = None,
    baseline_label: str = "Human baseline",
    value_format: str = "{:.1f}",
    value_label_offset_factor: float = 0.02,
    ylim: Optional[Tuple[float, float]] = None,
    legend_loc: str = 'upper right',
    ylabel_fontsize: Optional[int] = None,
    xticklabel_rotation: float = 45,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a simple bar chart with our standard style.
    
    Args:
        df: DataFrame with 'agent' column and value column
        value_col: Name of column with values to plot
        ylabel: Label for y-axis
        title: Plot title (optional, no title if None)
        figsize: Figure size (width, height)
        with_confidence_intervals: If True, add Wilson score confidence intervals
        count_col: Column name for total counts (required if with_confidence_intervals=True)
        baseline_label: Label for Human baseline horizontal line
        value_format: Format string for value labels (default: "{:.1f}")
        value_label_offset_factor: Offset factor for value labels as fraction of max value
        ylim: Optional tuple of (ymin, ymax) to set y-axis limits
        legend_loc: Location for legend (default: 'upper right')
        ylabel_fontsize: Optional font size for y-axis label (overrides default)
        xticklabel_rotation: Rotation angle for x-axis tick labels in degrees (default: 45)
    
    Returns:
        Figure and axes objects
    """
    # Sort with Human first
    df = sort_agents_human_first(df)
    
    agents = df['agent'].to_list()
    values = df[value_col].to_list()
    
    # Human baseline
    human_value = values[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(agents))
    width = 0.6
    
    # Color scheme: Human red, others teal
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Plot bars with optional confidence intervals
    if with_confidence_intervals and count_col is not None:
        counts = df[count_col].to_list()
        
        # Calculate Wilson score confidence intervals
        ci_lower = []
        ci_upper = []
        for value, count in zip(values, counts):
            # Assume values are percentages, convert to proportion
            proportion = value / 100.0
            successes = int(proportion * count)
            lower, upper = wilson_score_interval(successes, count)
            ci_lower.append(lower * 100)
            ci_upper.append(upper * 100)
        
        # Calculate error bar sizes
        yerr_lower = [val - lower for val, lower in zip(values, ci_lower)]
        yerr_upper = [upper - val for val, upper in zip(values, ci_upper)]
        
        bars = ax.bar(x, values, width, color=colors, alpha=0.8,
                     yerr=[yerr_lower, yerr_upper], capsize=5, error_kw={'linewidth': 2})
        
        # Add value labels accounting for error bars
        for i, (bar, val, upper_err) in enumerate(zip(bars, values, yerr_upper)):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + upper_err + max(values) * value_label_offset_factor,
                   value_format.format(val), ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
    else:
        bars = ax.bar(x, values, width, color=colors, alpha=0.8)
        
        # Add value labels on top of bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + max(values) * value_label_offset_factor,
                   value_format.format(val), ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
    
    # Add horizontal line for Human baseline
    ax.axhline(y=human_value, color='#FF6B6B', linestyle='--', linewidth=2, 
               alpha=0.5, label=baseline_label)
    
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xlabel('Agent')
    if title is not None:
        ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    
    # Set x-axis tick labels with configurable rotation
    if xticklabel_rotation == 0:
        ax.set_xticklabels(agents, rotation=0, ha='center')
    else:
        ax.set_xticklabels(agents, rotation=xticklabel_rotation, ha='right')
    
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return fig, ax


def plot_box_distribution(
    df: pl.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    figsize: Tuple[float, float] = (8, 6),
    use_log_scale: bool = False,
    baseline_label: str = "Human median",
    percentiles: Tuple[float, float, float, float, float] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a custom box plot (with p10-p90 whiskers) in our standard style.
    
    Args:
        df: DataFrame with 'agent' column and value column
        value_col: Name of column with values to plot
        ylabel: Label for y-axis
        title: Plot title
        figsize: Figure size (width, height)
        use_log_scale: If True, use log scale for y-axis
        baseline_label: Label for Human baseline horizontal line
        percentiles: Tuple of (p10, p25, p50, p75, p90) percentiles to plot
    
    Returns:
        Figure and axes objects
    """
    # Get agent order with Human first
    agents_order = ['Human'] + sorted([a for a in df['agent'].unique().to_list() if a != 'Human'])
    
    # Calculate percentiles for each agent
    p10, p25, p50, p75, p90 = percentiles
    percentile_stats = df.group_by("agent").agg([
        pl.col(value_col).quantile(p10).alias("p10"),
        pl.col(value_col).quantile(p25).alias("p25"),
        pl.col(value_col).quantile(p50).alias("p50"),
        pl.col(value_col).quantile(p75).alias("p75"),
        pl.col(value_col).quantile(p90).alias("p90"),
    ])
    
    # Sort with Human first
    percentile_stats = sort_agents_human_first(percentile_stats)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = np.arange(len(agents_order))
    
    # Draw custom box plots
    for i, agent in enumerate(agents_order):
        agent_stats = percentile_stats.filter(pl.col("agent") == agent).to_dicts()[0]
        
        p10_val = agent_stats['p10']
        p25_val = agent_stats['p25']
        p50_val = agent_stats['p50']
        p75_val = agent_stats['p75']
        p90_val = agent_stats['p90']
        
        # Color scheme
        if agent == 'Human':
            box_color = '#FF6B6B'
            whisker_color = '#CC5555'
        else:
            box_color = '#4ECDC4'
            whisker_color = '#3EBAB0'
        
        # Draw box (IQR: p25 to p75)
        box_width = 0.5
        box = plt.Rectangle((i - box_width/2, p25_val), box_width, p75_val - p25_val,
                           facecolor=box_color, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        
        # Draw median line
        ax.plot([i - box_width/2, i + box_width/2], [p50_val, p50_val], 
               color='black', linewidth=2.5, zorder=3)
        
        # Draw whisker caps (p10 and p90)
        cap_width = 0.3
        # P10 cap
        ax.plot([i - cap_width/2, i + cap_width/2], [p10_val, p10_val],
               color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
        # P90 cap
        ax.plot([i - cap_width/2, i + cap_width/2], [p90_val, p90_val],
               color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
    
    # Human baseline median
    human_median = percentile_stats.filter(pl.col("agent") == "Human")["p50"].to_list()[0]
    ax.axhline(y=human_median, color='#FF6B6B', linestyle='--', linewidth=2, 
               alpha=0.4, label=baseline_label, zorder=0)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Agent')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(agents_order, rotation=45, ha='right')
    
    if use_log_scale:
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y', which='both')
    else:
        ax.grid(True, alpha=0.3, axis='y')
    
    ax.legend()
    
    return fig, ax


def save_plot(fig: plt.Figure, output_path, dpi: int = 300):
    """Save a plot to both PNG and PDF formats.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Path without extension (e.g., 'plots/my_plot')
        dpi: DPI for PNG output
    """
    fig.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    fig.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved to {output_path}.png and .pdf")

