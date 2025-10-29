"""Reusable utilities for creating stratified analyses by repository star counts.

This module provides a framework for creating stratified plots that break down
metrics by repository star count bins. It abstracts common operations like:
- Loading and joining PR data with repository data
- Computing star count bins based on quantiles
- Creating multi-panel plots with consistent styling
"""
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Any, List
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents
from spoiler.util.plotting_helpers import sort_agents_human_first, wilson_score_interval


def load_pr_repo_data() -> pl.DataFrame:
    """Load and join PR and Repository data with merge status.
    
    Returns:
        DataFrame with PR data joined to base repository data, including:
        - is_merged: boolean indicating if PR was merged
        - time_to_merge_hours: hours from creation to merge (null for unmerged)
        - stargazer_count: star count of the base repository
        - All original PR and repository columns
    """
    print("Loading PR data...")
    prs_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Loading Repository data...")
    repos_lazy = load_lazy_table_for_all_agents(TableNames.REPOSITORIES)
    
    # Filter repos to only base repositories (the target repo of the PR)
    print("Filtering to BASE repositories...")
    base_repos = repos_lazy.filter(pl.col("role") == "BASE")
    
    # Join PRs with their base repositories
    print("Joining PRs with repositories...")
    pr_repo_data = prs_lazy.join(
        base_repos,
        left_on=["id", "agent"],
        right_on=["pr_id", "agent"],
        how="left"
    )
    
    # Add merge status and time to merge
    pr_repo_data = pr_repo_data.with_columns([
        pl.col("merged_at").is_not_null().alias("is_merged"),
        # Parse datetime columns
        pl.col("created_at").str.to_datetime(),
        pl.col("merged_at").str.to_datetime(),
        # Time to merge (in hours) - only for merged PRs
        ((pl.col("merged_at").str.to_datetime() - pl.col("created_at").str.to_datetime())
            .dt.total_seconds() / 3600.0)
            .alias("time_to_merge_hours"),
    ])
    
    print("Collecting data...")
    return pr_repo_data.collect()


def compute_star_bins(
    df: pl.DataFrame,
    n_bins: int = 3,
    quantile_agent: str = "Human"
) -> Tuple[pl.DataFrame, List[float], List[str]]:
    """Compute star count bins based on quantiles of a reference agent's data.
    
    Uses Polars' cut() function to create bins with human-readable labels.
    The bins are computed based on the quantiles of a reference agent (default: Human)
    but are then applied uniformly to all agents' data.
    
    Args:
        df: DataFrame with stargazer_count column
        n_bins: Number of bins to create (default: 3 for tertiles)
        quantile_agent: Agent to use for computing quantile boundaries (default: "Human")
    
    Returns:
        tuple: (df with star_bin column, list of bin edges, list of bin labels)
    """
    print(f"\nComputing {n_bins} star bins based on {quantile_agent} data...")
    
    # Get the reference agent's star counts
    ref_data = df.filter(pl.col("agent") == quantile_agent)
    
    # Compute quantiles for bin edges (interior boundaries only, not min/max)
    quantiles = [i / n_bins for i in range(1, n_bins)]
    
    if len(quantiles) > 0:
        bin_edges = ref_data.select(
            [pl.col("stargazer_count").quantile(q).alias(f"q{q}") for q in quantiles]
        ).row(0)
    else:
        bin_edges = []
    
    print(f"  Bin edges (interior boundaries, based on {quantile_agent} quantiles): {[int(x) for x in bin_edges]}")
    
    # Get min and max for display purposes
    min_stars = ref_data.select(pl.col("stargazer_count").min()).item()
    max_stars = ref_data.select(pl.col("stargazer_count").max()).item()
    
    # Create bin labels
    bin_labels = []
    categories = ["Low", "Medium", "High", "Very High", "Extreme"]  # Extended for more bins
    
    for i in range(n_bins):
        if i == 0:
            lower_rounded = int(min_stars)
            upper_rounded = int(bin_edges[0]) if len(bin_edges) > 0 else int(min_stars)
        elif i < n_bins - 1:
            lower_rounded = int(bin_edges[i-1]) + 1
            upper_rounded = int(bin_edges[i])
        else:
            lower_rounded = int(bin_edges[-1]) + 1
            upper_rounded = int(max_stars)
        
        category = categories[i] if i < len(categories) else f"Bin{i+1}"
        bin_labels.append(f"{category} — [{lower_rounded:,}, {upper_rounded:,}] Stars")
    
    print(f"  Bin labels: {bin_labels}")
    
    # Assign bins using cut
    df = df.with_columns([
        pl.col("stargazer_count").cut(
            breaks=list(bin_edges),
            labels=bin_labels,
            left_closed=False,
            include_breaks=False
        ).alias("star_bin")
    ])
    
    # Check for unassigned PRs
    unassigned = df.filter(pl.col("star_bin").is_null())
    if len(unassigned) > 0:
        print(f"  Warning: {len(unassigned)} PRs were not assigned to bins")
    
    # Print bin distributions by agent
    print("\n  PRs per bin by agent:")
    bin_dist = df.group_by(["agent", "star_bin"]).agg([
        pl.len().alias("count")
    ]).sort(["agent", "star_bin"])
    print(bin_dist)
    
    return df, bin_edges, bin_labels


class StratifiedPlotConfig:
    """Configuration for stratified plots."""
    
    def __init__(
        self,
        title: str,
        ylabel: str,
        figsize_per_bin: float = 6.0,
        fig_height: float = 4.5,
        show_distribution: bool = False,
        use_log_scale: bool = False,
        human_color: str = '#FF6B6B',
        other_color: str = '#4ECDC4',
        include_all_stars: bool = False,
        spacer_width_ratio: float = 1.0,
    ):
        """Initialize plot configuration.
        
        Args:
            title: Overall figure title
            ylabel: Label for y-axis
            figsize_per_bin: Width in inches per bin subplot
            fig_height: Height of figure in inches (when show_distribution=False)
            show_distribution: If True, add histogram showing data distribution below
            use_log_scale: If True, use log scale for y-axis
            human_color: Color for Human bars/boxes
            other_color: Color for other agent bars/boxes
            include_all_stars: If True, add an "Overall" panel showing all data
            spacer_width_ratio: Width ratio for spacer between Overall and bins (default: 1.0)
        """
        self.title = title
        self.ylabel = ylabel
        self.figsize_per_bin = figsize_per_bin
        self.fig_height = fig_height
        self.show_distribution = show_distribution
        self.use_log_scale = use_log_scale
        self.human_color = human_color
        self.other_color = other_color
        self.include_all_stars = include_all_stars
        self.spacer_width_ratio = spacer_width_ratio


def setup_stratified_figure(
    n_bins: int,
    config: StratifiedPlotConfig
) -> Tuple[plt.Figure, List[plt.Axes], Optional[List[plt.Axes]]]:
    """Set up a stratified figure with optional distribution histograms.
    
    Args:
        n_bins: Number of bins (subplots)
        config: Plot configuration
    
    Returns:
        tuple: (figure, list of top axes, optional list of bottom axes)
    """
    if config.include_all_stars:
        # Layout: [Overall] [Spacer] [Bin1] [Bin2] [Bin3]
        # Width ratios: [5] [spacer] [5] [5] [5]
        n_cols = n_bins + 2  # +1 for overall, +1 for spacer
        width_ratios = [5.0, config.spacer_width_ratio] + [5.0] * n_bins
        total_width = config.figsize_per_bin * (n_bins + 1)  # Overall + n_bins
        
        if config.show_distribution:
            # 2 rows: main plots and histogram
            fig = plt.figure(figsize=(total_width, config.fig_height))
            gs = fig.add_gridspec(2, n_cols, height_ratios=[3, 1], width_ratios=width_ratios,
                                hspace=0.3, wspace=0.0)  # wspace=0, spacing controlled by spacer column
            
            # Create axes with shared y-axis for top row
            # Overall panel (col 0), then stratified panels (col 2+)
            axes_top = []
            axes_top.append(fig.add_subplot(gs[0, 0]))  # Overall
            for i in range(n_bins):
                axes_top.append(fig.add_subplot(gs[0, i + 2], sharey=axes_top[0]))
            
            # Create axes for bottom row (skip overall for distribution)
            axes_bottom = []
            axes_bottom.append(None)  # No distribution for Overall
            for i in range(n_bins):
                if i == 0:
                    axes_bottom.append(fig.add_subplot(gs[1, i + 2]))
                else:
                    axes_bottom.append(fig.add_subplot(gs[1, i + 2], sharey=axes_bottom[1]))
        else:
            # Single row: main plots only
            fig = plt.figure(figsize=(total_width, config.fig_height))
            gs = fig.add_gridspec(1, n_cols, width_ratios=width_ratios, wspace=0.0)  # wspace=0, spacing controlled by spacer column
            
            axes_top = []
            axes_top.append(fig.add_subplot(gs[0, 0]))  # Overall
            for i in range(n_bins):
                axes_top.append(fig.add_subplot(gs[0, i + 2], sharey=axes_top[0]))
            
            axes_bottom = None
    else:
        # Original behavior: no overall panel
        if config.show_distribution:
            # 2 rows: main plots and histogram
            fig = plt.figure(figsize=(config.figsize_per_bin * n_bins, config.fig_height))
            gs = fig.add_gridspec(2, n_bins, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            # Create axes with shared y-axis for top row
            axes_top = []
            for i in range(n_bins):
                if i == 0:
                    axes_top.append(fig.add_subplot(gs[0, i]))
                else:
                    axes_top.append(fig.add_subplot(gs[0, i], sharey=axes_top[0]))
            
            # Create axes for bottom row (also shared)
            axes_bottom = []
            for i in range(n_bins):
                if i == 0:
                    axes_bottom.append(fig.add_subplot(gs[1, i]))
                else:
                    axes_bottom.append(fig.add_subplot(gs[1, i], sharey=axes_bottom[0]))
        else:
            # Single row: main plots only
            fig, axes_top = plt.subplots(1, n_bins, figsize=(config.figsize_per_bin * n_bins, config.fig_height), sharey=True)
            if n_bins == 1:
                axes_top = [axes_top]  # Make it a list for consistency
            axes_bottom = None
    
    return fig, axes_top, axes_bottom


def add_distribution_histogram(
    ax: plt.Axes,
    agents: List[str],
    fractions: List[float],
    counts: List[int],
    is_first: bool = False
):
    """Add a distribution histogram to a subplot.
    
    Args:
        ax: Axes to plot on
        agents: List of agent names
        fractions: Fraction of each agent's total PRs in this bin
        counts: Raw counts for each agent in this bin
        is_first: If True, add y-axis label
    """
    positions = np.arange(len(agents))
    bars_hist = ax.bar(positions, fractions, color='#808080', alpha=0.6)
    
    # Add text labels with raw counts
    for bar, fraction, count in zip(bars_hist, fractions, counts):
        ax.text(bar.get_x() + bar.get_width()/2, 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Agent', fontsize=10)
    if is_first:
        ax.set_ylabel("Fraction of\nAgent's PRs", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([])  # No labels - they appear on top plot
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelleft=True)


def plot_stratified_bar_chart(
    df: pl.DataFrame,
    bin_labels: List[str],
    value_col: str,
    count_col: str,
    config: StratifiedPlotConfig,
    output_path: Path,
    with_confidence_intervals: bool = True,
    value_format: str = "{:.1f}%",
    ylim: Optional[Tuple[float, float]] = None,
    max_stars: Optional[int] = None,
) -> None:
    """Create a stratified bar chart (e.g., for merge rates, comment rates).
    
    Args:
        df: DataFrame with columns [agent, star_bin, value_col, count_col, total_col]
        bin_labels: List of bin labels
        value_col: Column name for values to plot (e.g., 'merge_rate')
        count_col: Column name for success counts (for confidence intervals)
        config: Plot configuration
        output_path: Path to save plots (without extension)
        with_confidence_intervals: If True, add Wilson score confidence intervals
        value_format: Format string for value labels
        ylim: Optional y-axis limits tuple (min, max)
    """
    print("\nCreating stratified bar chart...")
    
    # Calculate fractions for distribution plot
    total_per_agent = df.group_by("agent").agg([
        pl.col(count_col).sum().alias("agent_total")
    ])
    
    stats_by_bin = df.join(
        total_per_agent,
        on="agent",
        how="left"
    ).with_columns([
        (pl.col(count_col) / pl.col("agent_total")).alias("fraction")
    ])
    
    n_bins = len(bin_labels)
    agents_order = ['Human'] + sorted([a for a in df['agent'].unique().to_list() if a != 'Human'])
    
    # Compute overall stats (all data combined) if needed
    overall_stats = None
    overall_label = None
    if config.include_all_stars:
        # Use provided max_stars or try to get from data
        stars_max = max_stars if max_stars is not None else (
            df.select(pl.col("stargazer_count").max()).item() 
            if "stargazer_count" in df.columns else 0
        )
        overall_label = f"Overall — [0, {int(stars_max):,}] Stars"
        
        # Aggregate overall stats (ignoring star_bin)
        overall_stats = df.group_by("agent").agg([
            pl.col(count_col).sum().alias(count_col),
            pl.col("total").sum().alias("total") if "total" in df.columns else pl.len().alias("total"),
        ])
        
        # Recalculate the value_col for overall
        if "total" in overall_stats.columns:
            overall_stats = overall_stats.with_columns([
                (pl.col(count_col) / pl.col("total") * 100).alias(value_col)
            ])
    
    # Set up figure
    fig, axes_top, axes_bottom = setup_stratified_figure(n_bins, config)
    
    # Colors
    agent_colors = {agent: config.human_color if agent == 'Human' else config.other_color 
                    for agent in agents_order}
    
    # Helper function to plot a bar chart panel
    def plot_bar_panel(ax, data, label, is_first_panel):
        data = sort_agents_human_first(data)
        agents = data['agent'].to_list()
        values = data[value_col].to_list()
        counts = data[count_col].to_list()
        totals = data['total'].to_list() if 'total' in data.columns else counts
        
        colors = [agent_colors[agent] for agent in agents]
        
        if with_confidence_intervals:
            # Calculate Wilson score confidence intervals
            ci_lower = []
            ci_upper = []
            for count, total in zip(counts, totals):
                lower, upper = wilson_score_interval(count, total)
                ci_lower.append(lower * 100)
                ci_upper.append(upper * 100)
            
            yerr_lower = [val - lower for val, lower in zip(values, ci_lower)]
            yerr_upper = [upper - val for val, upper in zip(values, ci_upper)]
            
            bars = ax.bar(agents, values,
                         yerr=[yerr_lower, yerr_upper],
                         color=colors, alpha=0.8, capsize=5, error_kw={'linewidth': 2})
            
            # Add value labels
            for bar, val, upper_err in zip(bars, values, yerr_upper):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 1,
                       value_format.format(val), ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
        else:
            bars = ax.bar(agents, values, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                offset = (ylim[1] - ylim[0]) * 0.02 if ylim else max(values) * 0.02
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                       value_format.format(val), ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
        
        # Human baseline
        if "Human" in agents:
            human_value = values[agents.index("Human")]
            ax.axhline(y=human_value, color=config.human_color, linestyle='--', 
                      linewidth=2, alpha=0.5, zorder=0)
        
        # Formatting
        if is_first_panel:
            ax.set_ylabel(config.ylabel, fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(range(len(agents)))
        ax.grid(True, alpha=0.3, axis='y')
        
        if ylim:
            ax.set_ylim(*ylim)
        
        return agents
    
    # Plot overall panel if requested
    start_idx = 0
    if config.include_all_stars and overall_stats is not None:
        agents = plot_bar_panel(axes_top[0], overall_stats, overall_label, is_first_panel=True)
        # Overall panel always shows unrotated labels (no distribution below it)
        axes_top[0].set_xticklabels(agents)
        start_idx = 1
    
    # Plot each stratified bin
    bottom_iter = axes_bottom if axes_bottom is not None else [None] * (n_bins + start_idx)
    for bin_idx, (ax_top, ax_bottom, bin_label) in enumerate(zip(axes_top[start_idx:], bottom_iter[start_idx:], bin_labels)):
        # Get data for this bin
        bin_data = stats_by_bin.filter(pl.col("star_bin") == bin_label)
        
        # Use helper to plot bar panel
        is_first = (bin_idx == 0 and start_idx == 0)
        agents = plot_bar_panel(ax_top, bin_data, bin_label, is_first_panel=is_first)
        
        # Hide y-tick labels on non-first stratified bins (when wspace is 0)
        if config.include_all_stars and bin_idx > 0:
            ax_top.tick_params(labelleft=False)
        
        # Get fractions for distribution plot
        fractions = bin_data.filter(pl.col("agent").is_in(agents))
        fractions = sort_agents_human_first(fractions)
        fractions_list = fractions['fraction'].to_list()
        counts_list = [int(x) for x in fractions[count_col].to_list()]
        
        if config.show_distribution:
            # Keep unrotated labels on top plot (between top and bottom)
            ax_top.set_xticklabels(agents)
            # === BOTTOM PLOT: Distribution ===
            if ax_bottom is not None:
                add_distribution_histogram(ax_bottom, agents, fractions_list, 
                                          counts_list, is_first=(bin_idx == 0))
                # Hide y-tick labels on non-first bins
                if config.include_all_stars and bin_idx > 0:
                    ax_bottom.tick_params(labelleft=False)
        else:
            # No distribution, keep labels unrotated (they fit now)
            ax_top.set_xticklabels(agents)
    
    # Overall title
    fig.suptitle(config.title, fontsize=16, fontweight='bold', y=1.0)
    if config.show_distribution:
        fig.subplots_adjust(top=0.85)
    else:
        fig.subplots_adjust(top=0.80)
    
    # Save
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {output_path}.png and .pdf")


def plot_stratified_box_chart(
    df: pl.DataFrame,
    bin_labels: List[str],
    value_col: str,
    config: StratifiedPlotConfig,
    output_path: Path,
    percentiles: Tuple[float, float, float, float, float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    max_stars: Optional[int] = None,
) -> None:
    """Create a stratified box plot (e.g., for time to merge, comment counts).
    
    Args:
        df: DataFrame with columns [agent, star_bin, value_col] (one row per PR)
        bin_labels: List of bin labels
        value_col: Column name for values to plot (e.g., 'time_to_merge_hours')
        config: Plot configuration
        output_path: Path to save plots (without extension)
        percentiles: Tuple of (p10, p25, p50, p75, p90) percentiles to plot
    """
    print("\nCreating stratified box plot...")
    
    n_bins = len(bin_labels)
    agents_order = ['Human'] + sorted([a for a in df['agent'].unique().to_list() if a != 'Human'])
    
    # Calculate percentiles for each agent × bin combination
    p10, p25, p50, p75, p90 = percentiles
    percentile_stats = df.group_by(["agent", "star_bin"]).agg([
        pl.len().alias("count"),
        pl.col(value_col).quantile(p10).alias("p10"),
        pl.col(value_col).quantile(p25).alias("p25"),
        pl.col(value_col).quantile(p50).alias("p50"),
        pl.col(value_col).quantile(p75).alias("p75"),
        pl.col(value_col).quantile(p90).alias("p90"),
    ])
    
    # Calculate fractions for distribution plot
    total_per_agent = df.group_by("agent").agg([
        pl.len().alias("agent_total")
    ])
    
    percentile_stats = percentile_stats.join(
        total_per_agent,
        on="agent",
        how="left"
    ).with_columns([
        (pl.col("count") / pl.col("agent_total")).alias("fraction")
    ])
    
    # Compute overall stats (all data combined) if needed
    overall_stats = None
    overall_label = None
    if config.include_all_stars:
        # Use provided max_stars or try to get from data
        stars_max = max_stars if max_stars is not None else (
            df.select(pl.col("stargazer_count").max()).item() 
            if "stargazer_count" in df.columns else 0
        )
        overall_label = f"Overall — [0, {int(stars_max):,}] Stars"
        
        # Aggregate overall stats (ignoring star_bin)
        overall_stats = df.group_by("agent").agg([
            pl.len().alias("count"),
            pl.col(value_col).quantile(p10).alias("p10"),
            pl.col(value_col).quantile(p25).alias("p25"),
            pl.col(value_col).quantile(p50).alias("p50"),
            pl.col(value_col).quantile(p75).alias("p75"),
            pl.col(value_col).quantile(p90).alias("p90"),
        ])
    
    # Set up figure
    fig, axes_top, axes_bottom = setup_stratified_figure(n_bins, config)
    
    # Helper function to plot a box plot panel
    def plot_box_panel(ax, data, label, is_first_panel):
        data = sort_agents_human_first(data)
        agents = data['agent'].to_list()
        positions = np.arange(len(agents))
        
        for i, agent in enumerate(agents):
            agent_stats = data.filter(pl.col("agent") == agent).to_dicts()[0]
            
            p10_val = agent_stats['p10']
            p25_val = agent_stats['p25']
            p50_val = agent_stats['p50']
            p75_val = agent_stats['p75']
            p90_val = agent_stats['p90']
            
            # Colors
            if agent == 'Human':
                box_color = config.human_color
                whisker_color = '#CC5555'
            else:
                box_color = config.other_color
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
            ax.plot([i - cap_width/2, i + cap_width/2], [p10_val, p10_val],
                   color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
            ax.plot([i - cap_width/2, i + cap_width/2], [p90_val, p90_val],
                   color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
        
        # Human baseline median
        human_data = data.filter(pl.col("agent") == "Human")
        if len(human_data) > 0:
            human_median = human_data["p50"].to_list()[0]
            ax.axhline(y=human_median, color=config.human_color, linestyle='--', 
                      linewidth=2, alpha=0.4, zorder=0)
        
        # Formatting
        if is_first_panel:
            ax.set_ylabel(config.ylabel, fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(positions)
        
        if config.use_log_scale:
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, axis='y', which='both')
        else:
            ax.grid(True, alpha=0.3, axis='y')
        
        ax.tick_params(labelleft=True)
        
        return agents
    
    # Plot overall panel if requested
    start_idx = 0
    if config.include_all_stars and overall_stats is not None:
        agents = plot_box_panel(axes_top[0], overall_stats, overall_label, is_first_panel=True)
        # Overall panel always shows unrotated labels (no distribution below it)
        axes_top[0].set_xticklabels(agents)
        start_idx = 1
    
    # Plot each stratified bin
    bottom_iter = axes_bottom if axes_bottom is not None else [None] * (n_bins + start_idx)
    for bin_idx, (ax_top, ax_bottom, bin_label) in enumerate(zip(axes_top[start_idx:], bottom_iter[start_idx:], bin_labels)):
        # Get data for this bin
        bin_data = percentile_stats.filter(pl.col("star_bin") == bin_label)
        
        # Use helper to plot box panel
        is_first = (bin_idx == 0 and start_idx == 0)
        agents = plot_box_panel(ax_top, bin_data, bin_label, is_first_panel=is_first)
        
        # Hide y-tick labels on non-first stratified bins (when wspace is 0)
        if config.include_all_stars and bin_idx > 0:
            ax_top.tick_params(labelleft=False)
        
        # Get fractions and counts for distribution plot
        bin_data = sort_agents_human_first(bin_data)
        counts = bin_data['count'].to_list()
        fractions = bin_data['fraction'].to_list()
        
        if config.show_distribution:
            # Keep unrotated labels on top plot (between top and bottom)
            ax_top.set_xticklabels(agents)
            # === BOTTOM PLOT: Distribution ===
            if ax_bottom is not None:
                add_distribution_histogram(ax_bottom, agents, fractions, counts, 
                                          is_first=(bin_idx == 0))
                # Hide y-tick labels on non-first bins
                if config.include_all_stars and bin_idx > 0:
                    ax_bottom.tick_params(labelleft=False)
        else:
            # No distribution, keep labels unrotated (they fit now)
            ax_top.set_xticklabels(agents)
    
    # Overall title
    fig.suptitle(config.title, fontsize=16, fontweight='bold', y=1.0)
    if config.show_distribution:
        fig.subplots_adjust(top=0.85)
    else:
        fig.subplots_adjust(top=0.80)
    
    # Save
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {output_path}.png and .pdf")


def run_stratified_analysis(
    data_loader: Callable[[], pl.DataFrame],
    aggregator: Callable[[pl.DataFrame], pl.DataFrame],
    plot_type: str,  # 'bar' or 'box'
    config: StratifiedPlotConfig,
    output_path: Path,
    n_bins: int = 3,
    quantile_agent: str = "Human",
    **plot_kwargs: Any
) -> None:
    """Run a complete stratified analysis pipeline.
    
    This is the main entry point for creating stratified analyses. It handles:
    1. Loading data
    2. Computing star bins
    3. Aggregating metrics
    4. Creating plots
    
    Args:
        data_loader: Function that returns a DataFrame with 'agent' and 'stargazer_count'
        aggregator: Function that takes binned data and returns aggregated statistics
        plot_type: Either 'bar' for bar charts or 'box' for box plots
        config: Plot configuration
        output_path: Path to save plots (without extension)
        n_bins: Number of star count bins (default: 3)
        quantile_agent: Agent to use for bin boundaries (default: "Human")
        **plot_kwargs: Additional arguments to pass to the plotting function
    """
    print("="*80)
    print(f"STRATIFIED ANALYSIS: {config.title.upper()}")
    print("="*80)
    print(f"Configuration: {n_bins} bins based on {quantile_agent} quantiles")
    
    # Load data
    df = data_loader()
    print(f"\nLoaded {len(df):,} records")
    
    # Store max stars for Overall panel label (before aggregation)
    max_stars = None
    if config.include_all_stars and "stargazer_count" in df.columns:
        max_stars = df.select(pl.col("stargazer_count").max()).item()
    
    # Compute star bins
    df, bin_edges, bin_labels = compute_star_bins(df, n_bins=n_bins, quantile_agent=quantile_agent)
    
    # Aggregate statistics
    stats = aggregator(df)
    
    # Create plot
    if plot_type == 'bar':
        plot_stratified_bar_chart(stats, bin_labels, config=config, output_path=output_path, 
                                 max_stars=max_stars, **plot_kwargs)
    elif plot_type == 'box':
        plot_stratified_box_chart(df, bin_labels, config=config, output_path=output_path, 
                                 max_stars=max_stars, **plot_kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Must be 'bar' or 'box'")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Plot saved to: {output_path}")

