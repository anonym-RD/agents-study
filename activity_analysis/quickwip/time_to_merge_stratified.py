"""Generate time-to-merge analysis stratified by repository star counts.

NOTE: This script has been superseded by ../stratified_core_metrics.py, which uses
the new plotting_stratified framework and combines both merge rate and time-to-merge
analyses. This original script is kept for reference.
"""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents
from spoiler.util.plotting_helpers import plots_root


plots_path = plots_root / "time_to_merge_stratified"

# Create plots directory if it doesn't exist
plots_path.mkdir(parents=True, exist_ok=True)

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def sort_agents_human_first(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a dataframe so Human comes first, then others alphabetically."""
    human_row = df.filter(pl.col("agent") == "Human")
    other_rows = df.filter(pl.col("agent") != "Human").sort("agent")
    return pl.concat([human_row, other_rows])


def load_pr_repo_data():
    """Load and join PR and Repository data."""
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


def compute_star_bins(df: pl.DataFrame, n_bins: int = 3, quantile_agent: str = "Human"):
    """Compute star count bins based on quantiles of a reference agent's data.
    
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
    all_edges = [min_stars] + list(bin_edges) + [max_stars]
    
    categories = ["Low", "Medium", "High"]
    
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


def plot_time_to_merge_stratified(df: pl.DataFrame, bin_labels: list, show_distribution: bool = False):
    """Plot time-to-merge distribution stratified by repository star count bins.
    
    Creates a faceted plot with one subplot per star count bin, showing box plots
    of time to merge distribution (p10-p90).
    
    Args:
        df: DataFrame with PR and repository data
        bin_labels: List of bin labels for stratification
        show_distribution: If True, show bottom histogram of data distribution (default: False)
    """
    print("\nCreating stratified time-to-merge plot...")
    
    # Filter to merged PRs only
    merged_df = df.filter(pl.col("is_merged"))
    
    print(f"  Using {len(merged_df):,} merged PRs out of {len(df):,} total PRs")
    
    # Get agent order (Human first)
    agents_order = ['Human'] + sorted([a for a in merged_df['agent'].unique().to_list() if a != 'Human'])
    
    # Get unique bins
    bins = bin_labels
    n_bins = len(bins)
    
    # Calculate percentiles for each agent × bin combination
    percentile_stats = merged_df.group_by(["agent", "star_bin"]).agg([
        pl.len().alias("count"),
        pl.col("time_to_merge_hours").quantile(0.10).alias("p10"),
        pl.col("time_to_merge_hours").quantile(0.25).alias("p25"),
        pl.col("time_to_merge_hours").quantile(0.50).alias("p50"),
        pl.col("time_to_merge_hours").quantile(0.75).alias("p75"),
        pl.col("time_to_merge_hours").quantile(0.90).alias("p90"),
    ])
    
    # Calculate total per agent for fraction
    total_per_agent = merged_df.group_by("agent").agg([
        pl.len().alias("agent_total")
    ])
    
    # Join to get fractions
    percentile_stats = percentile_stats.join(
        total_per_agent,
        on="agent",
        how="left"
    ).with_columns([
        (pl.col("count") / pl.col("agent_total")).alias("fraction")
    ])
    
    # Create subplot layout
    if show_distribution:
        # 2 rows: box plots and histogram
        fig = plt.figure(figsize=(6 * n_bins, 4.5))
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
        # Single row: box plots only
        fig, axes_top = plt.subplots(1, n_bins, figsize=(6 * n_bins, 3), sharey=True)
        if n_bins == 1:
            axes_top = [axes_top]  # Make it a list for consistency
        axes_bottom = None
    
    # Plot each bin
    bottom_iter = axes_bottom if axes_bottom is not None else [None] * n_bins
    for bin_idx, (ax_top, ax_bottom, bin_label) in enumerate(zip(axes_top, bottom_iter, bins)):
        # Get data for this bin
        bin_data = percentile_stats.filter(pl.col("star_bin") == bin_label)
        
        # Sort agents with Human first
        bin_data = sort_agents_human_first(bin_data)
        
        # Extract data for plotting
        agents = bin_data['agent'].to_list()
        counts = bin_data['count'].to_list()
        fractions = bin_data['fraction'].to_list()
        
        # === TOP PLOT: Time to Merge Box Plots ===
        positions = np.arange(len(agents))
        
        for i, agent in enumerate(agents):
            agent_stats = bin_data.filter(pl.col("agent") == agent).to_dicts()[0]
            
            p10 = agent_stats['p10']
            p25 = agent_stats['p25']
            p50 = agent_stats['p50']
            p75 = agent_stats['p75']
            p90 = agent_stats['p90']
            
            # Color scheme
            if agent == 'Human':
                box_color = '#FF6B6B'
                whisker_color = '#CC5555'
            else:
                box_color = '#4ECDC4'
                whisker_color = '#3EBAB0'
            
            # Draw box (IQR: p25 to p75)
            box_width = 0.5
            box = plt.Rectangle((i - box_width/2, p25), box_width, p75 - p25,
                               facecolor=box_color, edgecolor='black', linewidth=1.5, alpha=0.7)
            ax_top.add_patch(box)
            
            # Draw median line
            ax_top.plot([i - box_width/2, i + box_width/2], [p50, p50], 
                   color='black', linewidth=2.5, zorder=3)
            
            # Draw whisker caps (p10 and p90)
            cap_width = 0.3
            ax_top.plot([i - cap_width/2, i + cap_width/2], [p10, p10],
                   color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
            ax_top.plot([i - cap_width/2, i + cap_width/2], [p90, p90],
                   color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
            
            # Dashed lines connecting caps to box
            ax_top.plot([i, i], [p10, p25], color=whisker_color, linewidth=0.8, 
                   linestyle=':', alpha=0.5, zorder=1)
            ax_top.plot([i, i], [p75, p90], color=whisker_color, linewidth=0.8, 
                   linestyle=':', alpha=0.5, zorder=1)
        
        # Human baseline median for this bin
        human_bin_data = bin_data.filter(pl.col("agent") == "Human")
        if len(human_bin_data) > 0:
            human_median = human_bin_data["p50"].to_list()[0]
            ax_top.axhline(y=human_median, color='#FF6B6B', linestyle='--', 
                          linewidth=2, alpha=0.4, zorder=0)
        
        # Formatting top plot
        if bin_idx == 0:
            ax_top.set_ylabel('Time to Merge (hours, log scale)', fontsize=12)
        ax_top.set_title(bin_label, fontsize=13, fontweight='bold', pad=12)
        ax_top.set_xticks(positions)
        ax_top.set_yscale('log')
        ax_top.grid(True, alpha=0.3, axis='y', which='both')
        # Enable y-axis tick labels on all subplots
        ax_top.tick_params(labelleft=True)
        
        if show_distribution:
            # No tick labels on top plot when showing distribution below
            ax_top.set_xticklabels([])
            
            # === BOTTOM PLOT: Data Distribution (Histogram) ===
            bars_hist = ax_bottom.bar(positions, fractions, color='#808080', alpha=0.6)
            
            # Add text labels with raw counts
            for bar, fraction, count in zip(bars_hist, fractions, counts):
                ax_bottom.text(bar.get_x() + bar.get_width()/2, 0.01,
                              f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Formatting bottom plot
            ax_bottom.set_xlabel('Agent', fontsize=10)
            if bin_idx == 0:
                ax_bottom.set_ylabel("Fraction of\nAgent's PRs", fontsize=10)
            ax_bottom.set_xticks(positions)
            ax_bottom.set_xticklabels(agents, rotation=0, ha='right')
            ax_bottom.grid(True, alpha=0.3, axis='y')
            # Enable y-axis tick labels on all subplots
            ax_bottom.tick_params(labelleft=True)
        else:
            # Show agent labels on top plot when no distribution plot
            ax_top.set_xticklabels(agents, rotation=45, ha='right')
    
    # Overall title
    fig.suptitle('Time to Merge by Repository Star Count', fontsize=16, fontweight='bold', y=1.0)
    if show_distribution:
        fig.subplots_adjust(top=0.85)
    else:
        fig.subplots_adjust(top=0.80)
    
    plt.savefig(plots_path / 'time_to_merge_stratified.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'time_to_merge_stratified.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'time_to_merge_stratified.png'}")


def print_summary_stats(df: pl.DataFrame):
    """Print summary statistics for the stratified analysis."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Filter to merged PRs
    merged_df = df.filter(pl.col("is_merged"))
    
    print("\nOverall time-to-merge medians by agent (hours):")
    overall_stats = merged_df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("time_to_merge_hours").median().alias("median_hours"),
        pl.col("time_to_merge_hours").quantile(0.25).alias("p25_hours"),
        pl.col("time_to_merge_hours").quantile(0.75).alias("p75_hours"),
    ])
    overall_stats = sort_agents_human_first(overall_stats)
    print(overall_stats)
    
    print("\nTime-to-merge medians by agent and star bin (hours):")
    stratified_stats = merged_df.group_by(["agent", "star_bin"]).agg([
        pl.len().alias("total"),
        pl.col("time_to_merge_hours").median().alias("median_hours"),
    ]).sort(["agent", "star_bin"])
    print(stratified_stats)


def main(n_bins: int = 3, quantile_agent: str = "Human"):
    """Main entry point for stratified time-to-merge analysis.
    
    Args:
        n_bins: Number of star count bins to create (default: 3 for tertiles)
        quantile_agent: Agent to use for computing bin boundaries (default: "Human")
    """
    print("="*80)
    print("STRATIFIED TIME-TO-MERGE ANALYSIS")
    print("="*80)
    print(f"Configuration: {n_bins} bins based on {quantile_agent} quantiles")
    
    # Load data
    df = load_pr_repo_data()
    print(f"\nLoaded {len(df):,} PRs with repository information")
    
    # Compute star bins
    df, bin_edges, bin_labels = compute_star_bins(df, n_bins=n_bins, quantile_agent=quantile_agent)
    
    # Print summary stats
    print_summary_stats(df)
    
    # Generate plot
    plot_time_to_merge_stratified(df, bin_labels)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Plot saved to: {plots_path}")
    print("  - time_to_merge_stratified.png/pdf")


if __name__ == "__main__":
    # Default: 3 bins (tertiles) based on Human data
    main(n_bins=3, quantile_agent="Human")

