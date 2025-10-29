"""Generate merge rate analysis stratified by repository star counts.

NOTE: This script has been superseded by stratified_core_metrics.py, which uses
the new plotting_stratified framework and combines both merge rate and time-to-merge
analyses. This original script is kept for reference.
"""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


cur_path = Path(__file__).parent
root_path = cur_path.parent.parent
plots_path = root_path / "plots" / "main_metrics"

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


def wilson_score_interval(successes, trials, confidence=0.95):
    """Calculate Wilson score confidence interval for a proportion."""
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
    
    # Note: stargazer_count is automatically cleaned in load_hf_data_polars
    # (NULL values are imputed as 0 to fix data collection bug)
    
    # Add merge status
    pr_repo_data = pr_repo_data.with_columns([
        pl.col("merged_at").is_not_null().alias("is_merged"),
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
    # For 3 bins, we need 2 interior boundaries (33rd and 67th percentiles)
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
    
    # Create bin labels with interval notation and rounding
    # Add descriptive categories (Low/Medium/High)
    bin_labels = []
    all_edges = [min_stars] + list(bin_edges) + [max_stars]
    
    # Category names for tertiles
    categories = ["Low", "Medium", "High"]
    
    # Note: cut() with left_closed=False and breaks=[b1, b2, ...] creates bins:
    # Bin 0: values <= b1 (includes min and b1)
    # Bin 1: b1 < values <= b2 
    # Bin 2: values > b2 (includes everything above b2)
    # 
    # Example with breaks=[0, 18]:
    # Bin 0: values <= 0 -> [0, 0]
    # Bin 1: 0 < values <= 18 -> [1, 18]
    # Bin 2: values > 18 -> [19, max]
    
    for i in range(n_bins):
        if i == 0:
            # First bin: values <= first break
            lower_rounded = int(min_stars)
            upper_rounded = int(bin_edges[0]) if len(bin_edges) > 0 else int(min_stars)
        elif i < n_bins - 1:
            # Middle bins: break[i-1] < values <= break[i]
            lower_rounded = int(bin_edges[i-1]) + 1
            upper_rounded = int(bin_edges[i])
        else:
            # Last bin: values > last break
            lower_rounded = int(bin_edges[-1]) + 1
            upper_rounded = int(max_stars)
        
        # Use interval notation [a, b] with category prefix
        category = categories[i] if i < len(categories) else f"Bin{i+1}"
        bin_labels.append(f"{category} — [{lower_rounded:,}, {upper_rounded:,}] Stars")
    
    print(f"  Bin labels: {bin_labels}")
    
    # Assign bins using cut
    # cut() with n interior breaks creates n+1 bins
    df = df.with_columns([
        pl.col("stargazer_count").cut(
            breaks=list(bin_edges),
            labels=bin_labels,
            left_closed=False,
            include_breaks=False
        ).alias("star_bin")
    ])
    
    # Check for any PRs that didn't get assigned (shouldn't happen with cut)
    unassigned = df.filter(pl.col("star_bin").is_null())
    if len(unassigned) > 0:
        print(f"  Warning: {len(unassigned)} PRs were not assigned to bins")
        print(f"    Star range of unassigned: {unassigned['stargazer_count'].min()} - {unassigned['stargazer_count'].max()}")
    
    # Print bin distributions by agent
    print("\n  PRs per bin by agent:")
    bin_dist = df.group_by(["agent", "star_bin"]).agg([
        pl.len().alias("count")
    ]).sort(["agent", "star_bin"])
    print(bin_dist)
    
    return df, bin_edges, bin_labels


def plot_merge_rate_stratified(df: pl.DataFrame, bin_labels: list):
    """Plot merge rate stratified by repository star count bins.
    
    Creates a faceted plot with one subplot per star count bin, plus
    a smaller histogram below showing the data distribution.
    """
    print("\nCreating stratified merge rate plot...")
    
    # Calculate merge rate stats for each agent × bin combination
    stats_by_bin = df.group_by(["agent", "star_bin"]).agg([
        pl.len().alias("total"),
        pl.col("is_merged").sum().alias("merged"),
    ]).with_columns([
        (pl.col("merged") / pl.col("total") * 100).alias("merge_rate"),
    ])
    
    # Calculate total PRs per agent for fraction calculation
    total_per_agent = df.group_by("agent").agg([
        pl.len().alias("agent_total")
    ])
    
    # Join to get fractions
    stats_by_bin = stats_by_bin.join(
        total_per_agent,
        on="agent",
        how="left"
    ).with_columns([
        (pl.col("total") / pl.col("agent_total")).alias("fraction")
    ])
    
    # Sort with Human first within each bin
    stats_by_bin = stats_by_bin.sort(["star_bin", "agent"])
    
    # Get unique bins in the order they appear (should match bin_labels order)
    bins = bin_labels
    n_bins = len(bins)
    
    # Get agent order (Human first)
    agents_order = ['Human'] + sorted([a for a in df['agent'].unique().to_list() if a != 'Human'])
    
    # Create subplot layout with 2 rows (merge rate and histogram)
    # Use gridspec for custom height ratios (3:1)
    fig = plt.figure(figsize=(6 * n_bins, 4.5))
    gs = fig.add_gridspec(2, n_bins, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Create axes
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(n_bins)]
    # Create bottom axes with shared y-axis
    axes_bottom = []
    for i in range(n_bins):
        if i == 0:
            axes_bottom.append(fig.add_subplot(gs[1, i]))
        else:
            axes_bottom.append(fig.add_subplot(gs[1, i], sharey=axes_bottom[0]))
    
    # Colors: Human red, others teal
    agent_colors = {agent: '#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents_order}
    
    # Get Human baseline merge rate (overall)
    human_overall_stats = df.filter(pl.col("agent") == "Human").select([
        pl.len().alias("total"),
        pl.col("is_merged").sum().alias("merged"),
    ])
    human_overall_merge_rate = (human_overall_stats["merged"][0] / human_overall_stats["total"][0] * 100)
    
    # Plot each bin
    for bin_idx, (ax_top, ax_bottom, bin_label) in enumerate(zip(axes_top, axes_bottom, bins)):
        # Get data for this bin
        bin_data = stats_by_bin.filter(pl.col("star_bin") == bin_label)
        
        # Sort agents with Human first
        bin_data = sort_agents_human_first(bin_data)
        
        # Extract data for plotting
        agents = bin_data['agent'].to_list()
        merge_rates = bin_data['merge_rate'].to_list()
        merged_counts = bin_data['merged'].to_list()
        total_counts = bin_data['total'].to_list()
        fractions = bin_data['fraction'].to_list()
        
        # === TOP PLOT: Merge Rates ===
        # Calculate confidence intervals
        ci_lower = []
        ci_upper = []
        for merged, total in zip(merged_counts, total_counts):
            lower, upper = wilson_score_interval(merged, total)
            ci_lower.append(lower * 100)
            ci_upper.append(upper * 100)
        
        # Calculate error bar sizes
        yerr_lower = [rate - lower for rate, lower in zip(merge_rates, ci_lower)]
        yerr_upper = [upper - rate for rate, upper in zip(merge_rates, ci_upper)]
        
        # Plot bars
        colors = [agent_colors[agent] for agent in agents]
        bars = ax_top.bar(agents, merge_rates,
                          yerr=[yerr_lower, yerr_upper],
                          color=colors, alpha=0.8, capsize=5, error_kw={'linewidth': 2})
        
        # Add Human baseline (for this bin) as reference
        human_bin_merge_rate = merge_rates[agents.index("Human")] if "Human" in agents else None
        if human_bin_merge_rate is not None:
            ax_top.axhline(y=human_bin_merge_rate, color='#FF6B6B', linestyle='--', 
                          linewidth=2, alpha=0.5, zorder=0)
        
        # Formatting top plot
        if bin_idx == 0:
            ax_top.set_ylabel('Merge Rate (%)', fontsize=12)
        ax_top.set_title(bin_label, fontsize=13, fontweight='bold', pad=12)
        ax_top.set_xticks(range(len(agents)))
        ax_top.grid(True, alpha=0.3, axis='y')
        ax_top.set_ylim(0, 105)  # Give some headroom
        
        # Add value labels on bars
        for bar, rate, upper_err in zip(bars, merge_rates, yerr_upper):
            # Show merge rate percentage
            ax_top.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 1,
                       f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # === BOTTOM PLOT: Data Distribution (Histogram) ===
        # Use neutral grey color for all bars
        bars_hist = ax_bottom.bar(agents, fractions, color='#808080', alpha=0.6)
        
        # Add text labels with raw counts (always at bottom of bars)
        for bar, fraction, count in zip(bars_hist, fractions, total_counts):
            ax_bottom.text(bar.get_x() + bar.get_width()/2, 0.01,
                          f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Formatting bottom plot
        ax_bottom.set_xlabel('Agent', fontsize=10)
        if bin_idx == 0:
            ax_bottom.set_ylabel("Fraction of\nAgent's PRs", fontsize=10)
        ax_bottom.set_xticks(range(len(agents)))
        ax_bottom.set_xticklabels([])  # No tick labels on bottom plot
        ax_bottom.grid(True, alpha=0.3, axis='y')
    
    # Overall title with more vertical space below it
    fig.suptitle('Merge Rate by Repository Star Count', fontsize=16, fontweight='bold', y=1.0)
    fig.subplots_adjust(top=0.85)
 
    plt.savefig(plots_path / 'merge_rate_stratified.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'merge_rate_stratified.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'merge_rate_stratified.png'}")


def print_summary_stats(df: pl.DataFrame):
    """Print summary statistics for the stratified analysis."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nOverall merge rates by agent:")
    overall_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("is_merged").sum().alias("merged"),
    ]).with_columns([
        (pl.col("merged") / pl.col("total") * 100).alias("merge_rate"),
    ])
    overall_stats = sort_agents_human_first(overall_stats)
    print(overall_stats)
    
    print("\nMerge rates by agent and star bin:")
    stratified_stats = df.group_by(["agent", "star_bin"]).agg([
        pl.len().alias("total"),
        pl.col("is_merged").sum().alias("merged"),
    ]).with_columns([
        (pl.col("merged") / pl.col("total") * 100).alias("merge_rate"),
    ]).sort(["agent", "star_bin"])
    print(stratified_stats)


def main(n_bins: int = 3, quantile_agent: str = "Human"):
    """Main entry point for stratified merge rate analysis.
    
    Args:
        n_bins: Number of star count bins to create (default: 3 for tertiles)
        quantile_agent: Agent to use for computing bin boundaries (default: "Human")
    """
    print("="*80)
    print("STRATIFIED MERGE RATE ANALYSIS")
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
    plot_merge_rate_stratified(df, bin_labels)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Plot saved to: {plots_path}")
    print("  - merge_rate_stratified.png/pdf")


if __name__ == "__main__":
    # Default: 3 bins (tertiles) based on Human data
    # You can easily change this to 4 bins (quartiles) or 5 bins (quintiles) etc.
    main(n_bins=3, quantile_agent="Human")