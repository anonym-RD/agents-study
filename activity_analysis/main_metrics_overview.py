"""Generate key metrics plots as individual files for publication figures."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents, AgentNames


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


def prepare_pr_data(df: pl.LazyFrame, comments_df: pl.LazyFrame) -> pl.DataFrame:
    """Prepare PR data by adding computed columns and joining with user comments count."""
    # Filter comments to only include those from users (not bots)
    user_comments = comments_df.filter(
        pl.col("author").struct.field("typename") == "User"
    )
    
    # Count user comments per PR
    user_comments_count = user_comments.group_by(["pr_id", "agent"]).agg([
        pl.len().alias("user_comments_count")
    ])
    
    # Join with PR data and prepare columns
    return df.join(
        user_comments_count,
        left_on=["id", "agent"],
        right_on=["pr_id", "agent"],
        how="left"
    ).with_columns([
        # Impute null additions/deletions with 0 (missing data from GitHub API)
        pl.col("additions").fill_null(0),
        pl.col("deletions").fill_null(0),
        
        # Body length and presence
        pl.col("body").str.len_chars().fill_null(0).alias("body_length"),
        pl.col("body").is_not_null().alias("has_body"),
        
        # Parse datetime columns
        pl.col("created_at").str.to_datetime(),
        pl.col("merged_at").str.to_datetime(),
        pl.col("closed_at").str.to_datetime(),
        
        # Merge status
        pl.col("merged_at").is_not_null().alias("is_merged"),
        
        # Time to merge (in hours) - only for merged PRs
        ((pl.col("merged_at").str.to_datetime() - pl.col("created_at").str.to_datetime())
            .dt.total_seconds() / 3600.0)
            .alias("time_to_merge_hours"),
        
        # Total changes
        (pl.col("additions") + pl.col("deletions")).alias("total_changes"),
        
        # Fill null user_comments_count with 0 (PRs with no user comments)
        pl.col("user_comments_count").fill_null(0),
    ]).collect()


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


def plot_change_size(df: pl.DataFrame):
    """Plot 1: Total change size and composition (2 subplots).
    
    Top: Median total lines changed (additions + deletions)
    Bottom: Median addition ratio (additions / total changes)
    """
    print("Creating change size plot...")
    
    # Calculate addition ratio for each PR
    df_with_ratio = df.with_columns([
        (pl.col("additions") / (pl.col("additions") + pl.col("deletions"))).alias("add_ratio")
    ])
    
    # Check for PRs with both additions and deletions = 0 (before filtering)
    both_zero_prs = df_with_ratio.filter((pl.col("additions") == 0) & (pl.col("deletions") == 0))
    both_zero_count = len(both_zero_prs)
    both_zero_pct = 100 * both_zero_count / len(df_with_ratio) if len(df_with_ratio) > 0 else 0
    print(f"  PRs with both additions=0 and deletions=0: {both_zero_count}/{len(df_with_ratio)} ({both_zero_pct:.2f}%)")
    
    if both_zero_count > 0:
        print(f"\n  Sample PRs with both=0 (by agent):")
        print(both_zero_prs.group_by('agent').agg(pl.len().alias('count')).sort('count', descending=True))
        print(f"\n  Sample PRs with both=0 (by state):")
        print(both_zero_prs.group_by('state').agg(pl.len().alias('count')).sort('count', descending=True))
        print(f"\n  Sample URLs to spot check:")
        sample = both_zero_prs.select(['url', 'state', 'agent', 'additions', 'deletions', 'title']).head(15)
        for row in sample.iter_rows(named=True):
            print(f"    {row['state']:6} {row['agent']:8} {row['url']}")
            print(f"           {row['title'][:80] if row['title'] else '(no title)'}")
    
    # Sanity check: PRs with zero changes (0 additions AND 0 deletions) -> NaN add_ratio
    # Note: Polars returns NaN (not null) for 0/0 division, so we need to check for NaN
    nan_ratio_prs = df_with_ratio.filter(pl.col("add_ratio").is_nan())
    nan_count = len(nan_ratio_prs)
    nan_pct = 100 * nan_count / len(df_with_ratio) if len(df_with_ratio) > 0 else 0
    print(f"\n  PRs with NaN add_ratio (from 0/0): {nan_count}/{len(df_with_ratio)} ({nan_pct:.2f}%)")
    
    assert nan_pct <= 3.1, f"Too many PRs with zero changes: {nan_pct:.2f}% (expected <= 3.1%)"
    
    # Filter out PRs with zero changes (0/0 -> NaN)
    df_with_ratio = df_with_ratio.filter(pl.col("add_ratio").is_nan().not_())
    print(f"  After filtering: {len(df_with_ratio)} PRs remaining for analysis")
    
    # Calculate stats for total changes
    code_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("total_changes").median().alias("median_total_changes"),
    ])
    
    # Calculate stats for addition ratio
    ratio_stats = df_with_ratio.group_by("agent").agg([
        pl.col("add_ratio").median().alias("median_add_ratio"),
    ])
    
    # Sort both with Human first
    code_stats = sort_agents_human_first(code_stats)
    ratio_stats = sort_agents_human_first(ratio_stats)
    
    agents = code_stats['agent'].to_list()
    median_total = code_stats['median_total_changes'].to_list()
    median_ratios = ratio_stats['median_add_ratio'].to_list()
    
    # Human baselines
    human_total = median_total[0]
    human_ratio = median_ratios[0]
    
    # Create 2x1 subplot layout with 2:1 height ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 6), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    x = np.arange(len(agents))
    width = 0.6
    
    # Color schemes for each plot
    # Top plot: Human red, others teal/cyan
    colors_top = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    # Bottom plot: All green, Human gets dark border
    colors_bottom = ['#7FB069' for agent in agents]
    
    # ===== TOP PLOT: Total Lines Changed =====
    bars1 = ax1.bar(x, median_total, width, color=colors_top, alpha=0.8)
    
    # Add horizontal line for Human baseline
    ax1.axhline(y=human_total, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.5, label='Human baseline')
    
    ax1.set_ylabel('Median Total Lines Changed')
    ax1.set_xlabel('Agent')
    ax1.set_title('Total Change Size', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels inside bars
    for i, (bar, val) in enumerate(zip(bars1, median_total)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                 f'{val:.0f}', ha='center', va='center', fontweight='bold', 
                 fontsize=10, color='white')
    
    # ===== BOTTOM PLOT: Addition Ratio =====
    bars2 = ax2.bar(x, median_ratios, width, color=colors_bottom, alpha=0.8,
                    edgecolor=['#2F4F2F' if agent == 'Human' else colors_bottom[0] for agent in agents],
                    linewidth=[3 if agent == 'Human' else 0 for agent in agents])
    
    # Add horizontal line for Human baseline
    ax2.axhline(y=human_ratio, color='#2F4F2F', linestyle='--', linewidth=2, 
                alpha=0.5)
    
    ax2.set_ylabel('Median Addition Ratio')
    ax2.set_xlabel('Agent')
    ax2.set_title('Proportion of Changes that are Additions', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents, rotation=45, ha='right')
    ax2.set_ylim(0, 1)  # Ratio is between 0 and 1
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels inside bars
    for i, (bar, val) in enumerate(zip(bars2, median_ratios)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                 f'{val:.2f}', ha='center', va='center', fontweight='bold', 
                 fontsize=10, color='white')
    
    plt.tight_layout()
    plt.savefig(plots_path / 'change_size.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'change_size.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'change_size.png'}")


def plot_merge_rate(df: pl.DataFrame):
    """Plot 2: Merge rate (merged / total) with confidence intervals."""
    print("Creating merge rate plot...")
    
    state_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("is_merged").sum().alias("merged"),
    ])
    
    state_stats = state_stats.with_columns([
        (pl.col("merged") / pl.col("total") * 100).alias("merge_rate"),
    ])
    
    # Sort with Human first
    state_stats = sort_agents_human_first(state_stats)
    
    agents = state_stats['agent'].to_list()
    merge_rate = state_stats['merge_rate'].to_list()
    merged_counts = state_stats['merged'].to_list()
    total_counts = state_stats['total'].to_list()
    
    # Calculate Wilson score confidence intervals
    ci_lower = []
    ci_upper = []
    for merged, total in zip(merged_counts, total_counts):
        lower, upper = wilson_score_interval(merged, total)
        ci_lower.append(lower * 100)
        ci_upper.append(upper * 100)
    
    # Calculate error bar sizes
    yerr_lower = [rate - lower for rate, lower in zip(merge_rate, ci_lower)]
    yerr_upper = [upper - rate for rate, upper in zip(merge_rate, ci_upper)]
    
    # Human baseline
    human_merge_rate = merge_rate[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    bars = ax.bar(agents, merge_rate, 
                  yerr=[yerr_lower, yerr_upper],
                  color=colors, alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    
    ax.axhline(y=human_merge_rate, color='#FF6B6B', linestyle='--', linewidth=2, 
               alpha=0.5, label='Human baseline')
    
    ax.set_ylabel('Merge Rate (%)')
    ax.set_xlabel('Agent')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, rate, upper_err) in enumerate(zip(bars, merge_rate, yerr_upper)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_path / 'merge_rate.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'merge_rate.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'merge_rate.png'}")


def plot_time_to_merge_distribution(df: pl.DataFrame):
    """Plot 3: Distribution of time to merge (custom box plot with p10-p90)."""
    print("Creating time to merge distribution plot...")
    
    # Filter to merged PRs only
    merged_df = df.filter(pl.col("is_merged"))
    
    # Get data sorted with Human first
    agents_order = ['Human'] + sorted([a for a in merged_df['agent'].unique().to_list() if a != 'Human'])
    
    # Convert to hours for better readability in this case
    merged_df = merged_df.with_columns([
        pl.col("time_to_merge_hours").alias("time_to_merge_hours")
    ])
    
    # Calculate percentiles for each agent
    percentile_stats = merged_df.group_by("agent").agg([
        pl.col("time_to_merge_hours").quantile(0.10).alias("p10"),
        pl.col("time_to_merge_hours").quantile(0.25).alias("p25"),
        pl.col("time_to_merge_hours").quantile(0.50).alias("p50"),
        pl.col("time_to_merge_hours").quantile(0.75).alias("p75"),
        pl.col("time_to_merge_hours").quantile(0.90).alias("p90"),
    ])
    
    # Sort with Human first
    percentile_stats = sort_agents_human_first(percentile_stats)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    positions = np.arange(len(agents_order))
    
    # Draw custom box plots
    for i, agent in enumerate(agents_order):
        agent_stats = percentile_stats.filter(pl.col("agent") == agent).to_dicts()[0]
        
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
        ax.add_patch(box)
        
        # Draw median line
        ax.plot([i - box_width/2, i + box_width/2], [p50, p50], 
               color='black', linewidth=2.5, zorder=3)
        
        # Draw whisker caps only (no connecting lines) - this makes it visually distinct
        cap_width = 0.3
        # P10 cap
        ax.plot([i - cap_width/2, i + cap_width/2], [p10, p10],
               color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
        # P90 cap
        ax.plot([i - cap_width/2, i + cap_width/2], [p90, p90],
               color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
        
        # Optional: add very thin dashed lines to connect caps to box (for clarity)
        ax.plot([i, i], [p10, p25], color=whisker_color, linewidth=0.8, 
               linestyle=':', alpha=0.5, zorder=1)
        ax.plot([i, i], [p75, p90], color=whisker_color, linewidth=0.8, 
               linestyle=':', alpha=0.5, zorder=1)
    
    # Human baseline median
    human_median = percentile_stats.filter(pl.col("agent") == "Human")["p50"].to_list()[0]
    ax.axhline(y=human_median, color='#FF6B6B', linestyle='--', linewidth=2, 
               alpha=0.4, label='Human median', zorder=0)
    
    ax.set_ylabel('Time to Merge (hours, log scale)')
    ax.set_xlabel('Agent')
    ax.set_xticks(positions)
    ax.set_xticklabels(agents_order, rotation=45, ha='right')
    ax.set_yscale('log')  # Use log scale for better visualization
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.legend()
    
    # Add readable time conversions as text annotations on select tick marks
    def hours_to_readable(hours):
        if hours < 1/60:  # Less than 1 minute
            secs = hours * 3600
            return f'≈{secs:.0f}s' if secs >= 1 else f'≈{secs:.1f}s'
        elif hours < 1:  # Less than 1 hour
            mins = hours * 60
            return f'≈{mins:.0f}min' if mins >= 1 else f'≈{mins:.1f}min'
        elif hours < 24:  # Less than 1 day
            return f'≈{hours:.1f}h'
        else:  # Days
            return f'≈{hours/24:.1f}d'
    
    # Get current y-ticks and add readable annotations for select power-of-10 values
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    
    # Only annotate every other power of 10 to avoid overlap
    annotated_powers = []
    for tick in yticks:
        if ylim[0] <= tick <= ylim[1] and tick > 0:
            log_val = np.log10(tick)
            # Only annotate if it's a power of 10
            if abs(log_val - round(log_val)) < 0.01:
                power = round(log_val)
                # Skip if adjacent power was already annotated
                if not any(abs(power - p) <= 1 for p in annotated_powers[-1:]):
                    readable = hours_to_readable(tick)
                    # Use transform to place in axis coordinates for better control
                    ax.text(0.02, tick, readable, transform=ax.get_yaxis_transform(),
                           ha='left', va='center', fontsize=8, color='#666666', 
                           style='italic', alpha=0.7)
                    annotated_powers.append(power)
    
    plt.tight_layout()
    plt.savefig(plots_path / 'time_to_merge_dist.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'time_to_merge_dist.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'time_to_merge_dist.png'}")


def plot_engagement(df: pl.DataFrame):
    """Plot 4: Engagement metrics (comments and reviews) - bar + line combo in 1x2 layout.
    
    Note: Uses user_comments_count which only includes comments from users (not bots).
    """
    print("Creating engagement plot...")
    
    engagement_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("user_comments_count").mean().alias("mean_comments"),
        (pl.col("user_comments_count") > 0).sum().alias("has_comments_count"),
        pl.col("reviews_count").fill_null(0).mean().alias("mean_reviews"),
        (pl.col("reviews_count").fill_null(0) > 0).sum().alias("has_reviews_count"),
    ])
    
    engagement_stats = engagement_stats.with_columns([
        (pl.col("has_comments_count") / pl.col("total") * 100).alias("has_comments_pct"),
        (pl.col("has_reviews_count") / pl.col("total") * 100).alias("has_reviews_pct"),
    ])
    
    # Sort with Human first
    engagement_stats = sort_agents_human_first(engagement_stats)
    
    # Create 1x2 plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    agents = engagement_stats['agent'].to_list()
    x = np.arange(len(agents))
    width = 0.6
    
    # Human baseline values
    human_mean_comments = engagement_stats['mean_comments'].to_list()[0]
    human_has_comments_pct = engagement_stats['has_comments_pct'].to_list()[0]
    human_mean_reviews = engagement_stats['mean_reviews'].to_list()[0]
    human_has_reviews_pct = engagement_stats['has_reviews_pct'].to_list()[0]
    
    # Plot 1: Comments
    mean_comments = engagement_stats['mean_comments'].to_list()
    has_comments_pct = engagement_stats['has_comments_pct'].to_list()
    
    # Bar chart for mean comments
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    bars = ax1.bar(x, mean_comments, width, color=colors, alpha=0.7, label='Mean comments')
    
    # Scatter plot for % with comments on secondary axis (no lines)
    ax1_twin = ax1.twinx()
    ax1_twin.scatter(x, has_comments_pct, color='#2E86AB', s=100, 
                     label='% with comments', zorder=3, marker='o', edgecolors='white', linewidths=1.5)
    
    ax1.set_ylabel('Mean User Comments per PR', color='#333333')
    ax1_twin.set_ylabel('% of PRs with User Comments', color='#2E86AB')
    ax1.set_xlabel('Agent')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=45, ha='right')
    ax1.set_title('UserComments per PR', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1_twin.set_ylim(0, 110)  # Give some headroom
    ax1_twin.spines['right'].set_color('#2E86AB')
    ax1_twin.tick_params(axis='y', labelcolor='#2E86AB')
    
    # Add value labels
    for i, (bar, val, pct) in enumerate(zip(bars, mean_comments, has_comments_pct)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_comments)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax1_twin.text(i, pct + 3, f'{pct:.0f}%', ha='center', va='bottom', 
                     fontweight='bold', fontsize=9, color='#2E86AB')
    
    # Plot 2: Reviews
    mean_reviews = engagement_stats['mean_reviews'].to_list()
    has_reviews_pct = engagement_stats['has_reviews_pct'].to_list()
    
    # Bar chart for mean reviews
    bars = ax2.bar(x, mean_reviews, width, color=colors, alpha=0.7, label='Mean reviews')
    
    # Scatter plot for % with reviews on secondary axis (no lines)
    ax2_twin = ax2.twinx()
    ax2_twin.scatter(x, has_reviews_pct, color='#A23B72', s=100, 
                     label='% with reviews', zorder=3, marker='o', edgecolors='white', linewidths=1.5)
    
    ax2.set_ylabel('Mean Reviews per PR', color='#333333')
    ax2_twin.set_ylabel('% of PRs with Reviews', color='#A23B72')
    ax2.set_xlabel('Agent')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents, rotation=45, ha='right')
    ax2.set_title('Reviews per PR', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2_twin.set_ylim(0, 110)  # Give some headroom
    ax2_twin.spines['right'].set_color('#A23B72')
    ax2_twin.tick_params(axis='y', labelcolor='#A23B72')
    
    # Add value labels
    for i, (bar, val, pct) in enumerate(zip(bars, mean_reviews, has_reviews_pct)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_reviews)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax2_twin.text(i, pct + 3, f'{pct:.0f}%', ha='center', va='bottom', 
                     fontweight='bold', fontsize=9, color='#A23B72')
    
    plt.tight_layout()
    plt.savefig(plots_path / 'engagement.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'engagement.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'engagement.png'}")


def main():
    """Main entry point for the metrics overview script."""
    print("="*80)
    print("GENERATING MAIN METRICS OVERVIEW PLOTS")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Loading Comments data from HuggingFace dataset...")
    comments_lazy = load_lazy_table_for_all_agents(TableNames.COMMENTS)
    
    print("Preparing data (filtering user comments and joining)...")
    df = prepare_pr_data(df_lazy, comments_lazy)
    
    print(f"Loaded {len(df):,} PRs\n")
    
    # Generate individual plots
    plot_change_size(df)
    plot_merge_rate(df)
    plot_time_to_merge_distribution(df)
    plot_engagement(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"All plots saved to: {plots_path}")
    print("  - change_size.png/pdf")
    print("  - merge_rate.png/pdf")
    print("  - time_to_merge_dist.png/pdf")
    print("  - engagement.png/pdf")


if __name__ == "__main__":
    main()