"""Analysis of PRs from HuggingFace dataset, comparing agents to Human baseline."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents, AgentNames


cur_path = Path(__file__).parent
root_path = cur_path.parent.parent
plots_path = root_path / "plots" / "agent_comparison"

# Create plots directory if it doesn't exist
plots_path.mkdir(parents=True, exist_ok=True)


def prepare_pr_data(df: pl.LazyFrame) -> pl.DataFrame:
    """Prepare PR data by adding computed columns."""
    return df.with_columns([
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
        # Use total_seconds() / 3600 for better precision (dt.total_hours() returns int)
        ((pl.col("merged_at").str.to_datetime() - pl.col("created_at").str.to_datetime())
            .dt.total_seconds() / 3600.0)
            .alias("time_to_merge_hours"),
        
        # Total changes
        (pl.col("additions") + pl.col("deletions")).alias("total_changes"),
    ]).collect()


def print_overall_stats(df: pl.DataFrame):
    """Print overall statistics across all agents."""
    print("=" * 80)
    print("OVERALL PR STATISTICS ACROSS ALL AGENTS")
    print("=" * 80)
    
    total_prs = len(df)
    print(f"\nTotal PRs: {total_prs:,}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    # Agent distribution
    print("\n" + "=" * 60)
    print("PR Count by Agent")
    print("=" * 60)
    agent_counts = df.group_by("agent").agg(
        pl.len().alias("count")
    ).sort("agent")
    
    for row in agent_counts.iter_rows(named=True):
        pct = (row['count'] / total_prs) * 100
        print(f"  {row['agent']:10s}: {row['count']:6,} ({pct:5.1f}%)")


def sort_agents_human_first(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a dataframe so Human comes first, then others alphabetically."""
    human_row = df.filter(pl.col("agent") == "Human")
    other_rows = df.filter(pl.col("agent") != "Human").sort("agent")
    return pl.concat([human_row, other_rows])


def wilson_score_interval(successes, trials, confidence=0.95):
    """Calculate Wilson score confidence interval for a proportion.
    
    This is more accurate than normal approximation, especially for proportions
    near 0 or 1, or for small sample sizes.
    
    Args:
        successes: Number of successes (e.g., merged PRs)
        trials: Total number of trials (e.g., total PRs)
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (lower_bound, upper_bound): Confidence interval as proportions (0-1)
    """
    if trials == 0:
        return 0.0, 0.0
    
    p = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    
    lower = max(0.0, centre - margin)  # Ensure bounds are in [0, 1]
    upper = min(1.0, centre + margin)
    
    return lower, upper


def analyze_pr_states_by_agent(df: pl.DataFrame):
    """Analyze PR states (merged, closed, open) by agent."""
    print("\n" + "=" * 80)
    print("1. PR STATE ANALYSIS BY AGENT")
    print("=" * 80)
    
    # Calculate state statistics by agent
    state_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("is_merged").sum().alias("merged"),
        ((pl.col("state") == "CLOSED") & pl.col("merged_at").is_null()).sum().alias("closed_not_merged"),
        (pl.col("state") == "OPEN").sum().alias("open"),
        # Also count total closed (merged + closed_not_merged)
        ((pl.col("state") == "CLOSED") | pl.col("is_merged")).sum().alias("total_closed"),
    ])
    
    # Add percentages
    state_stats = state_stats.with_columns([
        (pl.col("merged") / pl.col("total") * 100).alias("merge_rate_of_total"),
        (pl.col("merged") / pl.col("total_closed") * 100).alias("merge_rate_of_closed"),
        (pl.col("closed_not_merged") / pl.col("total") * 100).alias("closed_pct"),
        (pl.col("open") / pl.col("total") * 100).alias("open_pct"),
    ])
    
    # Sort with Human first
    state_stats = sort_agents_human_first(state_stats)
    
    print("\nDetailed State Analysis:")
    print(f"{'Agent':<10} {'Total':>7} {'Merged':>7} {'M/Total %':>10} {'M/Closed %':>11} "
          f"{'Closed':>7} {'Closed %':>9} {'Open':>7} {'Open %':>9}")
    print("-" * 100)
    
    for row in state_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total']:>7,} {row['merged']:>7,} "
              f"{row['merge_rate_of_total']:>9.1f}% {row['merge_rate_of_closed']:>10.1f}% "
              f"{row['closed_not_merged']:>7,} {row['closed_pct']:>8.1f}% "
              f"{row['open']:>7,} {row['open_pct']:>8.1f}%")
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get data
    agents = state_stats['agent'].to_list()
    merge_rate_total = state_stats['merge_rate_of_total'].to_list()
    merge_rate_closed = state_stats['merge_rate_of_closed'].to_list()
    
    # Get counts for calculating confidence intervals
    merged_counts = state_stats['merged'].to_list()
    total_counts = state_stats['total'].to_list()
    total_closed_counts = state_stats['total_closed'].to_list()
    
    # Calculate Wilson score confidence intervals
    # For merge rate of total PRs
    ci_total_lower = []
    ci_total_upper = []
    for merged, total in zip(merged_counts, total_counts):
        lower, upper = wilson_score_interval(merged, total)
        ci_total_lower.append(lower * 100)  # Convert to percentage
        ci_total_upper.append(upper * 100)
    
    # For merge rate of closed PRs
    ci_closed_lower = []
    ci_closed_upper = []
    for merged, total_closed in zip(merged_counts, total_closed_counts):
        lower, upper = wilson_score_interval(merged, total_closed)
        ci_closed_lower.append(lower * 100)
        ci_closed_upper.append(upper * 100)
    
    # Calculate error bar sizes (distance from point estimate to CI bounds)
    yerr_total_lower = [rate - lower for rate, lower in zip(merge_rate_total, ci_total_lower)]
    yerr_total_upper = [upper - rate for rate, upper in zip(merge_rate_total, ci_total_upper)]
    
    yerr_closed_lower = [rate - lower for rate, lower in zip(merge_rate_closed, ci_closed_lower)]
    yerr_closed_upper = [upper - rate for rate, upper in zip(merge_rate_closed, ci_closed_upper)]
    
    # Human baseline values (first in the list)
    human_merge_rate_total = merge_rate_total[0]
    human_merge_rate_closed = merge_rate_closed[0]
    
    # Color Human differently
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Plot 1: Merge Rate (Merged / Total PRs)
    bars = ax1.bar(agents, merge_rate_total, 
                   yerr=[yerr_total_lower, yerr_total_upper],
                   color=colors, alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    ax1.axhline(y=human_merge_rate_total, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax1.set_title('Merge Rate: Merged / Total PRs (95% CI)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Merge Rate (%)')
    ax1.set_xlabel('Agent')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # Add value labels (adjust position to account for error bars)
    for i, (bar, rate, upper_err) in enumerate(zip(bars, merge_rate_total, yerr_total_upper)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Merge Rate (Merged / Closed PRs)
    bars = ax2.bar(agents, merge_rate_closed,
                   yerr=[yerr_closed_lower, yerr_closed_upper],
                   color=colors, alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    ax2.axhline(y=human_merge_rate_closed, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax2.set_title('Merge Rate: Merged / Closed PRs (95% CI)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Merge Rate (%)')
    ax2.set_xlabel('Agent')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    
    # Add value labels (adjust position to account for error bars)
    for i, (bar, rate, upper_err) in enumerate(zip(bars, merge_rate_closed, yerr_closed_upper)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + upper_err + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Stacked bar chart showing all states
    merged = state_stats['merge_rate_of_total'].to_list()
    closed = state_stats['closed_pct'].to_list()
    open_prs = state_stats['open_pct'].to_list()
    
    x = np.arange(len(agents))
    width = 0.6
    
    ax3.bar(x, merged, width, label='Merged', color='#2E8B57', alpha=0.8)
    ax3.bar(x, closed, width, bottom=merged, label='Closed (not merged)', color='#DC143C', alpha=0.8)
    ax3.bar(x, open_prs, width, bottom=np.array(merged) + np.array(closed), 
            label='Open', color='#4169E1', alpha=0.8)
    
    ax3.set_title('PR State Distribution by Agent', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xlabel('Agent')
    ax3.set_xticks(x)
    ax3.set_xticklabels(agents, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Comparison of merge rates (both definitions)
    x = np.arange(len(agents))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, merge_rate_total, width, label='Merged/Total', 
                    color='#4ECDC4', alpha=0.8)
    bars2 = ax4.bar(x + width/2, merge_rate_closed, width, label='Merged/Closed', 
                    color='#95E1D3', alpha=0.8)
    
    ax4.axhline(y=human_merge_rate_total, color='#FF6B6B', linestyle='--', linewidth=1.5, 
                alpha=0.5, label=f'Human M/Total ({human_merge_rate_total:.1f}%)')
    ax4.axhline(y=human_merge_rate_closed, color='#FF6B6B', linestyle=':', linewidth=1.5, 
                alpha=0.5, label=f'Human M/Closed ({human_merge_rate_closed:.1f}%)')
    
    ax4.set_title('Merge Rate Comparison: Two Definitions', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Merge Rate (%)')
    ax4.set_xlabel('Agent')
    ax4.set_xticks(x)
    ax4.set_xticklabels(agents, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_path / 'pr_states_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Saved PR state comparison to {plots_path / 'pr_states_by_agent.png'}")


def analyze_body_length_by_agent(df: pl.DataFrame):
    """Analyze PR body length by agent."""
    print("\n" + "=" * 80)
    print("2. PR BODY LENGTH ANALYSIS BY AGENT")
    print("=" * 80)
    
    body_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("body_length").mean().alias("mean_length"),
        pl.col("body_length").median().alias("median_length"),
        pl.col("body_length").min().alias("min_length"),
        pl.col("body_length").max().alias("max_length"),
        pl.col("body_length").std().alias("std_length"),
        pl.col("has_body").sum().alias("has_body_count"),
    ])
    
    body_stats = body_stats.with_columns([
        (pl.col("has_body_count") / pl.col("total") * 100).alias("has_body_pct")
    ])
    
    # Sort with Human first
    body_stats = sort_agents_human_first(body_stats)
    
    print("\nBody Length Statistics:")
    print(f"{'Agent':<10} {'Mean':>8} {'Median':>8} {'Std':>8} {'Has Body %':>11}")
    print("-" * 60)
    
    for row in body_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['mean_length']:>8.0f} {row['median_length']:>8.0f} "
              f"{row['std_length']:>8.0f} {row['has_body_pct']:>10.1f}%")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    agents = body_stats['agent'].to_list()
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Human baseline values (first in the list)
    human_mean = body_stats['mean_length'].to_list()[0]
    human_median = body_stats['median_length'].to_list()[0]
    human_has_body_pct = body_stats['has_body_pct'].to_list()[0]
    
    # Mean body length
    means = body_stats['mean_length'].to_list()
    bars = ax1.bar(agents, means, color=colors, alpha=0.8)
    ax1.axhline(y=human_mean, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax1.set_title('Average PR Body Length by Agent', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Body Length (characters)')
    ax1.set_xlabel('Agent')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                f'{mean:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Median body length
    medians = body_stats['median_length'].to_list()
    bars = ax2.bar(agents, medians, color=colors, alpha=0.8)
    ax2.axhline(y=human_median, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax2.set_title('Median PR Body Length by Agent', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Median Body Length (characters)')
    ax2.set_xlabel('Agent')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, median in zip(bars, medians):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(medians)*0.02,
                f'{median:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Has body percentage
    has_body_pcts = body_stats['has_body_pct'].to_list()
    bars = ax3.bar(agents, has_body_pcts, color=colors, alpha=0.8)
    ax3.axhline(y=human_has_body_pct, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax3.set_title('% of PRs with Body Text by Agent', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xlabel('Agent')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    for bar, pct in zip(bars, has_body_pcts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Box plot comparison - ensure Human is first
    agent_list = agents  # Use the same sorted order
    body_data = [df.filter(pl.col("agent") == agent)["body_length"].to_list() 
                 for agent in agent_list]
    
    bp = ax4.boxplot(body_data, tick_labels=agent_list, patch_artist=True)
    for patch, agent in zip(bp['boxes'], agent_list):
        color = '#FF6B6B' if agent == 'Human' else '#4ECDC4'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_title('Body Length Distribution by Agent', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Body Length (characters)')
    ax4.set_xlabel('Agent')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_path / 'pr_body_length_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved body length comparison to {plots_path / 'pr_body_length_by_agent.png'}")


def analyze_code_changes_by_agent(df: pl.DataFrame):
    """Analyze code changes (additions, deletions, files) by agent."""
    print("\n" + "=" * 80)
    print("3. CODE CHANGES ANALYSIS BY AGENT")
    print("=" * 80)
    
    code_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("additions").mean().alias("mean_additions"),
        pl.col("additions").median().alias("median_additions"),
        pl.col("deletions").mean().alias("mean_deletions"),
        pl.col("deletions").median().alias("median_deletions"),
        pl.col("total_changes").mean().alias("mean_total_changes"),
        pl.col("total_changes").median().alias("median_total_changes"),
        pl.col("changed_files").mean().alias("mean_files"),
        pl.col("changed_files").median().alias("median_files"),
    ])
    
    # Sort with Human first
    code_stats = sort_agents_human_first(code_stats)
    
    print("\nCode Changes Statistics:")
    print(f"{'Agent':<10} {'Mean Add':>10} {'Mean Del':>10} {'Mean Total':>12} {'Mean Files':>11}")
    print("-" * 70)
    
    for row in code_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['mean_additions']:>10.1f} {row['mean_deletions']:>10.1f} "
              f"{row['mean_total_changes']:>12.1f} {row['mean_files']:>11.1f}")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    agents = code_stats['agent'].to_list()
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Human baseline values
    human_adds = code_stats['mean_additions'].to_list()[0]
    human_dels = code_stats['mean_deletions'].to_list()[0]
    human_total = code_stats['mean_total_changes'].to_list()[0]
    human_files = code_stats['mean_files'].to_list()[0]
    
    # Mean additions
    mean_adds = code_stats['mean_additions'].to_list()
    bars = ax1.bar(agents, mean_adds, color=colors, alpha=0.8)
    ax1.axhline(y=human_adds, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax1.set_title('Average Lines Added by Agent', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Additions')
    ax1.set_xlabel('Agent')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    for bar, val in zip(bars, mean_adds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_adds)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Mean deletions
    mean_dels = code_stats['mean_deletions'].to_list()
    bars = ax2.bar(agents, mean_dels, color=colors, alpha=0.8)
    ax2.axhline(y=human_dels, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax2.set_title('Average Lines Deleted by Agent', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Deletions')
    ax2.set_xlabel('Agent')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, val in zip(bars, mean_dels):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_dels)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Mean total changes
    mean_total = code_stats['mean_total_changes'].to_list()
    bars = ax3.bar(agents, mean_total, color=colors, alpha=0.8)
    ax3.axhline(y=human_total, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax3.set_title('Average Total Changes by Agent', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Total Changes (Add + Del)')
    ax3.set_xlabel('Agent')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    for bar, val in zip(bars, mean_total):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_total)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Mean changed files
    mean_files = code_stats['mean_files'].to_list()
    bars = ax4.bar(agents, mean_files, color=colors, alpha=0.8)
    ax4.axhline(y=human_files, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax4.set_title('Average Files Changed by Agent', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean Files Changed')
    ax4.set_xlabel('Agent')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    for bar, val in zip(bars, mean_files):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_files)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_path / 'pr_code_changes_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved code changes comparison to {plots_path / 'pr_code_changes_by_agent.png'}")


def analyze_engagement_by_agent(df: pl.DataFrame):
    """Analyze engagement metrics (comments, reviews) by agent."""
    print("\n" + "=" * 80)
    print("4. ENGAGEMENT ANALYSIS BY AGENT")
    print("=" * 80)
    
    engagement_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("comments_count").mean().alias("mean_comments"),
        pl.col("comments_count").median().alias("median_comments"),
        (pl.col("comments_count") > 0).sum().alias("has_comments_count"),
        pl.col("reviews_count").mean().alias("mean_reviews"),
        pl.col("reviews_count").median().alias("median_reviews"),
        (pl.col("reviews_count") > 0).sum().alias("has_reviews_count"),
    ])
    
    engagement_stats = engagement_stats.with_columns([
        (pl.col("has_comments_count") / pl.col("total") * 100).alias("has_comments_pct"),
        (pl.col("has_reviews_count") / pl.col("total") * 100).alias("has_reviews_pct"),
    ])
    
    # Sort with Human first
    engagement_stats = sort_agents_human_first(engagement_stats)
    
    print("\nEngagement Statistics:")
    print(f"{'Agent':<10} {'Mean Cmt':>10} {'% w/ Cmt':>10} {'Mean Rev':>10} {'% w/ Rev':>10}")
    print("-" * 70)
    
    for row in engagement_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['mean_comments']:>10.1f} {row['has_comments_pct']:>9.1f}% "
              f"{row['mean_reviews']:>10.1f} {row['has_reviews_pct']:>9.1f}%")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    agents = engagement_stats['agent'].to_list()
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Human baseline values
    human_mean_comments = engagement_stats['mean_comments'].to_list()[0]
    human_has_comments_pct = engagement_stats['has_comments_pct'].to_list()[0]
    human_mean_reviews = engagement_stats['mean_reviews'].to_list()[0]
    human_has_reviews_pct = engagement_stats['has_reviews_pct'].to_list()[0]
    
    # Mean comments
    mean_comments = engagement_stats['mean_comments'].to_list()
    bars = ax1.bar(agents, mean_comments, color=colors, alpha=0.8)
    ax1.axhline(y=human_mean_comments, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax1.set_title('Average Comments per PR by Agent', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Comments')
    ax1.set_xlabel('Agent')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    for bar, val in zip(bars, mean_comments):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_comments)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # % with comments
    has_comments_pcts = engagement_stats['has_comments_pct'].to_list()
    bars = ax2.bar(agents, has_comments_pcts, color=colors, alpha=0.8)
    ax2.axhline(y=human_has_comments_pct, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax2.set_title('% of PRs with Comments by Agent', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel('Agent')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, val in zip(bars, has_comments_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Mean reviews
    mean_reviews = engagement_stats['mean_reviews'].to_list()
    bars = ax3.bar(agents, mean_reviews, color=colors, alpha=0.8)
    ax3.axhline(y=human_mean_reviews, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax3.set_title('Average Reviews per PR by Agent', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Reviews')
    ax3.set_xlabel('Agent')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    for bar, val in zip(bars, mean_reviews):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_reviews)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # % with reviews
    has_reviews_pcts = engagement_stats['has_reviews_pct'].to_list()
    bars = ax4.bar(agents, has_reviews_pcts, color=colors, alpha=0.8)
    ax4.axhline(y=human_has_reviews_pct, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax4.set_title('% of PRs with Reviews by Agent', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_xlabel('Agent')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    for bar, val in zip(bars, has_reviews_pcts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_path / 'pr_engagement_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved engagement comparison to {plots_path / 'pr_engagement_by_agent.png'}")


def analyze_commits_by_agent(df: pl.DataFrame):
    """Analyze commits count by agent."""
    print("\n" + "=" * 80)
    print("5. COMMITS ANALYSIS BY AGENT")
    print("=" * 80)
    
    commits_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("commits_count").fill_null(0).mean().alias("mean_commits"),
        pl.col("commits_count").fill_null(0).median().alias("median_commits"),
        (pl.col("commits_count").fill_null(0) > 0).sum().alias("has_commits_count"),
        pl.col("commits_count").fill_null(0).min().alias("min_commits"),
        pl.col("commits_count").fill_null(0).max().alias("max_commits"),
        pl.col("commits_count").fill_null(0).std().alias("std_commits"),
    ])
    
    commits_stats = commits_stats.with_columns([
        (pl.col("has_commits_count") / pl.col("total") * 100).alias("has_commits_pct")
    ])
    
    # Sort with Human first
    commits_stats = sort_agents_human_first(commits_stats)
    
    print("\nCommits Statistics:")
    print(f"{'Agent':<10} {'Mean Commits':>13} {'% w/ Commits':>13} {'Median':>8} {'Max':>8}")
    print("-" * 70)
    
    for row in commits_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['mean_commits']:>13.1f} {row['has_commits_pct']:>12.1f}% "
              f"{row['median_commits']:>8.0f} {row['max_commits']:>8,}")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    agents = commits_stats['agent'].to_list()
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Human baseline values
    human_mean_commits = commits_stats['mean_commits'].to_list()[0]
    human_has_commits_pct = commits_stats['has_commits_pct'].to_list()[0]
    human_median_commits = commits_stats['median_commits'].to_list()[0]
    
    # Mean commits
    mean_commits = commits_stats['mean_commits'].to_list()
    bars = ax1.bar(agents, mean_commits, color=colors, alpha=0.8)
    ax1.axhline(y=human_mean_commits, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax1.set_title('Average Commits per PR by Agent', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Commits')
    ax1.set_xlabel('Agent')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    for bar, val in zip(bars, mean_commits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_commits)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Median commits
    median_commits = commits_stats['median_commits'].to_list()
    bars = ax2.bar(agents, median_commits, color=colors, alpha=0.8)
    ax2.axhline(y=human_median_commits, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax2.set_title('Median Commits per PR by Agent', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Median Commits')
    ax2.set_xlabel('Agent')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, val in zip(bars, median_commits):
        max_val = max([v for v in median_commits if v is not None])
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # % with commits
    has_commits_pcts = commits_stats['has_commits_pct'].to_list()
    bars = ax3.bar(agents, has_commits_pcts, color=colors, alpha=0.8)
    ax3.axhline(y=human_has_commits_pct, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax3.set_title('% of PRs with Commits by Agent', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xlabel('Agent')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    for bar, val in zip(bars, has_commits_pcts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Box plot comparison
    agent_list = agents  # Use the same sorted order
    commits_data = [df.filter(pl.col("agent") == agent)["commits_count"].fill_null(0).to_list() 
                    for agent in agent_list]
    
    bp = ax4.boxplot(commits_data, tick_labels=agent_list, patch_artist=True)
    for patch, agent in zip(bp['boxes'], agent_list):
        color = '#FF6B6B' if agent == 'Human' else '#4ECDC4'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_title('Commits Distribution by Agent', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Commits')
    ax4.set_xlabel('Agent')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_path / 'pr_commits_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved commits comparison to {plots_path / 'pr_commits_by_agent.png'}")


def analyze_time_to_merge_by_agent(df: pl.DataFrame):
    """Analyze time to merge for merged PRs by agent."""
    print("\n" + "=" * 80)
    print("6. TIME TO MERGE ANALYSIS BY AGENT (Merged PRs Only)")
    print("=" * 80)
    
    # Filter to merged PRs only
    merged_df = df.filter(pl.col("is_merged"))
    
    time_stats = merged_df.group_by("agent").agg([
        pl.len().alias("merged_count"),
        pl.col("time_to_merge_hours").mean().alias("mean_hours"),
        pl.col("time_to_merge_hours").median().alias("median_hours"),
        pl.col("time_to_merge_hours").min().alias("min_hours"),
        pl.col("time_to_merge_hours").max().alias("max_hours"),
        pl.col("time_to_merge_hours").std().alias("std_hours"),
    ])
    
    time_stats = time_stats.with_columns([
        (pl.col("mean_hours") / 24).alias("mean_days"),
        (pl.col("median_hours") / 24).alias("median_days"),
    ])
    
    # Sort with Human first
    time_stats = sort_agents_human_first(time_stats)
    
    print("\nTime to Merge Statistics:")
    print(f"{'Agent':<10} {'Merged':>8} {'Mean (h)':>10} {'Mean (d)':>10} {'Median (h)':>12} {'Median (d)':>12}")
    print("-" * 85)
    
    for row in time_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['merged_count']:>8,} {row['mean_hours']:>10.1f} "
              f"{row['mean_days']:>10.2f} {row['median_hours']:>12.1f} {row['median_days']:>12.2f}")
    
    # Create comparison plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    agents = time_stats['agent'].to_list()
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Human baseline values
    human_mean_days = time_stats['mean_days'].to_list()[0]
    human_median_days = time_stats['median_days'].to_list()[0]
    
    # Mean time to merge (days)
    mean_days = time_stats['mean_days'].to_list()
    bars = ax1.bar(agents, mean_days, color=colors, alpha=0.8)
    ax1.axhline(y=human_mean_days, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax1.set_title('Average Time to Merge by Agent', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Time to Merge (days)')
    ax1.set_xlabel('Agent')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    for bar, val in zip(bars, mean_days):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_days)*0.02,
                f'{val:.1f}d', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Median time to merge (days)
    median_days = time_stats['median_days'].to_list()
    bars = ax2.bar(agents, median_days, color=colors, alpha=0.8)
    ax2.axhline(y=human_median_days, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax2.set_title('Median Time to Merge by Agent', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Median Time to Merge (days)')
    ax2.set_xlabel('Agent')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, val in zip(bars, median_days):
        # Handle case where median might be 0
        y_offset = max(0.01, max(median_days)*0.02)
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                f'{val:.2f}d', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Box plot comparison - ensure Human is first
    agent_list = agents  # Use the same sorted order
    time_data = [merged_df.filter(pl.col("agent") == agent)["time_to_merge_hours"].to_list() 
                 for agent in agent_list]
    # Convert to days for better readability
    time_data_days = [[h/24 for h in hours] for hours in time_data]
    
    bp = ax3.boxplot(time_data_days, tick_labels=agent_list, patch_artist=True)
    for patch, agent in zip(bp['boxes'], agent_list):
        color = '#FF6B6B' if agent == 'Human' else '#4ECDC4'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_title('Time to Merge Distribution by Agent', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time to Merge (days)')
    ax3.set_xlabel('Agent')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_path / 'pr_time_to_merge_by_agent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved time to merge comparison to {plots_path / 'pr_time_to_merge_by_agent.png'}")


def analyze_substantive_review_prs(df: pl.DataFrame):
    """Analyze PRs that received substantive review (not instant auto-merge).
    
    Filters for PRs that meet ANY of these criteria:
    - Has at least 1 review
    - Has comments above the 10th percentile for that agent
    - Time to merge >= 10 minutes (0.167 hours)
    - At least 2 seconds per changed line
    """
    print("\n" + "=" * 80)
    print("7. SUBSTANTIVE REVIEW ANALYSIS (Excluding Instant Auto-Merges)")
    print("=" * 80)
    
    # First, calculate the p10 threshold for comments per agent
    comment_p10 = df.group_by("agent").agg([
        pl.col("comments_count").quantile(0.10).alias("p10_comments")
    ])
    
    # Join back to get the threshold for each row
    df_with_p10 = df.join(comment_p10, on="agent", how="left")
    
    # Calculate seconds per line changed
    df_with_p10 = df_with_p10.with_columns([
        (pl.col("time_to_merge_hours") * 3600 / pl.col("total_changes"))
            .fill_nan(0)
            .fill_null(0)
            .alias("seconds_per_line")
    ])
    
    # Apply filters - PR needs to meet AT LEAST ONE criterion
    substantive_df = df_with_p10.filter(
        (pl.col("reviews_count") >= 1) |
        (pl.col("comments_count") > pl.col("p10_comments")) |
        (pl.col("time_to_merge_hours") >= 0.167) |  # 10 minutes
        (pl.col("seconds_per_line") >= 2.0)
    )
    
    # Focus on merged PRs for time analysis
    merged_substantive = substantive_df.filter(pl.col("is_merged"))
    
    print("\n" + "=" * 60)
    print("FILTERING CRITERIA (PRs meeting ANY of these):")
    print("=" * 60)
    print("  • Has at least 1 review")
    print("  • Comments > p10 for that agent")
    print("  • Time to merge >= 10 minutes")
    print("  • >= 2 seconds per changed line")
    
    # Show how many PRs pass the filter
    print("\n" + "=" * 60)
    print("PR Count: All vs Substantive Review")
    print("=" * 60)
    
    filter_stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        (pl.col("is_merged")).sum().alias("total_merged"),
    ])
    
    substantive_stats = substantive_df.group_by("agent").agg([
        pl.len().alias("substantive_prs"),
        (pl.col("is_merged")).sum().alias("substantive_merged"),
    ])
    
    filter_comparison = filter_stats.join(substantive_stats, on="agent", how="left")
    filter_comparison = filter_comparison.with_columns([
        (pl.col("substantive_prs") / pl.col("total_prs") * 100).alias("pct_substantive"),
        (pl.col("substantive_merged") / pl.col("substantive_prs") * 100).alias("substantive_merge_rate"),
        (pl.col("total_merged") / pl.col("total_prs") * 100).alias("overall_merge_rate"),
    ])
    
    filter_comparison = sort_agents_human_first(filter_comparison)
    
    print(f"{'Agent':<10} {'Total':>8} {'Subst.':>8} {'% Subst.':>9} {'Subst. Merged':>14} "
          f"{'Subst. MR%':>12} {'Overall MR%':>12}")
    print("-" * 90)
    
    for row in filter_comparison.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total_prs']:>8,} {row['substantive_prs']:>8,} "
              f"{row['pct_substantive']:>8.1f}% {row['substantive_merged']:>14,} "
              f"{row['substantive_merge_rate']:>11.1f}% {row['overall_merge_rate']:>11.1f}%")
    
    # Now analyze time to merge for substantive review PRs
    print("\n" + "=" * 80)
    print("TIME TO MERGE: Substantive Review PRs Only")
    print("=" * 80)
    
    time_stats = merged_substantive.group_by("agent").agg([
        pl.len().alias("count"),
        pl.col("time_to_merge_hours").mean().alias("mean_hours"),
        pl.col("time_to_merge_hours").median().alias("median_hours"),
        pl.col("time_to_merge_hours").quantile(0.25).alias("p25_hours"),
        pl.col("time_to_merge_hours").quantile(0.75).alias("p75_hours"),
        pl.col("time_to_merge_hours").quantile(0.90).alias("p90_hours"),
    ])
    
    time_stats = time_stats.with_columns([
        (pl.col("mean_hours") / 24).alias("mean_days"),
        (pl.col("median_hours") / 24).alias("median_days"),
    ])
    
    time_stats = sort_agents_human_first(time_stats)
    
    print(f"{'Agent':<10} {'Count':>8} {'Mean (h)':>10} {'Mean (d)':>10} "
          f"{'Median (h)':>12} {'Median (d)':>12} {'P90 (h)':>10}")
    print("-" * 90)
    
    for row in time_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['count']:>8,} {row['mean_hours']:>10.1f} "
              f"{row['mean_days']:>10.2f} {row['median_hours']:>12.1f} "
              f"{row['median_days']:>12.2f} {row['p90_hours']:>10.1f}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    agents = time_stats['agent'].to_list()
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    # Human baseline
    human_mean_hours = time_stats.filter(pl.col("agent") == "Human")["mean_hours"].to_list()[0]
    human_median_hours = time_stats.filter(pl.col("agent") == "Human")["median_hours"].to_list()[0]
    
    # Plot 1: Mean time to merge
    mean_hours = time_stats['mean_hours'].to_list()
    bars = ax1.bar(agents, mean_hours, color=colors, alpha=0.8)
    ax1.axhline(y=human_mean_hours, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax1.set_title('Mean Time to Merge (Substantive Review PRs)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Time (hours)')
    ax1.set_xlabel('Agent')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    for bar, val in zip(bars, mean_hours):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_hours)*0.02,
                f'{val:.1f}h', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Median time to merge
    median_hours = time_stats['median_hours'].to_list()
    bars = ax2.bar(agents, median_hours, color=colors, alpha=0.8)
    ax2.axhline(y=human_median_hours, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax2.set_title('Median Time to Merge (Substantive Review PRs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Median Time (hours)')
    ax2.set_xlabel('Agent')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, val in zip(bars, median_hours):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(median_hours)*0.02,
                f'{val:.1f}h', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: % of PRs that are substantive
    pct_substantive = filter_comparison['pct_substantive'].to_list()
    bars = ax3.bar(agents, pct_substantive, color=colors, alpha=0.8)
    human_pct = filter_comparison.filter(pl.col("agent") == "Human")["pct_substantive"].to_list()[0]
    ax3.axhline(y=human_pct, color='#FF6B6B', linestyle='--', linewidth=2, 
                alpha=0.7, label='Human baseline')
    ax3.set_title('% of PRs with Substantive Review', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_xlabel('Agent')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    for bar, val in zip(bars, pct_substantive):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 4: Box plot comparison
    agent_list = agents
    time_data = [merged_substantive.filter(pl.col("agent") == agent)["time_to_merge_hours"].to_list() 
                 for agent in agent_list]
    
    bp = ax4.boxplot(time_data, tick_labels=agent_list, patch_artist=True)
    for patch, agent in zip(bp['boxes'], agent_list):
        color = '#FF6B6B' if agent == 'Human' else '#4ECDC4'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_title('Time to Merge Distribution (Substantive Review)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time to Merge (hours)')
    ax4.set_xlabel('Agent')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_path / 'pr_substantive_review_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Saved substantive review analysis to {plots_path / 'pr_substantive_review_analysis.png'}")
    
    # Show comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: All PRs vs Substantive Review PRs")
    print("=" * 80)
    
    # Get overall time stats for comparison
    overall_time = df.filter(pl.col("is_merged")).group_by("agent").agg([
        pl.col("time_to_merge_hours").mean().alias("overall_mean_hours"),
        pl.col("time_to_merge_hours").median().alias("overall_median_hours"),
    ])
    
    comparison = time_stats.join(overall_time, on="agent", how="left")
    comparison = comparison.with_columns([
        (pl.col("mean_hours") / pl.col("overall_mean_hours")).alias("mean_ratio"),
        (pl.col("median_hours") / pl.col("overall_median_hours")).alias("median_ratio"),
    ])
    comparison = sort_agents_human_first(comparison)
    
    print(f"{'Agent':<10} {'All Mean (h)':>12} {'Subst Mean (h)':>15} {'Ratio':>7} "
          f"{'All Med (h)':>12} {'Subst Med (h)':>15} {'Ratio':>7}")
    print("-" * 95)
    
    for row in comparison.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['overall_mean_hours']:>12.1f} {row['mean_hours']:>15.1f} "
              f"{row['mean_ratio']:>7.2f}x {row['overall_median_hours']:>12.1f} "
              f"{row['median_hours']:>15.1f} {row['median_ratio']:>7.2f}x")


def create_summary_comparison(df: pl.DataFrame):
    """Create a comprehensive summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON: AGENTS vs HUMAN BASELINE")
    print("=" * 80)
    
    summary = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("is_merged").sum().alias("merged_count"),
        (pl.col("is_merged").sum() / pl.len() * 100).alias("merge_rate"),
        pl.col("body_length").mean().alias("avg_body_length"),
        pl.col("total_changes").mean().alias("avg_total_changes"),
        pl.col("changed_files").mean().alias("avg_files_changed"),
        pl.col("comments_count").mean().alias("avg_comments"),
        pl.col("reviews_count").mean().alias("avg_reviews"),
        pl.col("time_to_merge_hours").mean().alias("avg_time_to_merge_hours"),
    ])
    
    summary = summary.with_columns([
        (pl.col("avg_time_to_merge_hours") / 24).alias("avg_time_to_merge_days")
    ])
    
    # Sort with Human first
    summary = sort_agents_human_first(summary)
    
    print("\nKey Metrics Comparison:")
    print(f"{'Agent':<10} {'Total PRs':>10} {'Merge %':>9} {'Avg Body':>10} {'Avg Changes':>12} "
          f"{'Avg Files':>10} {'Avg Cmt':>9} {'Avg Rev':>9} {'Avg TTM (d)':>12}")
    print("-" * 110)
    
    for row in summary.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total_prs']:>10,} {row['merge_rate']:>8.1f}% "
              f"{row['avg_body_length']:>10.0f} {row['avg_total_changes']:>12.0f} "
              f"{row['avg_files_changed']:>10.1f} {row['avg_comments']:>9.1f} "
              f"{row['avg_reviews']:>9.1f} {row['avg_time_to_merge_days']:>12.2f}")
    
    # Calculate differences from Human baseline
    human_row = summary.filter(pl.col("agent") == "Human").to_dicts()[0]
    
    print("\n" + "=" * 80)
    print("DIFFERENCES FROM HUMAN BASELINE")
    print("=" * 80)
    print(f"{'Agent':<10} {'ΔMerge %':>10} {'ΔBody':>10} {'ΔChanges':>12} {'ΔFiles':>10} "
          f"{'ΔComments':>11} {'ΔReviews':>10} {'ΔTTM (d)':>11}")
    print("-" * 100)
    
    for row in summary.iter_rows(named=True):
        if row['agent'] != 'Human':
            delta_merge = row['merge_rate'] - human_row['merge_rate']
            delta_body = row['avg_body_length'] - human_row['avg_body_length']
            delta_changes = row['avg_total_changes'] - human_row['avg_total_changes']
            delta_files = row['avg_files_changed'] - human_row['avg_files_changed']
            delta_comments = row['avg_comments'] - human_row['avg_comments']
            delta_reviews = row['avg_reviews'] - human_row['avg_reviews']
            delta_ttm = row['avg_time_to_merge_days'] - human_row['avg_time_to_merge_days']
            
            print(f"{row['agent']:<10} {delta_merge:>+9.1f}% {delta_body:>+10.0f} "
                  f"{delta_changes:>+12.0f} {delta_files:>+10.1f} {delta_comments:>+11.1f} "
                  f"{delta_reviews:>+10.1f} {delta_ttm:>+11.2f}")


def main():
    """Main entry point for the analysis script."""
    print("Loading PR data from HuggingFace dataset...")
    
    # Load data for all agents
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    # Prepare data (compute derived columns and collect)
    print("Preparing data...")
    df = prepare_pr_data(df_lazy)
    
    print(f"Loaded {len(df):,} PRs\n")
    
    # Run analyses
    print_overall_stats(df)
    analyze_pr_states_by_agent(df)
    analyze_body_length_by_agent(df)
    analyze_code_changes_by_agent(df)
    analyze_engagement_by_agent(df)
    analyze_commits_by_agent(df)
    analyze_time_to_merge_by_agent(df)
    analyze_substantive_review_prs(df)
    create_summary_comparison(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All plots saved to: {plots_path}")


if __name__ == "__main__":
    main()

