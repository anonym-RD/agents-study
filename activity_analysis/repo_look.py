"""Analyze repository characteristics from the Repositories table."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


cur_path = Path(__file__).parent
root_path = cur_path.parent.parent
plots_path = root_path / "plots" / "repo_analysis"

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
    
    # Filter to PRs that have star data (not null)
    print("Filtering to PRs with valid star counts...")
    pr_repo_data = pr_repo_data.filter(pl.col("stargazer_count").is_not_null())
    
    print("Collecting data...")
    return pr_repo_data.collect()


def plot_repo_stars_distribution(df: pl.DataFrame):
    """Plot distribution of repository stars for PRs by agent."""
    print("\nCreating repository stars distribution plot...")
    
    # Get data sorted with Human first
    agents_order = ['Human'] + sorted([a for a in df['agent'].unique().to_list() if a != 'Human'])
    
    # Calculate percentiles for each agent
    percentile_stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("stargazer_count").quantile(0.10).alias("p10"),
        pl.col("stargazer_count").quantile(0.25).alias("p25"),
        pl.col("stargazer_count").quantile(0.50).alias("p50"),
        pl.col("stargazer_count").quantile(0.75).alias("p75"),
        pl.col("stargazer_count").quantile(0.90).alias("p90"),
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
        
        # Draw whisker caps (p10 and p90)
        cap_width = 0.3
        # P10 cap
        ax.plot([i - cap_width/2, i + cap_width/2], [p10, p10],
               color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
        # P90 cap
        ax.plot([i - cap_width/2, i + cap_width/2], [p90, p90],
               color=whisker_color, linewidth=3, solid_capstyle='round', zorder=2)
        
        # Dashed lines connecting caps to box
        ax.plot([i, i], [p10, p25], color=whisker_color, linewidth=0.8, 
               linestyle=':', alpha=0.5, zorder=1)
        ax.plot([i, i], [p75, p90], color=whisker_color, linewidth=0.8, 
               linestyle=':', alpha=0.5, zorder=1)
    
    # Human baseline median
    human_median = percentile_stats.filter(pl.col("agent") == "Human")["p50"].to_list()[0]
    ax.axhline(y=human_median, color='#FF6B6B', linestyle='--', linewidth=2, 
               alpha=0.4, label='Human median', zorder=0)
    
    ax.set_ylabel('Repository Stars (log scale)')
    ax.set_xlabel('Agent')
    ax.set_xticks(positions)
    ax.set_xticklabels(agents_order, rotation=45, ha='right')
    ax.set_yscale('log')  # Log scale for star counts
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(plots_path / 'repo_stars_dist.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'repo_stars_dist.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'repo_stars_dist.png'}")


def print_repo_stats(df: pl.DataFrame):
    """Print summary statistics about repositories."""
    print("\n" + "="*80)
    print("REPOSITORY STATISTICS")
    print("="*80)
    
    # Overall stats
    total_prs = len(df)
    unique_repos = df.select("name_with_owner").n_unique()
    
    print(f"\nTotal PRs: {total_prs:,}")
    print(f"Unique repositories: {unique_repos:,}")
    print(f"PRs per repo (avg): {total_prs / unique_repos:.1f}")
    
    # Stats by agent
    print("\nStats by agent:")
    agent_stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("name_with_owner").n_unique().alias("unique_repos"),
        pl.col("stargazer_count").quantile(0.5).alias("median_stars"),
        pl.col("stargazer_count").mean().alias("mean_stars"),
    ]).with_columns([
        (pl.col("total_prs") / pl.col("unique_repos")).alias("prs_per_repo")
    ])
    
    # Sort with Human first
    agent_stats = sort_agents_human_first(agent_stats)
    
    print(agent_stats)
    
    # Top repositories
    print("\n" + "="*80)
    print("TOP 10 REPOSITORIES BY PR COUNT")
    print("="*80)
    
    top_repos = df.group_by(["name_with_owner", "agent"]).agg([
        pl.len().alias("pr_count"),
        pl.col("stargazer_count").first().alias("stars"),
    ]).sort("pr_count", descending=True).head(10)
    
    print(top_repos)
    
    # Top starred repositories
    print("\n" + "="*80)
    print("TOP 10 MOST STARRED REPOSITORIES (by median)")
    print("="*80)
    
    top_starred = df.group_by("name_with_owner").agg([
        pl.len().alias("pr_count"),
        pl.col("stargazer_count").first().alias("stars"),
    ]).sort("stars", descending=True).head(10)
    
    print(top_starred)


def main():
    """Main entry point for repository analysis."""
    print("="*80)
    print("REPOSITORY ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_pr_repo_data()
    
    print(f"\nLoaded {len(df):,} PRs with repository information")
    
    # Print summary statistics
    print_repo_stats(df)
    
    # Generate plots
    plot_repo_stars_distribution(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"All plots saved to: {plots_path}")
    print("  - repo_stars_dist.png/pdf")


if __name__ == "__main__":
    main()

