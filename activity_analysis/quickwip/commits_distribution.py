"""Show distribution of commits per PR by agent."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents
from spoiler.util.plotting_helpers import plot_box_distribution, save_plot, plots_root


plots_path = plots_root / "commits"
# Create plots directory if it doesn't exist
plots_path.mkdir(parents=True, exist_ok=True)

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_commits_distribution(df: pl.DataFrame):
    """Create box plot showing distribution of commits per PR by agent."""
    print("Creating commits distribution plot...")
    
    fig, ax = plot_box_distribution(
        df=df,
        value_col="commits_count",
        ylabel="Commits per PR",
        title="Distribution of Commits per PR",
        figsize=(8, 6),
        use_log_scale=False,
        baseline_label="Human median",
    )
    
    plt.tight_layout()
    save_plot(fig, plots_path / 'commits_distribution')


def print_summary_stats(df: pl.DataFrame):
    """Print summary statistics for commits distribution."""
    print("\n" + "="*80)
    print("COMMITS DISTRIBUTION SUMMARY")
    print("="*80)
    
    stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("commits_count").mean().alias("mean"),
        pl.col("commits_count").median().alias("median"),
        pl.col("commits_count").quantile(0.10).alias("p10"),
        pl.col("commits_count").quantile(0.25).alias("p25"),
        pl.col("commits_count").quantile(0.75).alias("p75"),
        pl.col("commits_count").quantile(0.90).alias("p90"),
        pl.col("commits_count").min().alias("min"),
        pl.col("commits_count").max().alias("max"),
    ]).sort("agent")
    
    # Sort with Human first
    human_row = stats.filter(pl.col("agent") == "Human")
    other_rows = stats.filter(pl.col("agent") != "Human").sort("agent")
    stats = pl.concat([human_row, other_rows])
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'Mean':>8} {'Median':>8} {'P10':>8} {'P25':>8} {'P75':>8} {'P90':>8} {'Min':>8} {'Max':>8}")
    print("-" * 100)
    
    for row in stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['mean']:>8.2f} "
              f"{row['median']:>8.0f} {row['p10']:>8.0f} {row['p25']:>8.0f} "
              f"{row['p75']:>8.0f} {row['p90']:>8.0f} {row['min']:>8.0f} {row['max']:>8,.0f}")


def main():
    """Main entry point for commits distribution analysis."""
    print("="*80)
    print("COMMITS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Collecting data...")
    df = df_lazy.collect()
    
    print(f"Loaded {len(df):,} PRs\n")
    
    # Print summary stats
    print_summary_stats(df)
    
    # Generate plot
    plot_commits_distribution(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Plot saved to: {plots_path}")
    print("  - commits_distribution.png/pdf")


if __name__ == "__main__":
    main()

