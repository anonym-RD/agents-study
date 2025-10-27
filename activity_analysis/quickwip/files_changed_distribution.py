"""Show distribution of files changed per PR by agent."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents
from spoiler.util.plotting_helpers import plot_box_distribution, save_plot, plots_root


plots_path = plots_root / "files_changed"
# Create plots directory if it doesn't exist
plots_path.mkdir(parents=True, exist_ok=True)

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_files_changed_distribution(df: pl.DataFrame):
    """Create box plot showing distribution of files changed per PR by agent.
    
    Note: This uses the changed_files column at the PR level, not the detailed
    files list (which has limited coverage as shown in check_file_data_coverage.py).
    """
    print("Creating files changed distribution plot...")
    
    fig, ax = plot_box_distribution(
        df=df,
        value_col="changed_files",
        ylabel="Files Changed per PR",
        title="Distribution of Files Changed per PR",
        figsize=(8, 6),
        use_log_scale=False,
        baseline_label="Human median",
    )
    
    plt.tight_layout()
    save_plot(fig, plots_path / 'files_changed_distribution')


def print_summary_stats(df: pl.DataFrame):
    """Print summary statistics for files changed distribution."""
    print("\n" + "="*80)
    print("FILES CHANGED DISTRIBUTION SUMMARY")
    print("="*80)
    
    stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("changed_files").mean().alias("mean"),
        pl.col("changed_files").median().alias("median"),
        pl.col("changed_files").quantile(0.10).alias("p10"),
        pl.col("changed_files").quantile(0.25).alias("p25"),
        pl.col("changed_files").quantile(0.75).alias("p75"),
        pl.col("changed_files").quantile(0.90).alias("p90"),
        pl.col("changed_files").min().alias("min"),
        pl.col("changed_files").max().alias("max"),
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
    """Main entry point for files changed distribution analysis."""
    print("="*80)
    print("FILES CHANGED DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Collecting data...")
    df = df_lazy.collect()
    
    print(f"Loaded {len(df):,} PRs\n")
    
    # Print summary stats
    print_summary_stats(df)
    
    # Generate plot
    plot_files_changed_distribution(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Plot saved to: {plots_path}")
    print("  - files_changed_distribution.png/pdf")


if __name__ == "__main__":
    main()

