"""Analyze the fraction of PRs with linked issues by agent."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents
from spoiler.util.plotting_helpers import plot_simple_bar, save_plot, sort_agents_human_first, plots_root


plots_path = plots_root / "linked_issues"

# Create plots directory if it doesn't exist
plots_path.mkdir(parents=True, exist_ok=True)

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def analyze_linked_issues(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate fraction of PRs with linked issues by agent.
    
    Args:
        df: DataFrame with 'agent' and 'closing_issues_count' columns
    
    Returns:
        DataFrame with columns: agent, total, with_issues, fraction_pct
    """
    stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        (pl.col("closing_issues_count") > 0).sum().alias("with_issues"),
    ]).with_columns([
        (pl.col("with_issues") / pl.col("total") * 100).alias("fraction_pct"),
    ])
    
    return sort_agents_human_first(stats)


def plot_linked_issues_fraction(stats: pl.DataFrame):
    """Create bar chart showing fraction of PRs with linked issues.
    
    Args:
        stats: DataFrame from analyze_linked_issues()
    """
    print("Creating linked issues fraction plot...")
    
    fig, ax = plot_simple_bar(
        df=stats,
        value_col="fraction_pct",
        ylabel="PRs with Linked Issues (%)",
        title=None,
        figsize=(5, 2.5),
        with_confidence_intervals=True,
        count_col="total",
        baseline_label="Human baseline",
        value_format="{:.1f}%",
        ylim=(0, 60),
        legend_loc='upper left',
        ylabel_fontsize=11,
        xticklabel_rotation=0,
    )
    
    plt.tight_layout()
    save_plot(fig, plots_path / 'linked_issues_fraction')


def print_summary_stats(stats: pl.DataFrame):
    """Print summary statistics for linked issues analysis."""
    print("\n" + "="*80)
    print("LINKED ISSUES SUMMARY")
    print("="*80)
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'With Issues':>15} {'Fraction':>12}")
    print("-" * 54)
    
    for row in stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total']:>12,} {row['with_issues']:>15,} "
              f"{row['fraction_pct']:>11.1f}%")


def main():
    """Main entry point for linked issues analysis."""
    print("="*80)
    print("LINKED ISSUES ANALYSIS")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Collecting data...")
    df = df_lazy.collect()
    
    print(f"Loaded {len(df):,} PRs\n")
    
    # Analyze linked issues
    stats = analyze_linked_issues(df)
    
    # Print summary
    print_summary_stats(stats)
    
    # Generate plot
    plot_linked_issues_fraction(stats)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Plot saved to: {plots_path}")
    print("  - linked_issues_fraction.png/pdf")


if __name__ == "__main__":
    main()

