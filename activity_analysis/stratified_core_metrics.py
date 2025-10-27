"""Generate core stratified analyses: merge rate and time-to-merge.

This script produces both fundamental stratified analyses in one run:
1. Merge rate by repository star count (bar chart)
2. Time to merge by repository star count (box plot)

Run with: conda run -n spoiler python -m spoiler.analysis.stratified_core_metrics
"""
import polars as pl
from pathlib import Path
from spoiler.util.plotting_helpers import plots_root
from spoiler.util.plotting_stratified import (
    load_pr_repo_data,
    run_stratified_analysis,
    StratifiedPlotConfig,
)


def run_merge_rate_stratified(
    n_bins: int = 3, 
    quantile_agent: str = "Human", 
    include_all_stars: bool = False
):
    """Generate merge rate analysis stratified by repository star counts.
    
    Args:
        n_bins: Number of star count bins to create (default: 3 for tertiles)
        quantile_agent: Agent to use for computing bin boundaries (default: "Human")
        include_all_stars: If True, add "Overall" panel showing all data (default: False)
    """
    print("\n" + "="*80)
    print("ANALYSIS 1: MERGE RATE STRATIFIED")
    print("="*80 + "\n")
    
    plots_path = plots_root / "main_metrics"
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Data loader
    data_loader = load_pr_repo_data
    
    # Aggregator - compute merge rate per agent × bin
    def aggregate_merge_rates(df: pl.DataFrame) -> pl.DataFrame:
        return df.group_by(["agent", "star_bin"]).agg([
            pl.len().alias("total"),
            pl.col("is_merged").sum().alias("merged"),
        ]).with_columns([
            (pl.col("merged") / pl.col("total") * 100).alias("merge_rate"),
        ])
    
    # Configure plot
    config = StratifiedPlotConfig(
        title='Merge Rate by Repository Star Count',
        ylabel='Merge Rate (%)',
        figsize_per_bin=6.0,
        fig_height=4.5,
        show_distribution=True,  # Always show distribution for merge rate
        use_log_scale=False,
        include_all_stars=include_all_stars,
    )
    
    # Run analysis
    run_stratified_analysis(
        data_loader=data_loader,
        aggregator=aggregate_merge_rates,
        plot_type='bar',
        config=config,
        output_path=plots_path / 'merge_rate_stratified',
        n_bins=n_bins,
        quantile_agent=quantile_agent,
        value_col='merge_rate',
        count_col='merged',
        with_confidence_intervals=True,
        value_format='{:.1f}%',
        ylim=(0, 105),
    )


def run_time_to_merge_stratified(
    n_bins: int = 3, 
    quantile_agent: str = "Human",
    include_all_stars: bool = False
):
    """Generate time-to-merge analysis stratified by repository star counts.
    
    Args:
        n_bins: Number of star count bins to create (default: 3 for tertiles)
        quantile_agent: Agent to use for computing bin boundaries (default: "Human")
        include_all_stars: If True, add "Overall" panel showing all data (default: False)
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: TIME TO MERGE STRATIFIED")
    print("="*80 + "\n")
    
    plots_path = plots_root / "time_to_merge_stratified"
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Define data loader - filter to merged PRs only
    def load_merged_prs():
        df = load_pr_repo_data()
        return df.filter(df["is_merged"])
    
    # No aggregation needed - we plot raw data
    def no_aggregation(df):
        return df
    
    # Configure plot
    config = StratifiedPlotConfig(
        title='Time to Merge by Repository Star Count',
        ylabel='Time to Merge (hours, log scale)',
        figsize_per_bin=6.0,
        fig_height=4.0,
        show_distribution=False,  # Set to True to show distribution histogram
        use_log_scale=True,
        include_all_stars=include_all_stars,
    )
    
    # Run analysis
    run_stratified_analysis(
        data_loader=load_merged_prs,
        aggregator=no_aggregation,
        plot_type='box',
        config=config,
        output_path=plots_path / 'time_to_merge_stratified',
        n_bins=n_bins,
        quantile_agent=quantile_agent,
        value_col='time_to_merge_hours',
    )


def main(n_bins: int = 3, quantile_agent: str = "Human", include_all_stars: bool = False):
    """Run all core stratified analyses.
    
    Args:
        n_bins: Number of star count bins to create (default: 3 for tertiles)
        quantile_agent: Agent to use for computing bin boundaries (default: "Human")
        include_all_stars: If True, add "Overall" panel showing all data (default: False)
    """
    print("="*80)
    print("STRATIFIED CORE METRICS ANALYSIS")
    print("="*80)
    print(f"Configuration: {n_bins} bins based on {quantile_agent} quantiles")
    if include_all_stars:
        print("Including 'Overall' panel with all data combined")
    print()
    
    # Run both analyses
    run_merge_rate_stratified(n_bins=n_bins, quantile_agent=quantile_agent, 
                             include_all_stars=include_all_stars)
    run_time_to_merge_stratified(n_bins=n_bins, quantile_agent=quantile_agent,
                                 include_all_stars=include_all_stars)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print("Plots saved to:")
    print("  - plots/main_metrics/merge_rate_stratified.png/pdf")
    print("  - plots/time_to_merge_stratified/time_to_merge_stratified.png/pdf")


if __name__ == "__main__":
    # Default: 3 bins (tertiles) based on Human data, no Overall panel
    main(n_bins=3, quantile_agent="Human", include_all_stars=True)
    
    # To run with different configurations, use:
    # main(n_bins=3, quantile_agent="Human", include_all_stars=True)  # With Overall panel
    # main(n_bins=4, quantile_agent="Human")  # Quartiles
    # main(n_bins=5, quantile_agent="Human")  # Quintiles

