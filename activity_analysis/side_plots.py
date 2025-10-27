"""Generate supplementary plots for additional metrics."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


cur_path = Path(__file__).parent
root_path = cur_path.parent.parent
plots_path = root_path / "plots" / "side_plots"

# Create plots directory if it doesn't exist
plots_path.mkdir(parents=True, exist_ok=True)

# Set consistent style (matching main_metrics_overview.py)
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


def plot_files_changed(df: pl.DataFrame):
    """Plot median number of files changed by agent."""
    print("Creating files changed plot...")
    
    files_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("changed_files").quantile(0.5).alias("median_files_changed"),
    ])
    
    # Sort with Human first
    files_stats = sort_agents_human_first(files_stats)
    
    agents = files_stats['agent'].to_list()
    median_files = files_stats['median_files_changed'].to_list()
    
    # Human baseline
    human_median = median_files[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(agents))
    width = 0.6
    
    # Color scheme: Human gets distinct color, others get a different color
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    bars = ax.bar(x, median_files, width, color=colors, alpha=0.8)
    
    # Add horizontal line for Human baseline
    ax.axhline(y=human_median, color='#FF6B6B', linestyle='--', linewidth=2, 
               alpha=0.5, label='Human baseline')
    
    ax.set_ylabel('Median Files Changed')
    ax.set_xlabel('Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, median_files)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(median_files)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_path / 'files_changed.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'files_changed.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'files_changed.png'}")


def plot_addition_ratio(df: pl.DataFrame):
    """Plot addition ratio: additions / (additions + deletions) by agent."""
    print("Creating addition ratio plot...")
    
    # Impute null additions/deletions with 0
    df = df.with_columns([
        pl.col("additions").fill_null(0),
        pl.col("deletions").fill_null(0),
    ])
    
    # Calculate addition ratio for each PR
    df_with_ratio = df.with_columns([
        (pl.col("additions") / (pl.col("additions") + pl.col("deletions"))).alias("add_ratio")
    ])
    
    # Filter out PRs with zero changes (0/0 -> NaN)
    df_with_ratio = df_with_ratio.filter(pl.col("add_ratio").is_nan().not_())
    
    ratio_stats = df_with_ratio.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("add_ratio").quantile(0.5).alias("median_add_ratio"),
    ])
    
    # Sort with Human first
    ratio_stats = sort_agents_human_first(ratio_stats)
    
    agents = ratio_stats['agent'].to_list()
    median_ratios = ratio_stats['median_add_ratio'].to_list()
    
    # Human baseline
    human_median = median_ratios[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(agents))
    width = 0.6
    
    # Color scheme: Human gets distinct color, others get a different color
    colors = ['#FF6B6B' if agent == 'Human' else '#4ECDC4' for agent in agents]
    
    bars = ax.bar(x, median_ratios, width, color=colors, alpha=0.8)
    
    # Add horizontal line for Human baseline
    ax.axhline(y=human_median, color='#FF6B6B', linestyle='--', linewidth=2, 
               alpha=0.5, label='Human baseline')
    
    ax.set_ylabel('Median Addition Ratio')
    ax.set_xlabel('Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.set_ylim(0, 1)  # Ratio is between 0 and 1
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, median_ratios)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_path / 'addition_ratio.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'addition_ratio.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'addition_ratio.png'}")


def plot_files_changed_distribution(df: pl.DataFrame):
    """Plot distribution of files changed (custom box plot with p10-p90)."""
    print("Creating files changed distribution plot...")
    
    # Get data sorted with Human first
    agents_order = ['Human'] + sorted([a for a in df['agent'].unique().to_list() if a != 'Human'])
    
    # Calculate percentiles for each agent
    percentile_stats = df.group_by("agent").agg([
        pl.col("changed_files").quantile(0.10).alias("p10"),
        pl.col("changed_files").quantile(0.25).alias("p25"),
        pl.col("changed_files").quantile(0.50).alias("p50"),
        pl.col("changed_files").quantile(0.75).alias("p75"),
        pl.col("changed_files").quantile(0.90).alias("p90"),
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
    
    ax.set_ylabel('Files Changed')
    ax.set_xlabel('Agent')
    ax.set_xticks(positions)
    ax.set_xticklabels(agents_order, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(plots_path / 'files_changed_dist.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'files_changed_dist.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'files_changed_dist.png'}")


def plot_addition_ratio_distribution(df: pl.DataFrame):
    """Plot distribution of addition ratio (custom box plot with p10-p90)."""
    print("Creating addition ratio distribution plot...")
    
    # Impute null additions/deletions with 0
    df = df.with_columns([
        pl.col("additions").fill_null(0),
        pl.col("deletions").fill_null(0),
    ])
    
    # Calculate addition ratio for each PR
    df_with_ratio = df.with_columns([
        (pl.col("additions") / (pl.col("additions") + pl.col("deletions"))).alias("add_ratio")
    ])
    
    # Filter out PRs with zero changes (0/0 -> NaN)
    df_with_ratio = df_with_ratio.filter(pl.col("add_ratio").is_nan().not_())
    
    # Get data sorted with Human first
    agents_order = ['Human'] + sorted([a for a in df_with_ratio['agent'].unique().to_list() if a != 'Human'])
    
    # Calculate percentiles for each agent
    percentile_stats = df_with_ratio.group_by("agent").agg([
        pl.col("add_ratio").quantile(0.10).alias("p10"),
        pl.col("add_ratio").quantile(0.25).alias("p25"),
        pl.col("add_ratio").quantile(0.50).alias("p50"),
        pl.col("add_ratio").quantile(0.75).alias("p75"),
        pl.col("add_ratio").quantile(0.90).alias("p90"),
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
    
    ax.set_ylabel('Addition Ratio')
    ax.set_xlabel('Agent')
    ax.set_xticks(positions)
    ax.set_xticklabels(agents_order, rotation=45, ha='right')
    ax.set_ylim(0, 1)  # Ratio is between 0 and 1
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(plots_path / 'addition_ratio_dist.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_path / 'addition_ratio_dist.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {plots_path / 'addition_ratio_dist.png'}")


def main():
    """Main entry point for the side plots script."""
    print("="*80)
    print("GENERATING SIDE PLOTS")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Collecting data...")
    df = df_lazy.collect()
    
    print(f"Loaded {len(df):,} PRs\n")
    
    # Generate plots
    plot_files_changed(df)
    plot_addition_ratio(df)
    plot_files_changed_distribution(df)
    plot_addition_ratio_distribution(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"All plots saved to: {plots_path}")
    print("  - files_changed.png/pdf")
    print("  - addition_ratio.png/pdf")
    print("  - files_changed_dist.png/pdf")
    print("  - addition_ratio_dist.png/pdf")


if __name__ == "__main__":
    main()

