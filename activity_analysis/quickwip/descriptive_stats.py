"""Show descriptive statistics for key fields in the dataset without generating plots."""
import polars as pl
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents
from spoiler.util.plotting_helpers import sort_agents_human_first


def analyze_body_size(df: pl.DataFrame):
    """Analyze PR body size statistics by agent."""
    print("\n" + "="*80)
    print("PR BODY SIZE STATISTICS")
    print("="*80)
    
    # Calculate body length and stats
    body_stats = df.with_columns([
        pl.col("body").str.len_chars().fill_null(0).alias("body_length"),
        pl.col("body").is_null().alias("is_null_body"),
    ]).group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("body_length").mean().alias("mean_length"),
        pl.col("body_length").median().alias("median_length"),
        pl.col("body_length").quantile(0.25).alias("p25_length"),
        pl.col("body_length").quantile(0.75).alias("p75_length"),
        pl.col("body_length").min().alias("min_length"),
        pl.col("body_length").max().alias("max_length"),
        pl.col("is_null_body").sum().alias("null_count"),
        (pl.col("body_length") == 0).sum().alias("empty_count"),
    ]).with_columns([
        (pl.col("null_count") / pl.col("total_prs") * 100).alias("null_pct"),
        (pl.col("empty_count") / pl.col("total_prs") * 100).alias("empty_pct"),
    ])
    
    body_stats = sort_agents_human_first(body_stats)
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'Mean':>10} {'Median':>10} {'P25':>10} {'P75':>10} {'Min':>10} {'Max':>10}")
    print("-" * 90)
    
    for row in body_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['mean_length']:>10.1f} "
              f"{row['median_length']:>10.0f} {row['p25_length']:>10.0f} "
              f"{row['p75_length']:>10.0f} {row['min_length']:>10.0f} {row['max_length']:>10,.0f}")
    
    print(f"\n{'Agent':<12} {'Null Body':>15} {'Empty Body':>15}")
    print("-" * 45)
    
    for row in body_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['null_count']:>10,} ({row['null_pct']:>5.1f}%) "
              f"{row['empty_count']:>5,} ({row['empty_pct']:>5.1f}%)")


def analyze_state(df: pl.DataFrame):
    """Analyze PR state distribution by agent."""
    print("\n" + "="*80)
    print("PR STATE DISTRIBUTION")
    print("="*80)
    
    # Get state counts
    state_counts = df.group_by(["agent", "state"]).agg([
        pl.len().alias("count")
    ])
    
    # Get total per agent
    totals = df.group_by("agent").agg([
        pl.len().alias("total")
    ])
    
    # Join and calculate percentages
    state_stats = state_counts.join(totals, on="agent").with_columns([
        (pl.col("count") / pl.col("total") * 100).alias("percentage")
    ]).sort(["agent", "state"])
    
    # Get unique states
    states = sorted(df["state"].unique().to_list())
    
    # Get agents in order
    agents_order = ["Human"] + sorted([a for a in df["agent"].unique().to_list() if a != "Human"])
    
    print(f"\n{'Agent':<12}", end="")
    for state in states:
        print(f"{state:>15}", end="")
    print()
    print("-" * (12 + 15 * len(states)))
    
    for agent in agents_order:
        agent_data = state_stats.filter(pl.col("agent") == agent)
        print(f"{agent:<12}", end="")
        for state in states:
            state_row = agent_data.filter(pl.col("state") == state)
            if len(state_row) > 0:
                count = state_row["count"].to_list()[0]
                pct = state_row["percentage"].to_list()[0]
                print(f"{count:>8,} ({pct:>4.1f}%)", end="")
            else:
                print(f"{'0':>8} ({'0.0':>4}%)", end="")
        print()


def analyze_labels(df: pl.DataFrame):
    """Analyze label_count distribution by agent."""
    print("\n" + "="*80)
    print("LABEL COUNT STATISTICS")
    print("="*80)
    
    label_stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("label_count").mean().alias("mean_labels"),
        pl.col("label_count").median().alias("median_labels"),
        pl.col("label_count").quantile(0.25).alias("p25_labels"),
        pl.col("label_count").quantile(0.75).alias("p75_labels"),
        pl.col("label_count").min().alias("min_labels"),
        pl.col("label_count").max().alias("max_labels"),
        (pl.col("label_count") == 0).sum().alias("no_labels_count"),
        (pl.col("label_count") > 0).sum().alias("has_labels_count"),
    ]).with_columns([
        (pl.col("no_labels_count") / pl.col("total_prs") * 100).alias("no_labels_pct"),
        (pl.col("has_labels_count") / pl.col("total_prs") * 100).alias("has_labels_pct"),
    ])
    
    label_stats = sort_agents_human_first(label_stats)
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'Min':>8} {'Max':>8}")
    print("-" * 80)
    
    for row in label_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['mean_labels']:>8.2f} "
              f"{row['median_labels']:>8.0f} {row['p25_labels']:>8.0f} "
              f"{row['p75_labels']:>8.0f} {row['min_labels']:>8.0f} {row['max_labels']:>8.0f}")
    
    print(f"\n{'Agent':<12} {'No Labels':>18} {'Has Labels':>18}")
    print("-" * 50)
    
    for row in label_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['no_labels_count']:>10,} ({row['no_labels_pct']:>5.1f}%) "
              f"{row['has_labels_count']:>7,} ({row['has_labels_pct']:>5.1f}%)")


def analyze_time_diffs(df: pl.DataFrame):
    """Analyze time differences (updated_at - published_at) by agent."""
    print("\n" + "="*80)
    print("TIME DIFFERENCE STATISTICS (updated_at - published_at)")
    print("="*80)
    
    # Parse datetime columns and calculate difference
    time_stats = df.with_columns([
        pl.col("published_at").str.to_datetime(),
        pl.col("updated_at").str.to_datetime(),
    ]).with_columns([
        ((pl.col("updated_at") - pl.col("published_at")).dt.total_seconds() / 3600.0).alias("time_diff_hours")
    ]).filter(
        # Filter out null time diffs
        pl.col("time_diff_hours").is_not_null()
    ).group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("time_diff_hours").mean().alias("mean_hours"),
        pl.col("time_diff_hours").median().alias("median_hours"),
        pl.col("time_diff_hours").quantile(0.25).alias("p25_hours"),
        pl.col("time_diff_hours").quantile(0.75).alias("p75_hours"),
        pl.col("time_diff_hours").min().alias("min_hours"),
        pl.col("time_diff_hours").max().alias("max_hours"),
        (pl.col("time_diff_hours") == 0).sum().alias("zero_diff_count"),
    ]).with_columns([
        (pl.col("zero_diff_count") / pl.col("total_prs") * 100).alias("zero_diff_pct"),
    ])
    
    time_stats = sort_agents_human_first(time_stats)
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'Mean (hrs)':>12} {'Median (hrs)':>14} {'P25 (hrs)':>12} {'P75 (hrs)':>12}")
    print("-" * 78)
    
    for row in time_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['mean_hours']:>12.1f} "
              f"{row['median_hours']:>14.1f} {row['p25_hours']:>12.1f} {row['p75_hours']:>12.1f}")
    
    print(f"\n{'Agent':<12} {'Min (hrs)':>12} {'Max (hrs)':>12} {'Zero Diff':>18}")
    print("-" * 58)
    
    for row in time_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['min_hours']:>12.1f} {row['max_hours']:>12,.1f} "
              f"{row['zero_diff_count']:>10,} ({row['zero_diff_pct']:>5.1f}%)")
    
    # Convert to more readable units for display
    print("\n(Note: Hours converted to days for reference)")
    print(f"{'Agent':<12} {'Mean (days)':>12} {'Median (days)':>14} {'P25 (days)':>12} {'P75 (days)':>12}")
    print("-" * 64)
    
    for row in time_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['mean_hours']/24:>12.2f} "
              f"{row['median_hours']/24:>14.2f} {row['p25_hours']/24:>12.2f} {row['p75_hours']/24:>12.2f}")


def analyze_commits_count(df: pl.DataFrame):
    """Analyze commits_count distribution by agent."""
    print("\n" + "="*80)
    print("COMMITS COUNT STATISTICS")
    print("="*80)
    
    commits_stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("commits_count").mean().alias("mean_commits"),
        pl.col("commits_count").median().alias("median_commits"),
        pl.col("commits_count").quantile(0.25).alias("p25_commits"),
        pl.col("commits_count").quantile(0.75).alias("p75_commits"),
        pl.col("commits_count").min().alias("min_commits"),
        pl.col("commits_count").max().alias("max_commits"),
        (pl.col("commits_count") == 1).sum().alias("single_commit_count"),
    ]).with_columns([
        (pl.col("single_commit_count") / pl.col("total_prs") * 100).alias("single_commit_pct"),
    ])
    
    commits_stats = sort_agents_human_first(commits_stats)
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'Min':>8} {'Max':>8}")
    print("-" * 80)
    
    for row in commits_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['mean_commits']:>8.2f} "
              f"{row['median_commits']:>8.0f} {row['p25_commits']:>8.0f} "
              f"{row['p75_commits']:>8.0f} {row['min_commits']:>8.0f} {row['max_commits']:>8.0f}")
    
    print(f"\n{'Agent':<12} {'Single Commit PRs':>25}")
    print("-" * 40)
    
    for row in commits_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['single_commit_count']:>15,} ({row['single_commit_pct']:>5.1f}%)")


def analyze_files_changed(df: pl.DataFrame):
    """Analyze changed_files distribution by agent."""
    print("\n" + "="*80)
    print("FILES CHANGED STATISTICS")
    print("="*80)
    
    files_stats = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("changed_files").mean().alias("mean_files"),
        pl.col("changed_files").median().alias("median_files"),
        pl.col("changed_files").quantile(0.25).alias("p25_files"),
        pl.col("changed_files").quantile(0.75).alias("p75_files"),
        pl.col("changed_files").min().alias("min_files"),
        pl.col("changed_files").max().alias("max_files"),
        (pl.col("changed_files") == 1).sum().alias("single_file_count"),
    ]).with_columns([
        (pl.col("single_file_count") / pl.col("total_prs") * 100).alias("single_file_pct"),
    ])
    
    files_stats = sort_agents_human_first(files_stats)
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'Min':>8} {'Max':>8}")
    print("-" * 80)
    
    for row in files_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['mean_files']:>8.2f} "
              f"{row['median_files']:>8.0f} {row['p25_files']:>8.0f} "
              f"{row['p75_files']:>8.0f} {row['min_files']:>8.0f} {row['max_files']:>8.0f}")
    
    print(f"\n{'Agent':<12} {'Single File PRs':>25}")
    print("-" * 40)
    
    for row in files_stats.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['single_file_count']:>15,} ({row['single_file_pct']:>5.1f}%)")


def main():
    """Main entry point for descriptive statistics."""
    print("="*80)
    print("DESCRIPTIVE STATISTICS FOR KEY FIELDS")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Collecting data...")
    df = df_lazy.collect()
    
    print(f"Loaded {len(df):,} PRs")
    
    # Run all analyses
    analyze_body_size(df)
    analyze_state(df)
    analyze_labels(df)
    analyze_time_diffs(df)
    analyze_commits_count(df)
    analyze_files_changed(df)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

