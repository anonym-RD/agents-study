"""Explore PRs that don't have file-level data to understand why and how prevalent it is."""
import polars as pl
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


def main():
    """Investigate PRs without file data."""
    print("="*80)
    print("EXPLORING PRs WITHOUT FILE DATA")
    print("="*80)
    
    print("\nLoading PR data...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    # Load all relevant columns
    df = df_lazy.select([
        pl.col("id"),
        pl.col("agent"),
        pl.col("number"),
        pl.col("url"),
        pl.col("state"),
        pl.col("additions").fill_null(0).alias("additions"),
        pl.col("deletions").fill_null(0).alias("deletions"),
        pl.col("changed_files").fill_null(0).alias("changed_files"),
        pl.col("files"),
    ]).with_columns([
        (pl.col("additions") + pl.col("deletions")).alias("total_changes"),
        pl.col("files").is_not_null().alias("has_files_field"),
        pl.col("files").list.len().fill_null(0).alias("files_count"),
        (pl.col("files").list.len() > 0).fill_null(False).alias("has_files_data"),
    ]).collect()
    
    print(f"Loaded {len(df):,} PRs")
    
    # Identify PRs without file data
    no_files = df.filter(~pl.col("has_files_data"))
    has_files = df.filter(pl.col("has_files_data"))
    
    print(f"\n{'Category':<40} {'Count':>15} {'Percentage':>12}")
    print("-" * 70)
    print(f"{'Total PRs:':<40} {len(df):>15,} {'100.0%':>12}")
    print(f"{'PRs WITH file data:':<40} {len(has_files):>15,} {100*len(has_files)/len(df):>11.1f}%")
    print(f"{'PRs WITHOUT file data:':<40} {len(no_files):>15,} {100*len(no_files)/len(df):>11.1f}%")
    
    # THEORY 1: Giant edits that error out
    print("\n" + "="*80)
    print("THEORY 1: Large PRs that might have errored out")
    print("="*80)
    
    # Compare size distribution
    print("\nSize distribution (additions + deletions):")
    print(f"\n{'Metric':<30} {'PRs WITH files':>20} {'PRs WITHOUT files':>20}")
    print("-" * 75)
    
    for percentile in [50, 75, 90, 95, 99, 100]:
        with_files_val = has_files["total_changes"].quantile(percentile/100)
        without_files_val = no_files["total_changes"].quantile(percentile/100)
        print(f"{'p' + str(percentile) + ':':<30} {with_files_val:>20,.0f} {without_files_val:>20,.0f}")
    
    # Check for very large PRs without files
    large_threshold = 10000
    large_no_files = no_files.filter(pl.col("total_changes") > large_threshold)
    print(f"\nPRs without files with >{large_threshold:,} changes: {len(large_no_files):,} "
          f"({100*len(large_no_files)/len(no_files):.1f}% of no-files PRs)")
    
    # THEORY 2: Zero or very small edits
    print("\n" + "="*80)
    print("THEORY 2: PRs with zero or very few changes")
    print("="*80)
    
    zero_changes_no_files = no_files.filter(pl.col("total_changes") == 0)
    small_changes_no_files = no_files.filter((pl.col("total_changes") > 0) & (pl.col("total_changes") <= 10))
    
    print(f"\nPRs without files with ZERO changes: {len(zero_changes_no_files):,} "
          f"({100*len(zero_changes_no_files)/len(no_files):.1f}% of no-files PRs)")
    print(f"PRs without files with 1-10 changes: {len(small_changes_no_files):,} "
          f"({100*len(small_changes_no_files)/len(no_files):.1f}% of no-files PRs)")
    
    # Compare zero-change rates
    zero_changes_with_files = has_files.filter(pl.col("total_changes") == 0)
    print(f"\nFor comparison:")
    print(f"  - PRs WITH files that have zero changes: {len(zero_changes_with_files):,} "
          f"({100*len(zero_changes_with_files)/len(has_files):.1f}% of has-files PRs)")
    
    # Breakdown by agent
    print("\n" + "="*80)
    print("BREAKDOWN BY AGENT")
    print("="*80)
    
    by_agent = df.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("has_files_data").sum().alias("with_files"),
        (~pl.col("has_files_data")).sum().alias("without_files"),
        pl.col("total_changes").sum().alias("total_changes_all"),
        pl.col("total_changes").filter(~pl.col("has_files_data")).sum().alias("changes_without_files"),
    ]).with_columns([
        (pl.col("without_files") / pl.col("total_prs") * 100).alias("pct_without_files"),
        (pl.col("changes_without_files") / pl.col("total_changes_all") * 100).alias("pct_changes_without_files"),
    ])
    
    # Sort with Human first
    human_row = by_agent.filter(pl.col("agent") == "Human")
    other_rows = by_agent.filter(pl.col("agent") != "Human").sort("agent")
    by_agent_sorted = pl.concat([human_row, other_rows])
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'Without Files':>15} {'%':>8} "
          f"{'Changes w/o Files':>20} {'% of Changes':>15}")
    print("-" * 95)
    
    for row in by_agent_sorted.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['without_files']:>15,} "
              f"{row['pct_without_files']:>7.1f}% {row['changes_without_files']:>20,} "
              f"{row['pct_changes_without_files']:>14.1f}%")
    
    # SAMPLE PRs without file data
    print("\n" + "="*80)
    print("SAMPLE URLs OF PRs WITHOUT FILE DATA")
    print("="*80)
    
    # Sample different categories
    categories = [
        ("Zero changes", no_files.filter(pl.col("total_changes") == 0)),
        ("Small changes (1-100)", no_files.filter((pl.col("total_changes") > 0) & (pl.col("total_changes") <= 100))),
        ("Medium changes (100-1000)", no_files.filter((pl.col("total_changes") > 100) & (pl.col("total_changes") <= 1000))),
        ("Large changes (1000-10000)", no_files.filter((pl.col("total_changes") > 1000) & (pl.col("total_changes") <= 10000))),
        ("Giant changes (>10000)", no_files.filter(pl.col("total_changes") > 10000)),
    ]
    
    for category_name, category_df in categories:
        print(f"\n{category_name}: {len(category_df):,} PRs")
        print("-" * 80)
        
        if len(category_df) > 0:
            # Sample up to 5 PRs from each category
            sample = category_df.head(5)
            for row in sample.iter_rows(named=True):
                print(f"  Agent: {row['agent']:<10} | "
                      f"State: {row['state']:<8} | "
                      f"Changes: +{row['additions']:>6,} -{row['deletions']:>6,} | "
                      f"changed_files: {row['changed_files']:>4} | "
                      f"URL: {row['url']}")
    
    # Check if changed_files field is populated when files list is empty
    print("\n" + "="*80)
    print("CHANGED_FILES FIELD vs FILES LIST")
    print("="*80)
    
    no_files_with_changed_files = no_files.filter(pl.col("changed_files") > 0)
    no_files_with_zero_changed_files = no_files.filter(pl.col("changed_files") == 0)
    
    print(f"\nOf {len(no_files):,} PRs without file data:")
    print(f"  - Have changed_files > 0: {len(no_files_with_changed_files):,} "
          f"({100*len(no_files_with_changed_files)/len(no_files):.1f}%)")
    print(f"  - Have changed_files = 0: {len(no_files_with_zero_changed_files):,} "
          f"({100*len(no_files_with_zero_changed_files)/len(no_files):.1f}%)")
    
    # More details on changed_files distribution
    print(f"\nchanged_files distribution for PRs without file data:")
    print(f"  Mean: {no_files['changed_files'].mean():.1f}")
    print(f"  Median: {no_files['changed_files'].median():.0f}")
    print(f"  P90: {no_files['changed_files'].quantile(0.90):.0f}")
    print(f"  Max: {no_files['changed_files'].max():.0f}")
    
    # IMPACT ON language_analysis_files.py
    print("\n" + "="*80)
    print("IMPACT ON language_analysis_files.py")
    print("="*80)
    
    print(f"\nCurrently, language_analysis_files.py:")
    print(f"  - Filters out PRs without file data (line 285)")
    print(f"  - This excludes {len(no_files):,} PRs ({100*len(no_files)/len(df):.1f}% of all PRs)")
    print(f"  - These excluded PRs account for {no_files['total_changes'].sum():,} changes")
    print(f"    ({100*no_files['total_changes'].sum()/df['total_changes'].sum():.1f}% of all changes)")
    
    total_prs = len(df)
    prs_in_analysis = df["id"].n_unique()  # Should match total_prs
    
    # The analysis showed earlier: "Unique PRs: 103,736" out of presumably ~120k
    print(f"\nFrom earlier run: language_analysis_files reported 103,736 unique PRs")
    print(f"Total PRs in dataset: {total_prs:,}")
    print(f"Difference: {total_prs - 103736:,} PRs")
    print(f"PRs without file data: {len(no_files):,}")
    print(f"\n→ The difference (8,233 PRs) is larger than just PRs without file data (3,041).")
    print(f"  The additional ~5,192 PRs are likely:")
    print(f"    1. PRs with files but no recognized language extensions")
    print(f"    2. PRs with only non-code files (images, binaries, etc.)")
    
    # Additional investigation
    print("\n" + "="*80)
    print("ADDITIONAL CHECKS")
    print("="*80)
    
    # Check for PRs with files but possibly no recognized languages
    has_files_count = len(has_files)
    print(f"\nPRs with file data: {has_files_count:,}")
    print(f"PRs reported in language_analysis_files: 103,736")
    print(f"Difference: {has_files_count - 103736:,}")
    print(f"\nThis suggests ~{has_files_count - 103736:,} PRs have file data but no recognized language files.")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

