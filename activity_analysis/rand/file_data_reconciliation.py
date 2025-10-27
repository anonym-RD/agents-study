"""Reconcile PR-level vs file-level additions/deletions to understand the gap."""
import polars as pl
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


def main():
    """Reconcile PR-level and file-level change counts."""
    print("="*80)
    print("RECONCILING PR-LEVEL VS FILE-LEVEL CHANGES")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Computing file-level sums per PR...")
    # For each PR, sum up file-level additions/deletions
    pr_with_file_sums = df_lazy.select([
        pl.col("id"),
        pl.col("agent"),
        pl.col("additions").fill_null(0).alias("pr_additions"),
        pl.col("deletions").fill_null(0).alias("pr_deletions"),
        pl.col("files"),
        pl.col("changed_files").alias("changed_files_count"),
    ]).with_columns([
        # Sum up additions/deletions from files list
        pl.col("files").list.eval(
            pl.element().struct.field("additions").fill_null(0)
        ).list.sum().fill_null(0).alias("file_additions_sum"),
        pl.col("files").list.eval(
            pl.element().struct.field("deletions").fill_null(0)
        ).list.sum().fill_null(0).alias("file_deletions_sum"),
        pl.col("files").list.len().fill_null(0).alias("files_in_list"),
    ]).with_columns([
        (pl.col("pr_additions") + pl.col("pr_deletions")).alias("pr_total"),
        (pl.col("file_additions_sum") + pl.col("file_deletions_sum")).alias("file_total"),
        (pl.col("pr_additions") - pl.col("file_additions_sum")).alias("missing_additions"),
        (pl.col("pr_deletions") - pl.col("file_deletions_sum")).alias("missing_deletions"),
    ]).with_columns([
        (pl.col("pr_total") - pl.col("file_total")).alias("missing_total"),
        (pl.col("missing_additions") + pl.col("missing_deletions")).alias("missing_sum"),
    ]).collect()
    
    print("Analyzing discrepancies...")
    
    # Overall stats
    total_prs = len(pr_with_file_sums)
    
    # PRs where file-level matches PR-level (within small tolerance)
    perfect_match = pr_with_file_sums.filter(
        (pl.col("missing_additions").abs() <= 1) & (pl.col("missing_deletions").abs() <= 1)
    )
    
    # PRs with discrepancies
    has_discrepancy = pr_with_file_sums.filter(
        (pl.col("missing_additions").abs() > 1) | (pl.col("missing_deletions").abs() > 1)
    )
    
    print(f"\n{'Metric':<50} {'Count':>15} {'Percentage':>12}")
    print("-" * 80)
    print(f"{'Total PRs:':<50} {total_prs:>15,} {'100.0%':>12}")
    print(f"{'PRs where file-level matches PR-level:':<50} {len(perfect_match):>15,} {100*len(perfect_match)/total_prs:>11.1f}%")
    print(f"{'PRs with discrepancies:':<50} {len(has_discrepancy):>15,} {100*len(has_discrepancy)/total_prs:>11.1f}%")
    
    # Stats on discrepancies
    print("\n" + "="*80)
    print("ANALYZING DISCREPANCIES")
    print("="*80)
    
    discrepancy_stats = has_discrepancy.select([
        pl.col("missing_additions").sum().alias("total_missing_additions"),
        pl.col("missing_deletions").sum().alias("total_missing_deletions"),
        pl.col("missing_total").sum().alias("total_missing_changes"),
        pl.col("files_in_list").mean().alias("avg_files_in_list"),
        pl.col("changed_files_count").mean().alias("avg_changed_files_reported"),
    ])
    
    missing_adds = discrepancy_stats["total_missing_additions"][0]
    missing_dels = discrepancy_stats["total_missing_deletions"][0]
    missing_total = discrepancy_stats["total_missing_changes"][0]
    avg_files_list = discrepancy_stats["avg_files_in_list"][0]
    avg_changed_reported = discrepancy_stats["avg_changed_files_reported"][0]
    
    print(f"\nMissing from file-level data:")
    print(f"  Additions: {missing_adds:,}")
    print(f"  Deletions: {missing_dels:,}")
    print(f"  Total: {missing_total:,}")
    print(f"\nAverage files in list for discrepancy PRs: {avg_files_list:.1f}")
    print(f"Average changed_files reported for discrepancy PRs: {avg_changed_reported:.1f}")
    
    # Check if changed_files_count exceeds files_in_list (API limit indicator)
    print("\n" + "="*80)
    print("CHECKING FOR API LIMITS (changed_files > files_in_list)")
    print("="*80)
    
    likely_truncated = has_discrepancy.filter(
        pl.col("changed_files_count") > pl.col("files_in_list")
    )
    
    print(f"\nPRs where changed_files > files_in_list: {len(likely_truncated):,}")
    print(f"  ({100*len(likely_truncated)/len(has_discrepancy):.1f}% of discrepancy PRs)")
    
    if len(likely_truncated) > 0:
        truncated_missing = likely_truncated.select([
            pl.col("missing_total").sum().alias("missing"),
        ])["missing"][0]
        
        print(f"  Missing changes in these PRs: {truncated_missing:,}")
        print(f"  ({100*truncated_missing/missing_total:.1f}% of all missing changes)")
    
    # Look at distribution of files_in_list for discrepancy PRs
    print("\n" + "="*80)
    print("FILES IN LIST DISTRIBUTION (for discrepancy PRs)")
    print("="*80)
    
    files_dist = has_discrepancy.select([
        pl.col("files_in_list"),
        pl.col("changed_files_count"),
    ])
    
    # Count PRs at exactly 300 files (likely API limit)
    at_300 = files_dist.filter(pl.col("files_in_list") == 300)
    print(f"\nPRs with exactly 300 files in list: {len(at_300):,}")
    print(f"  (GitHub API typically limits to 300 files per PR)")
    
    if len(at_300) > 0:
        at_300_stats = at_300.select([
            pl.col("changed_files_count").mean().alias("avg_changed"),
            pl.col("changed_files_count").max().alias("max_changed"),
        ])
        print(f"  Average changed_files for these PRs: {at_300_stats['avg_changed'][0]:.1f}")
        print(f"  Max changed_files for these PRs: {at_300_stats['max_changed'][0]:,}")
        
        # Missing changes from 300-file PRs
        at_300_from_has_disc = has_discrepancy.filter(pl.col("files_in_list") == 300)
        at_300_missing = at_300_from_has_disc["missing_total"].sum()
        print(f"  Missing changes from 300-file PRs: {at_300_missing:,}")
        print(f"  ({100*at_300_missing/missing_total:.1f}% of all missing changes)")
    
    # Breakdown by agent
    print("\n" + "="*80)
    print("DISCREPANCIES BY AGENT")
    print("="*80)
    
    by_agent = pr_with_file_sums.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("pr_additions").sum().alias("pr_additions"),
        pl.col("file_additions_sum").sum().alias("file_additions"),
        pl.col("missing_additions").sum().alias("missing_additions"),
        ((pl.col("missing_additions").abs() > 1) | (pl.col("missing_deletions").abs() > 1)).sum().alias("prs_with_discrepancy"),
    ]).with_columns([
        (pl.col("file_additions") / pl.col("pr_additions") * 100).alias("coverage_pct"),
        (pl.col("prs_with_discrepancy") / pl.col("total_prs") * 100).alias("discrepancy_rate"),
    ]).sort("agent")
    
    # Sort with Human first
    human_row = by_agent.filter(pl.col("agent") == "Human")
    other_rows = by_agent.filter(pl.col("agent") != "Human").sort("agent")
    by_agent_sorted = pl.concat([human_row, other_rows])
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'PRs w/ Disc':>15} {'Disc Rate':>12} {'PR Adds':>18} {'File Adds':>18} {'Coverage':>10}")
    print("-" * 110)
    
    for row in by_agent_sorted.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['prs_with_discrepancy']:>15,} "
              f"{row['discrepancy_rate']:>11.1f}% {row['pr_additions']:>18,} {row['file_additions']:>18,} {row['coverage_pct']:>9.1f}%")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nConclusion: The file-level data is incomplete due to GitHub API limits")
    print("(typically 300 files per PR). For a complete analysis, we should use")
    print("PR-level additions/deletions totals, not file-level sums.")


if __name__ == "__main__":
    main()

