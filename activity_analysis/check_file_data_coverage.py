"""Check how many PRs have file-level data vs PR-level totals."""
import polars as pl
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


def main():
    """Check data coverage for file-level information."""
    print("="*80)
    print("CHECKING FILE-LEVEL DATA COVERAGE")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    # Check file data coverage
    print("\nAnalyzing file data coverage...")
    coverage = df_lazy.select([
        pl.col("id"),
        pl.col("agent"),
        pl.col("additions").fill_null(0).alias("pr_additions"),
        pl.col("deletions").fill_null(0).alias("pr_deletions"),
        pl.col("files"),
    ]).with_columns([
        # Check if files list exists and is not empty
        pl.col("files").is_not_null().alias("has_files_field"),
        pl.col("files").list.len().alias("files_count"),
        (pl.col("files").list.len() > 0).fill_null(False).alias("has_files_data"),
    ]).collect()
    
    # Overall stats
    total_prs = len(coverage)
    has_files_field = coverage.filter(pl.col("has_files_field")).height
    has_files_data = coverage.filter(pl.col("has_files_data")).height
    
    print(f"\n{'Metric':<40} {'Count':>15} {'Percentage':>12}")
    print("-" * 70)
    print(f"{'Total PRs:':<40} {total_prs:>15,} {'100.0%':>12}")
    print(f"{'PRs with files field:':<40} {has_files_field:>15,} {100*has_files_field/total_prs:>11.1f}%")
    print(f"{'PRs with non-empty files data:':<40} {has_files_data:>15,} {100*has_files_data/total_prs:>11.1f}%")
    
    # Compare PR-level vs file-level totals
    print("\n" + "="*80)
    print("PR-LEVEL vs FILE-LEVEL TOTALS")
    print("="*80)
    
    # PR-level totals (from additions/deletions fields)
    pr_level = coverage.select([
        pl.col("pr_additions").sum().alias("pr_additions"),
        pl.col("pr_deletions").sum().alias("pr_deletions"),
    ])
    
    pr_additions = pr_level["pr_additions"][0]
    pr_deletions = pr_level["pr_deletions"][0]
    pr_total = pr_additions + pr_deletions
    
    # File-level totals (sum from files list)
    # First explode files and sum
    file_level = coverage.filter(pl.col("has_files_data")).select([
        pl.col("id"),
        pl.col("files"),
    ]).explode("files").select([
        pl.col("files").struct.field("additions").fill_null(0).sum().alias("file_additions"),
        pl.col("files").struct.field("deletions").fill_null(0).sum().alias("file_deletions"),
    ])
    
    file_additions = file_level["file_additions"][0]
    file_deletions = file_level["file_deletions"][0]
    file_total = file_additions + file_deletions
    
    print(f"\n{'Source':<20} {'Additions':>20} {'Deletions':>20} {'Total':>20}")
    print("-" * 85)
    print(f"{'PR-level (all PRs):':<20} {pr_additions:>20,} {pr_deletions:>20,} {pr_total:>20,}")
    print(f"{'File-level sum:':<20} {file_additions:>20,} {file_additions:>20,} {file_total:>20,}")
    print(f"{'Missing/Difference:':<20} {pr_additions-file_additions:>20,} {pr_deletions-file_deletions:>20,} {pr_total-file_total:>20,}")
    print()
    print(f"File-level coverage: {100*file_total/pr_total:.1f}% of PR-level total")
    
    # Breakdown by agent
    print("\n" + "="*80)
    print("COVERAGE BY AGENT")
    print("="*80)
    
    by_agent = coverage.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("has_files_data").sum().alias("prs_with_files"),
        pl.col("pr_additions").sum().alias("pr_additions"),
        pl.col("pr_deletions").sum().alias("pr_deletions"),
    ]).with_columns([
        (pl.col("prs_with_files") / pl.col("total_prs") * 100).alias("coverage_pct"),
        (pl.col("pr_additions") + pl.col("pr_deletions")).alias("pr_total_changes"),
    ]).sort("agent")
    
    # Sort with Human first
    human_row = by_agent.filter(pl.col("agent") == "Human")
    other_rows = by_agent.filter(pl.col("agent") != "Human").sort("agent")
    by_agent_sorted = pl.concat([human_row, other_rows])
    
    print(f"\n{'Agent':<12} {'Total PRs':>12} {'PRs w/ Files':>15} {'Coverage %':>12} {'PR-Level Adds':>18} {'PR-Level Total':>18}")
    print("-" * 100)
    
    for row in by_agent_sorted.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>12,} {row['prs_with_files']:>15,} "
              f"{row['coverage_pct']:>11.1f}% {row['pr_additions']:>18,} {row['pr_total_changes']:>18,}")
    
    # Check if PRs without file data still have additions/deletions
    print("\n" + "="*80)
    print("PRs WITHOUT FILE DATA - DO THEY HAVE PR-LEVEL ADDITIONS/DELETIONS?")
    print("="*80)
    
    no_files = coverage.filter(~pl.col("has_files_data"))
    
    no_files_with_changes = no_files.filter(
        (pl.col("pr_additions") > 0) | (pl.col("pr_deletions") > 0)
    )
    
    no_files_adds = no_files["pr_additions"].sum()
    no_files_dels = no_files["pr_deletions"].sum()
    no_files_total = no_files_adds + no_files_dels
    
    print(f"\nPRs without file data: {len(no_files):,}")
    print(f"  - With non-zero changes: {len(no_files_with_changes):,}")
    print(f"  - Total additions: {no_files_adds:,}")
    print(f"  - Total deletions: {no_files_dels:,}")
    print(f"  - Total changes: {no_files_total:,}")
    print(f"\nThese PRs account for {100*no_files_total/pr_total:.1f}% of all changes!")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

