"""Print total additions, deletions, and total changes across all PRs in the dataset."""
import polars as pl
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


def main():
    """Print summary statistics for total changes across all PRs."""
    print("="*80)
    print("TOTAL CHANGES ACROSS ALL PRS IN DATASET")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    # Impute null additions/deletions with 0 (missing data from GitHub API)
    # and calculate total changes, then collect summary statistics
    summary = df_lazy.select([
        pl.col("additions").fill_null(0).sum().alias("total_additions"),
        pl.col("deletions").fill_null(0).sum().alias("total_deletions"),
        (pl.col("additions").fill_null(0) + pl.col("deletions").fill_null(0)).sum().alias("total_changes"),
        pl.len().alias("total_prs"),
    ]).collect()
    
    # Extract values
    total_additions = summary["total_additions"][0]
    total_deletions = summary["total_deletions"][0]
    total_changes = summary["total_changes"][0]
    total_prs = summary["total_prs"][0]
    
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-" * 52)
    print(f"{'Total PRs:':<30} {total_prs:>20,}")
    print(f"{'Total Lines Added:':<30} {total_additions:>20,}")
    print(f"{'Total Lines Deleted:':<30} {total_deletions:>20,}")
    print(f"{'Total Lines Changed:':<30} {total_changes:>20,}")
    print()
    
    # Also show breakdown by agent
    print("\n" + "="*80)
    print("BREAKDOWN BY AGENT")
    print("="*80)
    
    by_agent = df_lazy.group_by("agent").agg([
        pl.len().alias("total_prs"),
        pl.col("additions").fill_null(0).sum().alias("total_additions"),
        pl.col("deletions").fill_null(0).sum().alias("total_deletions"),
        (pl.col("additions").fill_null(0) + pl.col("deletions").fill_null(0)).sum().alias("total_changes"),
    ]).sort("agent").collect()
    
    # Sort with Human first
    human_row = by_agent.filter(pl.col("agent") == "Human")
    other_rows = by_agent.filter(pl.col("agent") != "Human").sort("agent")
    by_agent = pl.concat([human_row, other_rows])
    
    print(f"\n{'Agent':<12} {'PRs':>10} {'Additions':>15} {'Deletions':>15} {'Total Changes':>15}")
    print("-" * 70)
    
    for row in by_agent.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs']:>10,} {row['total_additions']:>15,} "
              f"{row['total_deletions']:>15,} {row['total_changes']:>15,}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

