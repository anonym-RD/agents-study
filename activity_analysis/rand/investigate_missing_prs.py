"""Investigate the ~5K PRs that have file data but don't appear in language analysis."""
import polars as pl
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


def main():
    """Find PRs with file data but no recognized languages."""
    print("="*80)
    print("INVESTIGATING PRs WITH FILES BUT NO RECOGNIZED LANGUAGES")
    print("="*80)
    
    print("\nLoading PR data...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    # Get PRs with file data
    prs_with_files = df_lazy.select([
        pl.col("id").alias("pr_id"),
        pl.col("agent"),
        pl.col("url"),
        pl.col("additions"),
        pl.col("deletions"),
        pl.col("files"),
    ]).filter(
        (pl.col("files").is_not_null()) & (pl.col("files").list.len() > 0)
    ).collect()
    
    print(f"PRs with file data: {len(prs_with_files):,}")
    
    # Explode files to see all file paths
    files_exploded = prs_with_files.select([
        pl.col("pr_id"),
        pl.col("agent"),
        pl.col("files"),
    ]).explode("files").select([
        pl.col("pr_id"),
        pl.col("agent"),
        pl.col("files").struct.field("path").alias("path"),
    ]).filter(pl.col("path").is_not_null())
    
    print(f"Total file records: {len(files_exploded):,}")
    
    # Extract extensions
    files_with_ext = files_exploded.with_columns([
        pl.when(pl.col("path").str.contains(r"\."))
          .then(
              pl.concat_str([
                  pl.lit("."),
                  pl.col("path").str.split("/").list.last().str.split(".").list.last()
              ])
          )
          .otherwise(pl.lit("(no extension)"))
          .alias("extension"),
    ])
    
    # Get unique PRs by whether they have any file with extension
    prs_with_extensions = files_with_ext.filter(
        pl.col("extension") != "(no extension)"
    ).select("pr_id").unique()
    
    prs_only_no_extension = files_with_ext.filter(
        ~pl.col("pr_id").is_in(prs_with_extensions["pr_id"])
    ).select("pr_id").unique()
    
    print(f"\nPRs with at least one file with extension: {len(prs_with_extensions):,}")
    print(f"PRs with ONLY files without extensions: {len(prs_only_no_extension):,}")
    
    # Now check which extensions are NOT in our language mapping
    from spoiler.analysis.language_analysis_files import EXTENSION_TO_LANGUAGE
    
    files_with_ext_info = files_with_ext.with_columns([
        pl.col("extension").is_in(list(EXTENSION_TO_LANGUAGE.keys())).alias("has_language_mapping")
    ])
    
    # Find PRs where NO files have a language mapping
    prs_with_mapping = files_with_ext_info.filter(
        pl.col("has_language_mapping")
    ).select("pr_id").unique()
    
    prs_without_mapping = files_with_ext_info.filter(
        ~pl.col("pr_id").is_in(prs_with_mapping["pr_id"])
    ).select(["pr_id", "agent"]).unique()
    
    print(f"\nPRs with at least one recognized language file: {len(prs_with_mapping):,}")
    print(f"PRs with NO recognized language files: {len(prs_without_mapping):,}")
    
    # This should match the 103,736 from language_analysis_files
    print(f"\n→ Expected PRs in language_analysis_files.py: {len(prs_with_mapping):,}")
    print(f"   (earlier run reported: 103,736)")
    
    # Look at what file extensions these PRs have
    print("\n" + "="*80)
    print("WHAT FILE EXTENSIONS DO THESE PRs HAVE?")
    print("="*80)
    
    # Get files for PRs without language mapping
    files_no_mapping = files_with_ext_info.filter(
        pl.col("pr_id").is_in(prs_without_mapping["pr_id"])
    )
    
    # Count by extension
    ext_counts = files_no_mapping.group_by("extension").agg([
        pl.col("pr_id").n_unique().alias("num_prs"),
        pl.len().alias("num_files")
    ]).sort("num_prs", descending=True)
    
    print(f"\nTop 30 extensions in PRs without recognized languages:")
    print(f"{'Extension':<25} {'PRs':>10} {'Files':>10}")
    print("-" * 50)
    
    for row in ext_counts.head(30).iter_rows(named=True):
        print(f"{row['extension']:<25} {row['num_prs']:>10,} {row['num_files']:>10,}")
    
    # Sample some URLs
    print("\n" + "="*80)
    print("SAMPLE PRs WITHOUT RECOGNIZED LANGUAGES")
    print("="*80)
    
    # Join with original data to get URLs
    sample_prs = prs_without_mapping.head(10)
    sample_with_info = prs_with_files.filter(
        pl.col("pr_id").is_in(sample_prs["pr_id"])
    ).head(10)
    
    print()
    for row in sample_with_info.iter_rows(named=True):
        adds = row['additions'] if row['additions'] is not None else 0
        dels = row['deletions'] if row['deletions'] is not None else 0
        print(f"Agent: {row['agent']:<10} | "
              f"+{adds:>6} -{dels:>6} | "
              f"URL: {row['url']}")
        
        # Show files for this PR
        pr_files = files_with_ext_info.filter(pl.col("pr_id") == row['pr_id'])
        extensions = pr_files["extension"].unique().to_list()[:10]
        print(f"  Extensions: {', '.join(extensions)}")
    
    # Breakdown by agent
    print("\n" + "="*80)
    print("BREAKDOWN BY AGENT")
    print("="*80)
    
    by_agent = prs_without_mapping.group_by("agent").agg([
        pl.len().alias("prs_without_lang")
    ])
    
    total_by_agent = prs_with_files.group_by("agent").agg([
        pl.len().alias("total_prs_with_files")
    ])
    
    comparison = total_by_agent.join(by_agent, on="agent", how="left").with_columns([
        (pl.col("prs_without_lang").fill_null(0) / pl.col("total_prs_with_files") * 100).alias("pct_without_lang")
    ])
    
    # Sort with Human first
    human_row = comparison.filter(pl.col("agent") == "Human")
    other_rows = comparison.filter(pl.col("agent") != "Human").sort("agent")
    comparison_sorted = pl.concat([human_row, other_rows])
    
    print(f"\n{'Agent':<12} {'PRs w/ Files':>15} {'No Recognized Lang':>20} {'%':>8}")
    print("-" * 60)
    
    for row in comparison_sorted.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total_prs_with_files']:>15,} "
              f"{row['prs_without_lang'] or 0:>20,} "
              f"{row['pct_without_lang']:>7.1f}%")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

