"""Analyze additions, deletions, and total changes by file extension."""
import polars as pl
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents


def extract_extension(path: str) -> str:
    """Extract file extension from a path.
    
    Returns the extension with the dot (e.g., '.py', '.js').
    Returns '(no extension)' for files without an extension.
    """
    if not path:
        return "(no extension)"
    
    # Get the filename from the path
    filename = Path(path).name
    
    # Check if there's a dot and it's not at the start (hidden files)
    if '.' in filename and not filename.startswith('.'):
        # Get everything after the last dot
        ext = '.' + filename.rsplit('.', 1)[-1]
        return ext
    else:
        return "(no extension)"


def main():
    """Analyze changes by file extension."""
    print("="*80)
    print("ANALYZING CHANGES BY FILE TYPE (EXTENSION)")
    print("="*80)
    
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Extracting file-level data...")
    # Explode the files list so each file becomes a row
    files_df = df_lazy.select([
        pl.col("id").alias("pr_id"),
        pl.col("agent"),
        pl.col("files"),
    ]).explode("files").select([
        pl.col("pr_id"),
        pl.col("agent"),
        pl.col("files").struct.field("path").alias("path"),
        pl.col("files").struct.field("additions").alias("additions"),
        pl.col("files").struct.field("deletions").alias("deletions"),
        pl.col("files").struct.field("change_type").alias("change_type"),
    ])
    
    # Filter out null files (PRs without file data)
    files_df = files_df.filter(pl.col("path").is_not_null())
    
    # Extract extension and calculate totals
    print("Extracting file extensions and calculating totals...")
    files_df = files_df.with_columns([
        # Extract extension using Polars string operations
        pl.when(pl.col("path").str.contains(r"\."))
          .then(
              pl.concat_str([
                  pl.lit("."),
                  pl.col("path").str.split("/").list.last().str.split(".").list.last()
              ])
          )
          .otherwise(pl.lit("(no extension)"))
          .alias("extension"),
        (pl.col("additions").fill_null(0) + pl.col("deletions").fill_null(0)).alias("total_changes"),
    ])
    
    # Aggregate by extension
    print("Aggregating by file extension...")
    by_extension = files_df.group_by("extension").agg([
        pl.len().alias("num_files"),
        pl.col("additions").fill_null(0).sum().alias("total_additions"),
        pl.col("deletions").fill_null(0).sum().alias("total_deletions"),
        pl.col("total_changes").sum().alias("total_changes"),
    ]).sort("total_changes", descending=True).collect()
    
    # Calculate overall totals for percentage
    overall_totals = by_extension.select([
        pl.col("num_files").sum().alias("total_files"),
        pl.col("total_additions").sum().alias("total_additions"),
        pl.col("total_deletions").sum().alias("total_deletions"),
        pl.col("total_changes").sum().alias("total_changes"),
    ])
    
    overall_files = overall_totals["total_files"][0]
    overall_additions = overall_totals["total_additions"][0]
    overall_deletions = overall_totals["total_deletions"][0]
    overall_changes = overall_totals["total_changes"][0]
    
    print(f"\n{'Extension':<20} {'Files':>12} {'Files %':>10} {'Additions':>15} {'Deletions':>15} {'Total Changes':>15} {'Changes %':>10}")
    print("-" * 120)
    
    # Print top 50 extensions
    for i, row in enumerate(by_extension.iter_rows(named=True)):
        if i >= 50:
            break
        
        ext = row['extension']
        num_files = row['num_files']
        additions = row['total_additions']
        deletions = row['total_deletions']
        changes = row['total_changes']
        
        files_pct = 100 * num_files / overall_files if overall_files > 0 else 0
        changes_pct = 100 * changes / overall_changes if overall_changes > 0 else 0
        
        print(f"{ext:<20} {num_files:>12,} {files_pct:>9.2f}% {additions:>15,} {deletions:>15,} {changes:>15,} {changes_pct:>9.2f}%")
    
    # Summary
    print("\n" + "="*120)
    print(f"{'TOTALS':<20} {overall_files:>12,} {'100.00%':>10} {overall_additions:>15,} {overall_deletions:>15,} {overall_changes:>15,} {'100.00%':>10}")
    print("="*120)
    
    # Also show breakdown by agent for top extensions
    print("\n" + "="*80)
    print("TOP 10 EXTENSIONS BY AGENT")
    print("="*80)
    
    # Get top 10 extensions
    top_10_extensions = by_extension.head(10)["extension"].to_list()
    
    for ext in top_10_extensions:
        print(f"\n{ext}:")
        print(f"{'Agent':<12} {'Files':>10} {'Additions':>15} {'Deletions':>15} {'Total Changes':>15}")
        print("-" * 70)
        
        by_agent = files_df.filter(pl.col("extension") == ext).group_by("agent").agg([
            pl.len().alias("num_files"),
            pl.col("additions").fill_null(0).sum().alias("total_additions"),
            pl.col("deletions").fill_null(0).sum().alias("total_deletions"),
            pl.col("total_changes").sum().alias("total_changes"),
        ]).sort("agent").collect()
        
        # Sort with Human first
        human_row = by_agent.filter(pl.col("agent") == "Human")
        other_rows = by_agent.filter(pl.col("agent") != "Human").sort("agent")
        by_agent_sorted = pl.concat([human_row, other_rows]) if len(human_row) > 0 else other_rows
        
        for row in by_agent_sorted.iter_rows(named=True):
            print(f"{row['agent']:<12} {row['num_files']:>10,} {row['total_additions']:>15,} "
                  f"{row['total_deletions']:>15,} {row['total_changes']:>15,}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

