"""Explore PRs that were merged quickly for each agent."""
import polars as pl
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents, AgentNames


cur_path = Path(__file__).parent
root_path = cur_path.parent.parent


def prepare_pr_data(df: pl.LazyFrame) -> pl.DataFrame:
    """Prepare PR data by adding computed columns."""
    return df.with_columns([
        # Impute null additions/deletions with 0 (missing data from GitHub API)
        pl.col("additions").fill_null(0),
        pl.col("deletions").fill_null(0),
        
        # Parse datetime columns
        pl.col("created_at").str.to_datetime(),
        pl.col("merged_at").str.to_datetime(),
        
        # Merge status
        pl.col("merged_at").is_not_null().alias("is_merged"),
        
        # Time to merge (in hours and minutes)
        ((pl.col("merged_at").str.to_datetime() - pl.col("created_at").str.to_datetime())
            .dt.total_seconds() / 3600.0)
            .alias("time_to_merge_hours"),
        
        ((pl.col("merged_at").str.to_datetime() - pl.col("created_at").str.to_datetime())
            .dt.total_seconds() / 60.0)
            .alias("time_to_merge_minutes"),
        
        # Total changes
        (pl.col("additions") + pl.col("deletions")).alias("total_changes"),
    ]).collect()


def format_time(minutes):
    """Format time in a human-readable way."""
    if minutes < 1:
        seconds = minutes * 60
        return f"{seconds:.0f}s"
    elif minutes < 60:
        return f"{minutes:.1f}min"
    else:
        hours = minutes / 60
        if hours < 24:
            return f"{hours:.1f}h"
        else:
            days = hours / 24
            return f"{days:.1f}d"


def explore_quick_merges(threshold_minutes=10, num_examples=10):
    """
    Explore PRs that were merged quickly.
    
    Args:
        threshold_minutes: Time threshold in minutes to consider a PR "quick"
        num_examples: Number of examples to show per agent
    """
    print("="*80)
    print(f"EXPLORING PRs MERGED IN UNDER {threshold_minutes} MINUTES")
    print("="*80)
    
    # Load PR data
    print("\nLoading PR data from HuggingFace dataset...")
    df_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    df = prepare_pr_data(df_lazy)
    
    # Filter to merged PRs only
    merged_df = df.filter(pl.col("is_merged"))
    
    # Filter to quick merges
    quick_merges = merged_df.filter(pl.col("time_to_merge_minutes") < threshold_minutes)
    
    print(f"Total merged PRs: {len(merged_df):,}")
    print(f"Quick merges (< {threshold_minutes} min): {len(quick_merges):,}")
    print(f"Percentage: {len(quick_merges) / len(merged_df) * 100:.1f}%\n")
    
    # Get agents sorted with Human first
    agents = ['Human'] + sorted([a for a in merged_df['agent'].unique().to_list() if a != 'Human'])
    
    for agent in agents:
        print("\n" + "="*80)
        print(f"AGENT: {agent}")
        print("="*80)
        
        agent_quick = quick_merges.filter(pl.col("agent") == agent)
        agent_all_merged = merged_df.filter(pl.col("agent") == agent)
        
        if len(agent_quick) == 0:
            print(f"No PRs merged under {threshold_minutes} minutes for {agent}")
            continue
        
        pct = len(agent_quick) / len(agent_all_merged) * 100
        print(f"Quick merges: {len(agent_quick):,} / {len(agent_all_merged):,} ({pct:.1f}%)")
        
        # Get statistics
        median_time = agent_quick['time_to_merge_minutes'].median()
        mean_time = agent_quick['time_to_merge_minutes'].mean()
        min_time = agent_quick['time_to_merge_minutes'].min()
        max_time = agent_quick['time_to_merge_minutes'].max()
        
        print(f"Time stats for quick merges:")
        print(f"  Min: {format_time(min_time)}")
        print(f"  Median: {format_time(median_time)}")
        print(f"  Mean: {format_time(mean_time)}")
        print(f"  Max: {format_time(max_time)}")
        
        # Sort by time to merge and get examples
        agent_quick_sorted = agent_quick.sort("time_to_merge_minutes")
        
        # Get a mix: fastest ones and some random ones
        fastest = agent_quick_sorted.head(num_examples // 2)
        # Get some from the middle/upper range too
        if len(agent_quick_sorted) > num_examples:
            sample_indices = pl.int_range(num_examples // 2, len(agent_quick_sorted), eager=True)
            sample_indices = sample_indices.sample(n=min(num_examples // 2, len(sample_indices)), seed=42)
            other_samples = agent_quick_sorted[sample_indices]
            examples = pl.concat([fastest, other_samples]).sort("time_to_merge_minutes")
        else:
            examples = fastest
        
        print(f"\n{len(examples)} Example PRs:")
        print("-" * 80)
        
        for i, row in enumerate(examples.iter_rows(named=True), 1):
            time_str = format_time(row['time_to_merge_minutes'])
            changes = row['total_changes'] if row['total_changes'] is not None else 0
            title = row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
            url = row['url']
            
            print(f"\n{i}. Time: {time_str:>8} | Changes: {changes:>4} lines")
            print(f"   Title: {title}")
            print(f"   URL: {url}")
            
            # Show body preview if available
            if row['body']:
                body_preview = row['body'][:100].replace('\n', ' ')
                if len(row['body']) > 100:
                    body_preview += "..."
                print(f"   Body: {body_preview}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)


def main():
    """Main entry point."""
    explore_quick_merges(threshold_minutes=1, num_examples=10)


if __name__ == "__main__":
    main()

