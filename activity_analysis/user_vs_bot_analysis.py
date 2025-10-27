"""Analyze the breakdown of User vs Bot engagement in Comments and Reviews."""
import polars as pl
from spoiler.analysis.load_hf_data_polars import load_lazy_table_for_all_agents, TableNames, AgentNames


def analyze_comments():
    """Analyze User vs Bot breakdown in Comments."""
    print("="*80)
    print("COMMENTS ANALYSIS")
    print("="*80)
    
    comments = load_lazy_table_for_all_agents(TableNames.COMMENTS)
    
    # Overall breakdown
    print("\nOverall Comment Breakdown:")
    print("-" * 40)
    
    total = comments.select(pl.len()).collect()[0, 0]
    user_comments = comments.filter(
        pl.col('author').struct.field('typename') == 'User'
    ).select(pl.len()).collect()[0, 0]
    bot_comments = comments.filter(
        pl.col('author').struct.field('typename') == 'Bot'
    ).select(pl.len()).collect()[0, 0]
    
    print(f"  Total comments:  {total:>8,} (100.0%)")
    print(f"  User comments:   {user_comments:>8,} ({user_comments/total*100:>5.1f}%)")
    print(f"  Bot comments:    {bot_comments:>8,} ({bot_comments/total*100:>5.1f}%)")
    
    # Breakdown by agent
    print("\n\nComment Breakdown by Agent:")
    print("-" * 40)
    
    by_agent = comments.group_by("agent").agg([
        pl.len().alias("total"),
        (pl.col('author').struct.field('typename') == 'User').sum().alias("user_count"),
        (pl.col('author').struct.field('typename') == 'Bot').sum().alias("bot_count"),
    ]).with_columns([
        (pl.col("user_count") / pl.col("total") * 100).alias("user_pct"),
        (pl.col("bot_count") / pl.col("total") * 100).alias("bot_pct"),
    ]).sort("agent").collect()
    
    print(f"\n{'Agent':<12} {'Total':>10} {'User':>10} {'User %':>8} {'Bot':>10} {'Bot %':>8}")
    print("-" * 70)
    for row in by_agent.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total']:>10,} {row['user_count']:>10,} "
              f"{row['user_pct']:>7.1f}% {row['bot_count']:>10,} {row['bot_pct']:>7.1f}%")


def analyze_reviews():
    """Analyze User vs Bot breakdown in Reviews."""
    print("\n\n" + "="*80)
    print("REVIEWS ANALYSIS")
    print("="*80)
    
    reviews = load_lazy_table_for_all_agents(TableNames.REVIEWS)
    
    # Overall breakdown
    print("\nOverall Review Breakdown:")
    print("-" * 40)
    
    total = reviews.select(pl.len()).collect()[0, 0]
    user_reviews = reviews.filter(
        pl.col('author').struct.field('typename') == 'User'
    ).select(pl.len()).collect()[0, 0]
    bot_reviews = reviews.filter(
        pl.col('author').struct.field('typename') == 'Bot'
    ).select(pl.len()).collect()[0, 0]
    
    print(f"  Total reviews:   {total:>8,} (100.0%)")
    print(f"  User reviews:    {user_reviews:>8,} ({user_reviews/total*100:>5.1f}%)")
    print(f"  Bot reviews:     {bot_reviews:>8,} ({bot_reviews/total*100:>5.1f}%)")
    
    # Breakdown by agent
    print("\n\nReview Breakdown by Agent:")
    print("-" * 40)
    
    by_agent = reviews.group_by("agent").agg([
        pl.len().alias("total"),
        (pl.col('author').struct.field('typename') == 'User').sum().alias("user_count"),
        (pl.col('author').struct.field('typename') == 'Bot').sum().alias("bot_count"),
    ]).with_columns([
        (pl.col("user_count") / pl.col("total") * 100).alias("user_pct"),
        (pl.col("bot_count") / pl.col("total") * 100).alias("bot_pct"),
    ]).sort("agent").collect()
    
    print(f"\n{'Agent':<12} {'Total':>10} {'User':>10} {'User %':>8} {'Bot':>10} {'Bot %':>8}")
    print("-" * 70)
    for row in by_agent.iter_rows(named=True):
        print(f"{row['agent']:<12} {row['total']:>10,} {row['user_count']:>10,} "
              f"{row['user_pct']:>7.1f}% {row['bot_count']:>10,} {row['bot_pct']:>7.1f}%")


def analyze_pr_level_impact():
    """Analyze how filtering affects PR-level statistics."""
    print("\n\n" + "="*80)
    print("PR-LEVEL IMPACT ANALYSIS")
    print("="*80)
    
    print("\nLoading data...")
    prs = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    comments = load_lazy_table_for_all_agents(TableNames.COMMENTS)
    reviews = load_lazy_table_for_all_agents(TableNames.REVIEWS)
    
    # Count user comments per PR
    user_comments_per_pr = comments.filter(
        pl.col('author').struct.field('typename') == 'User'
    ).group_by(["pr_id", "agent"]).agg([
        pl.len().alias("user_comments_count")
    ])
    
    # Count user reviews per PR
    user_reviews_per_pr = reviews.filter(
        pl.col('author').struct.field('typename') == 'User'
    ).group_by(["pr_id", "agent"]).agg([
        pl.len().alias("user_reviews_count")
    ])
    
    # Join with PRs
    pr_stats = prs.join(
        user_comments_per_pr,
        left_on=["id", "agent"],
        right_on=["pr_id", "agent"],
        how="left"
    ).join(
        user_reviews_per_pr,
        left_on=["id", "agent"],
        right_on=["pr_id", "agent"],
        how="left"
    ).with_columns([
        pl.col("user_comments_count").fill_null(0),
        pl.col("user_reviews_count").fill_null(0),
    ])
    
    # Calculate statistics by agent
    stats = pr_stats.group_by("agent").agg([
        pl.len().alias("total_prs"),
        # Original counts from PR table
        pl.col("comments_count").fill_null(0).mean().alias("mean_all_comments"),
        pl.col("reviews_count").fill_null(0).mean().alias("mean_all_reviews"),
        # User-only counts
        pl.col("user_comments_count").mean().alias("mean_user_comments"),
        pl.col("user_reviews_count").mean().alias("mean_user_reviews"),
        # PRs with any engagement
        (pl.col("comments_count").fill_null(0) > 0).sum().alias("prs_with_any_comments"),
        (pl.col("reviews_count").fill_null(0) > 0).sum().alias("prs_with_any_reviews"),
        (pl.col("user_comments_count") > 0).sum().alias("prs_with_user_comments"),
        (pl.col("user_reviews_count") > 0).sum().alias("prs_with_user_reviews"),
    ]).sort("agent").collect()
    
    print("\nMean Comments per PR (All vs User-only):")
    print("-" * 70)
    print(f"{'Agent':<12} {'All Comments':>14} {'User Comments':>15} {'Difference':>12}")
    print("-" * 70)
    for row in stats.iter_rows(named=True):
        diff = row['mean_all_comments'] - row['mean_user_comments']
        print(f"{row['agent']:<12} {row['mean_all_comments']:>14.2f} "
              f"{row['mean_user_comments']:>15.2f} {diff:>11.2f}")
    
    print("\n\nMean Reviews per PR (All vs User-only):")
    print("-" * 70)
    print(f"{'Agent':<12} {'All Reviews':>14} {'User Reviews':>15} {'Difference':>12}")
    print("-" * 70)
    for row in stats.iter_rows(named=True):
        diff = row['mean_all_reviews'] - row['mean_user_reviews']
        print(f"{row['agent']:<12} {row['mean_all_reviews']:>14.2f} "
              f"{row['mean_user_reviews']:>15.2f} {diff:>11.2f}")
    
    print("\n\n% of PRs with Engagement (All vs User-only):")
    print("-" * 90)
    print(f"{'Agent':<12} {'Any Comments':>13} {'User Comments':>15} "
          f"{'Any Reviews':>13} {'User Reviews':>14}")
    print("-" * 90)
    for row in stats.iter_rows(named=True):
        any_comments_pct = row['prs_with_any_comments'] / row['total_prs'] * 100
        user_comments_pct = row['prs_with_user_comments'] / row['total_prs'] * 100
        any_reviews_pct = row['prs_with_any_reviews'] / row['total_prs'] * 100
        user_reviews_pct = row['prs_with_user_reviews'] / row['total_prs'] * 100
        
        print(f"{row['agent']:<12} {any_comments_pct:>12.1f}% {user_comments_pct:>14.1f}% "
              f"{any_reviews_pct:>12.1f}% {user_reviews_pct:>13.1f}%")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("USER vs BOT ENGAGEMENT ANALYSIS")
    print("="*80)
    print("\nThis script analyzes the breakdown of User vs Bot engagement")
    print("in Comments and Reviews to help determine if filtering is needed.\n")
    
    analyze_comments()
    analyze_reviews()
    
    try:
        analyze_pr_level_impact()
    except OSError as e:
        if "rate limit" in str(e).lower():
            print("\n\n" + "="*80)
            print("NOTE: Hit HuggingFace rate limit for PR-level analysis.")
            print("The key results above are sufficient to make decisions.")
            print("="*80)
        else:
            raise
    
    print("\n\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

