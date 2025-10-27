"""Quick analysis of Reviews data from HuggingFace dataset."""
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents, AgentNames
import polars as pl


def sort_agents_human_first(df: pl.DataFrame) -> pl.DataFrame:
    """Sort a dataframe so Human comes first, then others alphabetically."""
    human_row = df.filter(pl.col("agent") == "Human")
    other_rows = df.filter(pl.col("agent") != "Human").sort("agent")
    return pl.concat([human_row, other_rows])


def prepare_reviews_data(df: pl.LazyFrame) -> pl.DataFrame:
    """Prepare reviews data by adding computed columns."""
    return df.with_columns([
        # Body length and presence
        pl.col("body").str.len_chars().fill_null(0).alias("body_length"),
        pl.col("body").is_not_null().alias("has_body"),
        pl.col("body").fill_null("").str.len_chars().gt(0).alias("has_non_empty_body"),
        
        # Parse datetime columns
        pl.col("created_at").str.to_datetime(),
        pl.col("submitted_at").str.to_datetime(),
        pl.col("published_at").str.to_datetime(),
    ]).collect()


def print_overall_stats(df: pl.DataFrame):
    """Print overall statistics across all agents."""
    print("=" * 80)
    print("REVIEWS TABLE - OVERALL STATISTICS")
    print("=" * 80)
    
    total_reviews = len(df)
    print(f"\nTotal Reviews: {total_reviews:,}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    
    # Agent distribution
    print("\n" + "=" * 60)
    print("Review Count by Agent")
    print("=" * 60)
    agent_counts = df.group_by("agent").agg(
        pl.len().alias("count")
    )
    agent_counts = sort_agents_human_first(agent_counts)
    
    for row in agent_counts.iter_rows(named=True):
        pct = (row['count'] / total_reviews) * 100
        print(f"  {row['agent']:10s}: {row['count']:6,} ({pct:5.1f}%)")


def analyze_review_states(df: pl.DataFrame):
    """Analyze review states (APPROVED, CHANGES_REQUESTED, COMMENTED, etc.)."""
    print("\n" + "=" * 80)
    print("1. REVIEW STATE ANALYSIS")
    print("=" * 80)
    
    # Overall state distribution
    print("\nOverall Review States:")
    state_counts = df.group_by("state").agg(
        pl.len().alias("count")
    ).sort("count", descending=True)
    
    total = len(df)
    for row in state_counts.iter_rows(named=True):
        pct = (row['count'] / total) * 100
        print(f"  {row['state']:20s}: {row['count']:8,} ({pct:5.1f}%)")
    
    # State distribution by agent
    print("\n" + "-" * 80)
    print("Review States by Agent:")
    print("-" * 80)
    
    state_by_agent = df.group_by(["agent", "state"]).agg(
        pl.len().alias("count")
    ).sort(["agent", "count"], descending=[False, True])
    
    # Pivot to show states as columns
    states_pivot = df.group_by("agent").agg([
        pl.len().alias("total"),
        (pl.col("state") == "APPROVED").sum().alias("approved"),
        (pl.col("state") == "CHANGES_REQUESTED").sum().alias("changes_requested"),
        (pl.col("state") == "COMMENTED").sum().alias("commented"),
        (pl.col("state") == "DISMISSED").sum().alias("dismissed"),
        (pl.col("state") == "PENDING").sum().alias("pending"),
    ])
    
    # Add percentages
    states_pivot = states_pivot.with_columns([
        (pl.col("approved") / pl.col("total") * 100).alias("approved_pct"),
        (pl.col("changes_requested") / pl.col("total") * 100).alias("changes_requested_pct"),
        (pl.col("commented") / pl.col("total") * 100).alias("commented_pct"),
        (pl.col("dismissed") / pl.col("total") * 100).alias("dismissed_pct"),
        (pl.col("pending") / pl.col("total") * 100).alias("pending_pct"),
    ])
    
    states_pivot = sort_agents_human_first(states_pivot)
    
    print(f"\n{'Agent':<10} {'Total':>8} {'Approved':>10} {'%':>6} {'Changes':>10} {'%':>6} "
          f"{'Commented':>10} {'%':>6} {'Dismissed':>10} {'%':>6}")
    print("-" * 100)
    
    for row in states_pivot.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total']:>8,} {row['approved']:>10,} {row['approved_pct']:>5.1f}% "
              f"{row['changes_requested']:>10,} {row['changes_requested_pct']:>5.1f}% "
              f"{row['commented']:>10,} {row['commented_pct']:>5.1f}% "
              f"{row['dismissed']:>10,} {row['dismissed_pct']:>5.1f}%")


def analyze_review_authors(df: pl.DataFrame):
    """Analyze review authors."""
    print("\n" + "=" * 80)
    print("2. REVIEW AUTHOR ANALYSIS")
    print("=" * 80)
    
    # Extract author login from the struct
    df_with_author = df.with_columns([
        pl.col("author").struct.field("login").alias("author_login"),
        pl.col("author").struct.field("typename").alias("author_type"),
    ])
    
    # Overall author distribution
    print("\nTop 20 Most Active Reviewers (Overall):")
    top_authors = df_with_author.group_by("author_login").agg(
        pl.len().alias("review_count")
    ).sort("review_count", descending=True).head(20)
    
    for i, row in enumerate(top_authors.iter_rows(named=True), 1):
        print(f"  {i:2d}. {row['author_login']:30s}: {row['review_count']:6,} reviews")
    
    # Unique reviewers by agent
    print("\n" + "-" * 80)
    print("Unique Reviewers by Agent:")
    print("-" * 80)
    
    unique_reviewers = df_with_author.group_by("agent").agg([
        pl.len().alias("total_reviews"),
        pl.col("author_login").n_unique().alias("unique_reviewers"),
    ])
    
    unique_reviewers = unique_reviewers.with_columns([
        (pl.col("total_reviews") / pl.col("unique_reviewers")).alias("reviews_per_reviewer")
    ])
    
    unique_reviewers = sort_agents_human_first(unique_reviewers)
    
    print(f"{'Agent':<10} {'Total Reviews':>14} {'Unique Reviewers':>18} {'Reviews/Reviewer':>18}")
    print("-" * 70)
    
    for row in unique_reviewers.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total_reviews']:>14,} {row['unique_reviewers']:>18,} "
              f"{row['reviews_per_reviewer']:>18.2f}")
    
    # Author type distribution
    print("\n" + "-" * 80)
    print("Author Type Distribution:")
    print("-" * 80)
    
    author_types = df_with_author.group_by("author_type").agg(
        pl.len().alias("count")
    ).sort("count", descending=True)
    
    total = len(df_with_author)
    for row in author_types.iter_rows(named=True):
        pct = (row['count'] / total) * 100
        print(f"  {row['author_type']:20s}: {row['count']:8,} ({pct:5.1f}%)")


def analyze_review_body(df: pl.DataFrame):
    """Analyze review body content."""
    print("\n" + "=" * 80)
    print("3. REVIEW BODY ANALYSIS")
    print("=" * 80)
    
    body_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("has_non_empty_body").sum().alias("has_body_count"),
        pl.col("body_length").mean().alias("mean_length"),
        pl.col("body_length").median().alias("median_length"),
        pl.col("body_length").max().alias("max_length"),
        pl.col("body_length").std().alias("std_length"),
    ])
    
    body_stats = body_stats.with_columns([
        (pl.col("has_body_count") / pl.col("total") * 100).alias("has_body_pct")
    ])
    
    body_stats = sort_agents_human_first(body_stats)
    
    print("\nReview Body Statistics by Agent:")
    print(f"{'Agent':<10} {'Total':>8} {'Has Body':>10} {'%':>6} {'Mean Len':>10} "
          f"{'Median Len':>11} {'Max Len':>10}")
    print("-" * 85)
    
    for row in body_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total']:>8,} {row['has_body_count']:>10,} "
              f"{row['has_body_pct']:>5.1f}% {row['mean_length']:>10.0f} "
              f"{row['median_length']:>11.0f} {row['max_length']:>10,}")


def analyze_reviews_per_pr(df: pl.DataFrame):
    """Analyze how many reviews each PR receives."""
    print("\n" + "=" * 80)
    print("4. REVIEWS PER PR ANALYSIS")
    print("=" * 80)
    
    # Count reviews per PR
    reviews_per_pr = df.group_by(["agent", "pr_id"]).agg(
        pl.len().alias("review_count")
    )
    
    # Get statistics on reviews per PR by agent
    pr_stats = reviews_per_pr.group_by("agent").agg([
        pl.len().alias("total_prs_with_reviews"),
        pl.col("review_count").mean().alias("mean_reviews_per_pr"),
        pl.col("review_count").median().alias("median_reviews_per_pr"),
        pl.col("review_count").min().alias("min_reviews_per_pr"),
        pl.col("review_count").max().alias("max_reviews_per_pr"),
        pl.col("review_count").std().alias("std_reviews_per_pr"),
    ])
    
    pr_stats = sort_agents_human_first(pr_stats)
    
    print("\nReviews per PR Statistics by Agent:")
    print(f"{'Agent':<10} {'PRs w/ Reviews':>15} {'Mean':>8} {'Median':>8} {'Min':>6} {'Max':>6} {'Std':>8}")
    print("-" * 80)
    
    for row in pr_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total_prs_with_reviews']:>15,} "
              f"{row['mean_reviews_per_pr']:>8.2f} {row['median_reviews_per_pr']:>8.0f} "
              f"{row['min_reviews_per_pr']:>6,} {row['max_reviews_per_pr']:>6,} "
              f"{row['std_reviews_per_pr']:>8.2f}")
    
    # Distribution of review counts
    print("\n" + "-" * 80)
    print("Distribution of Review Counts per PR (All Agents):")
    print("-" * 80)
    
    review_count_dist = reviews_per_pr.group_by("review_count").agg(
        pl.len().alias("pr_count")
    ).sort("review_count").head(20)
    
    total_prs = len(reviews_per_pr)
    for row in review_count_dist.iter_rows(named=True):
        pct = (row['pr_count'] / total_prs) * 100
        bar = "█" * int(pct)
        print(f"  {row['review_count']:3d} reviews: {row['pr_count']:6,} PRs ({pct:5.1f}%) {bar}")


def analyze_minimized_reviews(df: pl.DataFrame):
    """Analyze minimized reviews."""
    print("\n" + "=" * 80)
    print("5. MINIMIZED REVIEWS ANALYSIS")
    print("=" * 80)
    
    minimized_stats = df.group_by("agent").agg([
        pl.len().alias("total"),
        pl.col("is_minimized").sum().alias("minimized_count"),
    ])
    
    minimized_stats = minimized_stats.with_columns([
        (pl.col("minimized_count") / pl.col("total") * 100).alias("minimized_pct")
    ])
    
    minimized_stats = sort_agents_human_first(minimized_stats)
    
    print("\nMinimized Reviews by Agent:")
    print(f"{'Agent':<10} {'Total':>8} {'Minimized':>11} {'%':>6}")
    print("-" * 50)
    
    for row in minimized_stats.iter_rows(named=True):
        print(f"{row['agent']:<10} {row['total']:>8,} {row['minimized_count']:>11,} "
              f"{row['minimized_pct']:>5.2f}%")
    
    # Minimized reasons (if any are minimized)
    if df["is_minimized"].sum() > 0:
        print("\n" + "-" * 80)
        print("Minimized Reasons:")
        print("-" * 80)
        
        reasons = df.filter(pl.col("is_minimized")).group_by("minimized_reason").agg(
            pl.len().alias("count")
        ).sort("count", descending=True)
        
        total_minimized = df["is_minimized"].sum()
        for row in reasons.iter_rows(named=True):
            pct = (row['count'] / total_minimized) * 100
            reason = row['minimized_reason'] if row['minimized_reason'] else "(null)"
            print(f"  {reason:30s}: {row['count']:6,} ({pct:5.1f}%)")


def main():
    """Main entry point for the reviews analysis script."""
    print("Loading Reviews data from HuggingFace dataset...")
    
    # Load data for all agents
    df_lazy = load_lazy_table_for_all_agents(TableNames.REVIEWS)
    
    # Prepare data (compute derived columns and collect)
    print("Preparing data...")
    df = prepare_reviews_data(df_lazy)
    
    print(f"Loaded {len(df):,} reviews\n")
    
    # Run analyses
    print_overall_stats(df)
    analyze_review_states(df)
    analyze_review_authors(df)
    analyze_review_body(df)
    analyze_reviews_per_pr(df)
    analyze_minimized_reviews(df)
    
    print("\n" + "=" * 80)
    print("REVIEWS ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()