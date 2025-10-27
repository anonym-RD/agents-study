"""Analyze programming language distributions across agents."""
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from spoiler.analysis.load_hf_data_polars import TableNames, load_lazy_table_for_all_agents
from spoiler.util.plotting_faceted_bars import (
    plot_faceted_horizontal_bars,
    print_distribution_table,
    get_agents_order_reversed,
)


cur_path = Path(__file__).parent
root_path = cur_path.parent.parent
plots_path = root_path / "plots" / "language_analysis"

# Create plots directory if it doesn't exist
plots_path.mkdir(parents=True, exist_ok=True)

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def load_pr_repo_language_data():
    """Load and join PR and Repository data, focusing on languages."""
    print("Loading PR data...")
    prs_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Loading Repository data...")
    repos_lazy = load_lazy_table_for_all_agents(TableNames.REPOSITORIES)
    
    # Filter repos to only base repositories
    print("Filtering to BASE repositories...")
    base_repos = repos_lazy.filter(pl.col("role") == "BASE")
    
    # Join PRs with their base repositories
    print("Joining PRs with repositories...")
    pr_repo_data = prs_lazy.join(
        base_repos,
        left_on=["id", "agent"],
        right_on=["pr_id", "agent"],
        how="left"
    )
    
    # Select only needed columns and filter to non-null primary_language
    print("Filtering to PRs with valid primary_language...")
    pr_repo_data = pr_repo_data.select([
        "id",
        "agent",
        "primary_language",
    ]).filter(
        pl.col("primary_language").is_not_null()
    )
    
    print("Collecting data...")
    return pr_repo_data.collect()


def get_top_languages(df: pl.DataFrame, n: int = 5, combine_js_ts: bool = True):
    """Get the top N most common languages across all agents.
    
    Args:
        df: DataFrame with primary_language column
        n: Number of top languages to return
        combine_js_ts: If True, combine TypeScript and JavaScript into "TypeScript/JavaScript"
    """
    print(f"\nIdentifying top {n} languages...")
    
    # If combining JS/TS, first merge them
    if combine_js_ts:
        df = df.with_columns([
            pl.when(pl.col("primary_language").is_in(["TypeScript", "JavaScript"]))
              .then(pl.lit("TypeScript/JavaScript"))
              .otherwise(pl.col("primary_language"))
              .alias("primary_language")
        ])
    
    lang_counts = df.group_by("primary_language").agg([
        pl.len().alias("total_prs")
    ]).sort("total_prs", descending=True)
    
    top_langs = lang_counts.head(n)["primary_language"].to_list()
    
    print(f"Top {n} languages: {', '.join(top_langs)}")
    
    return top_langs, df


def prepare_language_data_with_other(df: pl.DataFrame, top_languages: list):
    """Prepare data with top languages and 'Other' category."""
    print("\nPreparing language distribution data...")
    
    # Create a new column that groups non-top languages as "Other"
    df_with_other = df.with_columns([
        pl.when(pl.col("primary_language").is_in(top_languages))
          .then(pl.col("primary_language"))
          .otherwise(pl.lit("Other"))
          .alias("language_category")
    ])
    
    # Calculate distribution by agent and language category
    lang_dist = df_with_other.group_by(["agent", "language_category"]).agg([
        pl.len().alias("pr_count")
    ])
    
    # Calculate total PRs per agent for percentages
    agent_totals = df_with_other.group_by("agent").agg([
        pl.len().alias("total_prs")
    ])
    
    # Join and calculate percentages
    lang_dist = lang_dist.join(
        agent_totals,
        on="agent",
        how="left"
    ).with_columns([
        (pl.col("pr_count") / pl.col("total_prs") * 100).alias("percentage")
    ])
    
    return lang_dist


def get_other_language_details(df: pl.DataFrame, top_languages: list, n_top: int = 3):
    """Get the top N languages that make up 'Other' for each agent.
    
    Returns a function that can be used as detail_generator in plot_faceted_horizontal_bars.
    """
    # Filter to languages not in top_languages
    other_langs = df.filter(~pl.col("primary_language").is_in(top_languages))
    
    # Get all agents
    agents_order = get_agents_order_reversed(df)
    
    # Get top languages for each agent in the "Other" category
    other_details = {}
    for agent in agents_order:
        agent_langs = other_langs.filter(pl.col("agent") == agent).group_by("primary_language").agg([
            pl.len().alias("count")
        ]).sort("count", descending=True).head(n_top)
        
        if len(agent_langs) > 0:
            # Calculate total for this agent to get percentages
            total = len(other_langs.filter(pl.col("agent") == agent))
            if total > 0:
                lang_strs = []
                for row in agent_langs.to_dicts():
                    lang = row['primary_language']
                    pct = (row['count'] / total) * 100
                    lang_strs.append(f"{lang} ({pct:.0f}%)")
                other_details[agent] = ", ".join(lang_strs) + ", ..."
            else:
                other_details[agent] = ""
        else:
            other_details[agent] = ""
    
    # Return a detail generator function
    def detail_generator(category: str, agent: str) -> str:
        if category == "Other" and agent in other_details:
            return other_details[agent]
        return ""
    
    return detail_generator


def plot_language_distribution_faceted(df: pl.DataFrame, top_languages: list):
    """Create faceted horizontal bar chart of language distribution by agent."""
    # Prepare data
    lang_dist = prepare_language_data_with_other(df, top_languages)
    
    # Get details about what's in "Other" for each agent
    detail_generator = get_other_language_details(df, top_languages)
    
    # Create ordered list of languages (top languages + Other)
    languages = top_languages + ["Other"]
    
    # Use the reusable plotting function
    plot_faceted_horizontal_bars(
        data=lang_dist,
        categories=languages,
        category_col="language_category",
        agent_col="agent",
        value_col="percentage",
        output_path=plots_path / 'language_distribution_faceted',
        xlabel='Percentage of PRs (%)',
        ylabel='Agent',
        figsize_per_category=4.0,
        fig_height=6.0,
        detail_generator=detail_generator,
        value_formatter=lambda v: f'{v:.1f}%',
        xlim_max=None,
    )
    
    plt.close()


def print_language_stats(df: pl.DataFrame, top_languages: list):
    """Print summary statistics about language distributions."""
    lang_dist = prepare_language_data_with_other(df, top_languages)
    languages = top_languages + ["Other"]
    
    print_distribution_table(
        data=lang_dist,
        categories=languages,
        category_col="language_category",
        agent_col="agent",
        percentage_col="percentage",
        count_col="pr_count",
        title="LANGUAGE DISTRIBUTION STATISTICS (BY REPOSITORY PRIMARY LANGUAGE)"
    )


def main():
    """Main entry point for language analysis."""
    print("="*80)
    print("PROGRAMMING LANGUAGE ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_pr_repo_language_data()
    
    print(f"\nLoaded {len(df):,} PRs with primary language information")
    
    # Get top languages (combining TypeScript and JavaScript)
    top_languages, df_processed = get_top_languages(df, n=5, combine_js_ts=True)
    
    # Print statistics
    print_language_stats(df_processed, top_languages)
    
    # Generate plot
    plot_language_distribution_faceted(df_processed, top_languages)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"All plots saved to: {plots_path}")
    print("  - language_distribution_faceted.png/pdf")


if __name__ == "__main__":
    main()

