"""Analyze programming language distributions based on file-level data.

This analysis examines what fraction of PRs include each language in at least one file.
Unlike language_analysis.py which uses repository primary language, this looks at actual
files changed in the PRs, so percentages won't sum to 100% (a PR can touch multiple languages).
"""
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


# Extension to language mapping
# Based on common file extensions and their associated languages
EXTENSION_TO_LANGUAGE = {
    # Python
    '.py': 'Python',
    '.pyx': 'Python',
    '.pyi': 'Python',
    
    # JavaScript/TypeScript
    '.js': 'JavaScript',
    '.mjs': 'JavaScript',
    '.cjs': 'JavaScript',
    '.jsx': 'JavaScript',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.mts': 'TypeScript',
    '.cts': 'TypeScript',
    
    # Java
    '.java': 'Java',
    
    # C/C++
    '.c': 'C',
    '.h': 'C',
    '.cpp': 'C++',
    '.cc': 'C++',
    '.cxx': 'C++',
    '.hpp': 'C++',
    '.hh': 'C++',
    '.hxx': 'C++',
    
    # C#
    '.cs': 'C#',
    
    # Go
    '.go': 'Go',
    
    # Rust
    '.rs': 'Rust',
    
    # PHP
    '.php': 'PHP',
    '.phtml': 'PHP',
    
    # Ruby
    '.rb': 'Ruby',
    '.erb': 'Ruby',
    
    # Swift
    '.swift': 'Swift',
    
    # Kotlin
    '.kt': 'Kotlin',
    '.kts': 'Kotlin',
    
    # Scala
    '.scala': 'Scala',
    '.sc': 'Scala',
    
    # R
    '.r': 'R',
    '.R': 'R',
    
    # Julia
    '.jl': 'Julia',
    
    # Elixir
    '.ex': 'Elixir',
    '.exs': 'Elixir',
    
    # Erlang
    '.erl': 'Erlang',
    '.hrl': 'Erlang',
    
    # Clojure
    '.clj': 'Clojure',
    '.cljs': 'Clojure',
    '.cljc': 'Clojure',
    
    # Dart
    '.dart': 'Dart',
    
    # Lua
    '.lua': 'Lua',
    
    # Perl
    '.pl': 'Perl',
    '.pm': 'Perl',
    
    # Shell
    '.sh': 'Shell',
    '.bash': 'Shell',
    '.zsh': 'Shell',
    '.fish': 'Shell',
    
    # PowerShell
    '.ps1': 'PowerShell',
    '.psm1': 'PowerShell',
    
    # Groovy
    '.groovy': 'Groovy',
    '.gradle': 'Groovy',
    
    # Vue
    '.vue': 'Vue',
    
    # Svelte
    '.svelte': 'Svelte',
    
    # Solidity
    '.sol': 'Solidity',
    
    # Haskell
    '.hs': 'Haskell',
    
    # Elm
    '.elm': 'Elm',
    
    # OCaml
    '.ml': 'OCaml',
    '.mli': 'OCaml',
    
    # F#
    '.fs': 'F#',
    '.fsx': 'F#',
    '.fsi': 'F#',
    
    # Visual Basic
    '.vb': 'Visual Basic',
    
    # Assembly
    '.asm': 'Assembly',
    '.s': 'Assembly',
    
    # Fortran
    '.f': 'Fortran',
    '.f90': 'Fortran',
    '.f95': 'Fortran',
    
    # Pascal
    '.pas': 'Pascal',
    '.pp': 'Pascal',
    
    # Ada
    '.ada': 'Ada',
    '.adb': 'Ada',
    '.ads': 'Ada',
    
    # Nim
    '.nim': 'Nim',
    
    # Crystal
    '.cr': 'Crystal',
    
    # D
    '.d': 'D',
    
    # Verilog/SystemVerilog
    '.v': 'Verilog',
    '.sv': 'SystemVerilog',
    
    # VHDL
    '.vhd': 'VHDL',
    '.vhdl': 'VHDL',
    
    # Objective-C
    '.m': 'Objective-C',
    '.mm': 'Objective-C++',
    
    # Zig
    '.zig': 'Zig',
    
    # Markdown (for documentation comparison)
    '.md': 'Markdown',
    '.markdown': 'Markdown',
    
    # HTML/CSS
    '.html': 'HTML',
    '.htm': 'HTML',
    '.css': 'CSS',
    '.scss': 'SCSS',
    '.sass': 'Sass',
    '.less': 'Less',
    
    # Config/Data formats
    '.json': 'JSON',
    '.yaml': 'YAML/TOML',
    '.yml': 'YAML/TOML',
    '.toml': 'YAML/TOML',
    '.xml': 'XML',
    '.sql': 'SQL',
}


def extract_extension(path: str) -> str:
    """Extract file extension from a path.
    
    Returns the extension with the dot (e.g., '.py', '.js').
    Returns None for files without an extension.
    """
    if not path:
        return None
    
    # Get the filename from the path
    filename = Path(path).name
    
    # Check if there's a dot and it's not at the start (hidden files)
    if '.' in filename and not filename.startswith('.'):
        # Get everything after the last dot
        ext = '.' + filename.rsplit('.', 1)[-1]
        return ext
    else:
        return None


def extension_to_language(ext: str) -> str:
    """Map file extension to programming language."""
    if not ext:
        return None
    
    # Try exact match first
    if ext in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext]
    
    # Try case-insensitive match
    ext_lower = ext.lower()
    if ext_lower in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext_lower]
    
    return None


def load_pr_file_language_data():
    """Load PR data and extract languages from file paths."""
    print("Loading PR data...")
    prs_lazy = load_lazy_table_for_all_agents(TableNames.PULL_REQUESTS)
    
    print("Extracting file-level data...")
    # Explode the files list so each file becomes a row
    files_df = prs_lazy.select([
        pl.col("id").alias("pr_id"),
        pl.col("agent"),
        pl.col("files"),
    ]).explode("files").select([
        pl.col("pr_id"),
        pl.col("agent"),
        pl.col("files").struct.field("path").alias("path"),
    ])
    
    # Filter out null files (PRs without file data)
    files_df = files_df.filter(pl.col("path").is_not_null())
    
    # Extract extension using Polars string operations
    print("Extracting file extensions...")
    files_df = files_df.with_columns([
        pl.when(pl.col("path").str.contains(r"\."))
          .then(
              pl.concat_str([
                  pl.lit("."),
                  pl.col("path").str.split("/").list.last().str.split(".").list.last()
              ])
          )
          .otherwise(pl.lit(None))
          .alias("extension"),
    ])
    
    # Filter to files with recognized extensions
    files_df = files_df.filter(pl.col("extension").is_not_null())
    
    print("Collecting data for language mapping...")
    files_collected = files_df.collect()
    
    # Map extensions to languages
    print("Mapping extensions to languages...")
    files_with_lang = files_collected.with_columns([
        pl.col("extension").map_elements(extension_to_language, return_dtype=pl.Utf8).alias("language")
    ])
    
    # Filter to files with recognized languages
    files_with_lang = files_with_lang.filter(pl.col("language").is_not_null())
    
    print(f"Found {len(files_with_lang):,} files with recognized languages")
    
    return files_with_lang


def get_top_languages_from_files(df: pl.DataFrame, n: int = 5, combine_js_ts: bool = True):
    """Get the top N most common languages across all agents based on PR presence.
    
    Args:
        df: DataFrame with columns [pr_id, agent, language]
        n: Number of top languages to return
        combine_js_ts: If True, combine TypeScript and JavaScript into "TypeScript/JavaScript"
    
    Returns:
        Tuple of (top_languages list, processed DataFrame)
    """
    print(f"\nIdentifying top {n} languages by PR presence...")
    
    # If combining JS/TS, first merge them
    if combine_js_ts:
        df = df.with_columns([
            pl.when(pl.col("language").is_in(["TypeScript", "JavaScript"]))
              .then(pl.lit("TypeScript/JavaScript"))
              .otherwise(pl.col("language"))
              .alias("language")
        ])
    
    # Count unique PRs per language (a PR can have multiple files in same language)
    lang_pr_counts = df.group_by("language").agg([
        pl.col("pr_id").n_unique().alias("num_prs")
    ]).sort("num_prs", descending=True)
    
    top_langs = lang_pr_counts.head(n)["language"].to_list()
    
    print(f"Top {n} languages: {', '.join(top_langs)}")
    
    # Show counts
    for row in lang_pr_counts.head(n).iter_rows(named=True):
        print(f"  {row['language']}: {row['num_prs']:,} PRs")
    
    return top_langs, df


def prepare_language_presence_data(df: pl.DataFrame, top_languages: list):
    """Prepare data showing what fraction of PRs include each language.
    
    For each agent and language, calculate the percentage of that agent's PRs
    that include at least one file in that language.
    """
    print("\nPreparing language presence data...")
    
    # Add "Other" category for non-top languages
    df_with_other = df.with_columns([
        pl.when(pl.col("language").is_in(top_languages))
          .then(pl.col("language"))
          .otherwise(pl.lit("Other"))
          .alias("language_category")
    ])
    
    # Get unique (pr_id, agent, language_category) combinations
    # This tells us which PRs have at least one file in each language
    pr_language_presence = df_with_other.select([
        "pr_id",
        "agent",
        "language_category"
    ]).unique()
    
    # Count PRs with each language by agent
    lang_pr_counts = pr_language_presence.group_by(["agent", "language_category"]).agg([
        pl.len().alias("pr_count")
    ])
    
    # Get total PRs per agent (from all PRs that have any recognized language files)
    agent_totals = pr_language_presence.group_by("agent").agg([
        pl.col("pr_id").n_unique().alias("total_prs")
    ])
    
    # Join and calculate percentages
    lang_presence = lang_pr_counts.join(
        agent_totals,
        on="agent",
        how="left"
    ).with_columns([
        (pl.col("pr_count") / pl.col("total_prs") * 100).alias("percentage")
    ])
    
    return lang_presence


def get_other_language_details(df: pl.DataFrame, top_languages: list, n_top: int = 4):
    """Get the top N languages that make up 'Other' for each agent.
    
    Returns a function that can be used as detail_generator in plot_faceted_horizontal_bars.
    """
    # Filter to languages not in top_languages
    other_langs = df.filter(~pl.col("language").is_in(top_languages))
    
    # Get unique PR+language combinations for "Other" languages
    other_pr_langs = other_langs.select([
        "pr_id",
        "agent",
        "language"
    ]).unique()
    
    # Get all agents
    agents_order = get_agents_order_reversed(df)
    
    # Get top languages for each agent in the "Other" category
    other_details = {}
    for agent in agents_order:
        # Count PRs with each "other" language for this agent
        agent_langs = other_pr_langs.filter(pl.col("agent") == agent).group_by("language").agg([
            pl.len().alias("count")
        ]).sort("count", descending=True).head(n_top)
        
        if len(agent_langs) > 0:
            # Calculate total "other" PRs for this agent
            total = len(other_pr_langs.filter(pl.col("agent") == agent).select("pr_id").unique())
            if total > 0:
                lang_strs = []
                for row in agent_langs.to_dicts():
                    lang = row['language']
                    pct = (row['count'] / total) * 100
                    lang_strs.append(f"{lang} ({pct:.0f}%)")
                # Format with line break after first 2 languages
                if len(lang_strs) > 2:
                    line1 = ", ".join(lang_strs[:2])
                    line2 = ", ".join(lang_strs[2:]) + ", ..."
                    other_details[agent] = (line1, line2)
                else:
                    other_details[agent] = (", ".join(lang_strs) + ", ...", None)
            else:
                other_details[agent] = ("", None)
        else:
            other_details[agent] = ("", None)
    
    # Return a detail generator function that returns tuple of (line1, line2)
    def detail_generator(category: str, agent: str) -> tuple:
        if category == "Other" and agent in other_details:
            return other_details[agent]
        return ("", None)
    
    return detail_generator


def plot_language_presence_faceted(df: pl.DataFrame, top_languages: list):
    """Create faceted horizontal bar chart of language presence by agent."""
    # Prepare data
    lang_presence = prepare_language_presence_data(df, top_languages)
    
    # Get details about what's in "Other" for each agent
    detail_generator = get_other_language_details(df, top_languages)
    
    # Create ordered list of languages (top languages + Other)
    languages = top_languages + ["Other"]
    
    # Use the reusable plotting function
    plot_faceted_horizontal_bars(
        data=lang_presence,
        categories=languages,
        category_col="language_category",
        agent_col="agent",
        value_col="percentage",
        output_path=plots_path / 'language_presence_by_files',
        xlabel='Percent of PRs with ≥1 file',
        ylabel='Agent',
        figsize_per_category=4.0,
        fig_height=6.0,
        detail_generator=detail_generator,
        value_formatter=lambda v: f'{v:.1f}%',
        xlim_max=None,
    )
    
    plt.close()


def print_language_presence_stats(df: pl.DataFrame, top_languages: list):
    """Print summary statistics about language presence."""
    lang_presence = prepare_language_presence_data(df, top_languages)
    languages = top_languages + ["Other"]
    
    print_distribution_table(
        data=lang_presence,
        categories=languages,
        category_col="language_category",
        agent_col="agent",
        percentage_col="percentage",
        count_col="pr_count",
        title="LANGUAGE PRESENCE STATISTICS (BY FILE-LEVEL ANALYSIS)"
    )
    
    # Additional context
    print("\n" + "="*80)
    print("NOTE: Percentages show the fraction of PRs that include at least one file")
    print("in each language. These percentages do NOT sum to 100% because a single")
    print("PR can include files in multiple languages.")
    print("="*80)


def main():
    """Main entry point for file-based language analysis."""
    print("="*80)
    print("PROGRAMMING LANGUAGE ANALYSIS (FILE-LEVEL)")
    print("="*80)
    
    # Load data
    df = load_pr_file_language_data()
    
    print(f"\nLoaded {len(df):,} file records from PRs")
    print(f"Unique PRs: {df['pr_id'].n_unique():,}")
    
    # Get top languages (combining TypeScript and JavaScript)
    top_languages, df_processed = get_top_languages_from_files(df, n=5, combine_js_ts=True)
    
    # Print statistics
    print_language_presence_stats(df_processed, top_languages)
    
    # Generate plot
    plot_language_presence_faceted(df_processed, top_languages)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"All plots saved to: {plots_path}")
    print("  - language_presence_by_files.png/pdf")


if __name__ == "__main__":
    main()

